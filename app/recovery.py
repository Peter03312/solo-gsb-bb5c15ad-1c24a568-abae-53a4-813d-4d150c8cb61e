"""展柜扰动恢复分析（disturbance recovery，纯计算，只读）。

以**事件时刻** ``event_ts`` 为界：

- **基线**取事件前窗口 ``[event_ts − baseline_seconds, event_ts]``（含两端）
  内读数的温度、湿度**中位数**（偶数样本取中间两数均值）；窗口内读数
  少于 ``min_baseline_samples`` 时抛 :class:`InsufficientBaselineSamples`；
- 恰好落在 ``event_ts`` 的读数**仍计入基线**，同时作为**事件后首段曲线的
  左端点**——它与窗口内首条严格晚于事件的读数构成第一条线性线段；
- **事件后**取窗口 ``(event_ts, event_ts + horizon_seconds]`` 内的读数，
  连同事件时刻读数（若存在）视为**分段线性**曲线，对温度、湿度两个维度
  分别精确求出进入/离开容差带 ``[中位数 − 容差, 中位数 + 容差]``
  （**含边界**）的交点；首段两点间隔大于 ``max_gap_seconds`` 时该段断开；
- 仅当两维**同时**在带内、且相邻读数间隔不超过 ``max_gap_seconds``
  （恰好等于仍连续）地连续保持 **≥** ``hold_seconds`` 时才判定恢复：
  首段与后续有效线段在读数时刻相接时连续计时，任一维出带或采样间隔
  超限即中断并重新计时；
- ``recovered_at`` 为成功保持段的起点（两维同时入带的精确交点，可为
  事件时刻本身）加上 ``hold_seconds`` 的确认时刻，``recovery_seconds``
  为其相对事件时刻的秒数；
- 事件后读数数目与**最大偏差**只统计**严格晚于**事件的读数：最大偏差
  为各维该类读数相对基线中位数的最大绝对偏差（分段线性曲线的极值必
  落在读数时刻，并列取最早）。

全部计算确定性：相同输入恒得相同结果。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from statistics import median
from typing import Protocol

TEMPERATURE = "temperature"
HUMIDITY = "humidity"

# 未恢复原因
NO_POST_EVENT_READINGS = "no_post_event_readings"
NEVER_SIMULTANEOUSLY_IN_BAND = "never_simultaneously_in_band"
DWELL_INTERRUPTED = "dwell_interrupted"
DWELL_INCOMPLETE_AT_HORIZON = "dwell_incomplete_at_horizon"


class _SampleLike(Protocol):
    ts: dt.datetime
    temperature: float
    humidity: float


class InsufficientBaselineSamples(Exception):
    """事件前基线窗口内样本数不足 ``min_baseline_samples``。"""

    def __init__(
        self,
        baseline_sample_count: int,
        min_baseline_samples: int,
        window_start: dt.datetime,
        window_end: dt.datetime,
    ) -> None:
        self.baseline_sample_count = baseline_sample_count
        self.min_baseline_samples = min_baseline_samples
        self.window_start = window_start
        self.window_end = window_end
        super().__init__(
            f"基线样本不足: 窗口 [{window_start.isoformat()}, {window_end.isoformat()}] "
            f"内仅 {baseline_sample_count} 条读数 < min_baseline_samples="
            f"{min_baseline_samples}"
        )


@dataclass(frozen=True)
class RecoveryBaseline:
    """事件前基线：窗口、样本数、温湿度中位数与由此确定的容差带边界。"""

    window_start: dt.datetime
    window_end: dt.datetime
    sample_count: int
    temperature: float
    humidity: float
    temperature_low: float
    temperature_high: float
    humidity_low: float
    humidity_high: float


@dataclass(frozen=True)
class RecoveryAnalysis:
    """扰动恢复分析结果。"""

    event_ts: dt.datetime
    horizon_end: dt.datetime
    baseline: RecoveryBaseline
    post_event_sample_count: int
    temp_tolerance: float
    humidity_tolerance: float
    hold_seconds: float
    max_gap_seconds: float
    recovered: bool
    band_enter_ts: dt.datetime | None
    recovered_at: dt.datetime | None
    recovery_seconds: float | None
    max_temperature_deviation: float | None
    max_temperature_deviation_ts: dt.datetime | None
    max_humidity_deviation: float | None
    max_humidity_deviation_ts: dt.datetime | None
    reason: str | None


def _cross_ts(t0: dt.datetime, t1: dt.datetime, frac: float) -> dt.datetime:
    """插值出交点时刻；端点分数直接返回原对象，保证相邻区间精确相接。"""
    if frac <= 0.0:
        return t0
    if frac >= 1.0:
        return t1
    return t0 + dt.timedelta(seconds=frac * (t1 - t0).total_seconds())


def _band_fractions(
    v0: float,
    v1: float,
    lo: float,
    hi: float,
) -> tuple[float, float] | None:
    """单采样区间上，一维曲线落在容差带 ``[lo, hi]``（含边界）内的分数区间。

    值沿线段线性变化 ``v(u) = v0 + (v1 − v0)·u``（u∈[0,1]）；带内部分
    必为单个（可为空、可为全长、可退化为单点的）分数区间 ``[a, b]``。
    """
    d = v1 - v0
    lower, upper = 0.0, 1.0
    if d > 0.0:
        lower = max(lower, (lo - v0) / d)  # lo ≤ v(u)
        upper = min(upper, (hi - v0) / d)  # v(u) ≤ hi
    elif d < 0.0:
        lower = max(lower, (hi - v0) / d)
        upper = min(upper, (lo - v0) / d)
    elif not (lo <= v0 <= hi):
        return None
    # 数值兜底：理论解已在 [0,1] 内，夹一刀消除浮点越界
    lower = min(max(lower, 0.0), 1.0)
    upper = min(max(upper, 0.0), 1.0)
    if lower > upper:
        return None
    return lower, upper


def _max_deviation(
    post: list[_SampleLike], attr: str, center: float
) -> tuple[float, dt.datetime] | None:
    """事件后读数相对基线中位数的最大绝对偏差及其时刻（并列取最早）。"""
    best: tuple[float, dt.datetime] | None = None
    for s in post:  # post 已按 ts 升序
        deviation = abs(getattr(s, attr) - center)
        if best is None or deviation > best[0]:
            best = (deviation, s.ts)
    return best


def analyze_recovery(
    samples: list[_SampleLike],
    *,
    event_ts: dt.datetime,
    baseline_seconds: float,
    horizon_seconds: float,
    min_baseline_samples: int,
    temp_tolerance: float,
    humidity_tolerance: float,
    hold_seconds: float,
    max_gap_seconds: float,
) -> RecoveryAnalysis:
    """按事件前中位数基线与事件后分段线性曲线判定扰动恢复。

    samples 无需预先排序；同一展柜的读数应已按 (device_sn, ts) 合并。
    基线样本不足时抛 :class:`InsufficientBaselineSamples`，由接口层判 422。
    """
    ordered = sorted(samples, key=lambda s: s.ts)

    window_start = event_ts - dt.timedelta(seconds=baseline_seconds)
    horizon_end = event_ts + dt.timedelta(seconds=horizon_seconds)

    # 事件前窗口（含两端）；恰好落在 event_ts 的读数归入基线
    baseline_samples = [s for s in ordered if window_start <= s.ts <= event_ts]
    if len(baseline_samples) < min_baseline_samples:
        raise InsufficientBaselineSamples(
            baseline_sample_count=len(baseline_samples),
            min_baseline_samples=min_baseline_samples,
            window_start=window_start,
            window_end=event_ts,
        )

    temp_median = float(median(s.temperature for s in baseline_samples))
    hum_median = float(median(s.humidity for s in baseline_samples))
    t_lo, t_hi = temp_median - temp_tolerance, temp_median + temp_tolerance
    h_lo, h_hi = hum_median - humidity_tolerance, hum_median + humidity_tolerance

    baseline = RecoveryBaseline(
        window_start=window_start,
        window_end=event_ts,
        sample_count=len(baseline_samples),
        temperature=temp_median,
        humidity=hum_median,
        temperature_low=t_lo,
        temperature_high=t_hi,
        humidity_low=h_lo,
        humidity_high=h_hi,
    )

    # 事件后窗口：严格晚于 event_ts、不晚于 horizon_end
    post = [s for s in ordered if event_ts < s.ts <= horizon_end]

    # 事件时刻读数仍计入基线（中位数已含它），同时作为事件后首段曲线的
    # 左端点；同刻多条读数按 (temperature, humidity) 确定性取一条即可，
    # 它们对中位数的影响已在基线中体现。
    at_event = [s for s in ordered if s.ts == event_ts]
    chain: list[_SampleLike] = []
    if at_event:
        chain.append(min(at_event, key=lambda s: (s.temperature, s.humidity)))
    chain.extend(post)

    # 最大偏差与读数数目只统计严格晚于事件的读数（不含事件时刻读数）
    t_dev = _max_deviation(post, "temperature", temp_median)
    h_dev = _max_deviation(post, "humidity", hum_median)

    # 逐采样区间求“两维同时在带内”的子区间；间隔超 max_gap 直接跳过，
    # 断档两侧的片天然不会相接合并。首个区间为事件时刻读数 → 首条事件后
    # 读数（两点间隔不超过 max_gap 时才存在）。
    pieces: list[tuple[dt.datetime, dt.datetime]] = []
    for i in range(len(chain) - 1):
        t0, t1 = chain[i].ts, chain[i + 1].ts
        span = (t1 - t0).total_seconds()
        if span <= 0.0 or span > max_gap_seconds:
            continue
        tf = _band_fractions(chain[i].temperature, chain[i + 1].temperature, t_lo, t_hi)
        hf = _band_fractions(chain[i].humidity, chain[i + 1].humidity, h_lo, h_hi)
        if tf is None or hf is None:
            continue
        a = max(tf[0], hf[0])  # 两维同时入带 = 两维带内区间的交集
        b = min(tf[1], hf[1])
        if a > b:
            continue
        pieces.append((_cross_ts(t0, t1, a), _cross_ts(t0, t1, b)))

    # 合并在相邻读数时刻相接的片（容差带含边界：单点相接不算中断）；
    # 任一维出带或采样间隔超限时片之间存在间隙，绝不合并，计时重新开始。
    merged: list[list[dt.datetime]] = []
    for start_ts, end_ts in pieces:
        if merged and start_ts <= merged[-1][1]:
            if end_ts > merged[-1][1]:
                merged[-1][1] = end_ts
        else:
            merged.append([start_ts, end_ts])

    recovered = False
    band_enter_ts: dt.datetime | None = None
    recovered_at: dt.datetime | None = None
    recovery_seconds: float | None = None
    for start_ts, end_ts in merged:
        if (end_ts - start_ts).total_seconds() >= hold_seconds:
            recovered = True
            band_enter_ts = start_ts
            recovered_at = start_ts + dt.timedelta(seconds=hold_seconds)
            recovery_seconds = (recovered_at - event_ts).total_seconds()
            break

    reason: str | None = None
    if not recovered:
        if not post:
            reason = NO_POST_EVENT_READINGS
        else:
            positive = [
                (s, e) for s, e in merged if (e - s).total_seconds() > 0.0
            ]
            if not positive:
                # 从未形成正长度的两维同时在带区间（仅单点恰好落在边界不算）
                reason = NEVER_SIMULTANEOUSLY_IN_BAND
            elif positive[-1][1] == post[-1].ts:
                # 最后一段同时在带连续延伸到观测末点，只是保持时间不足
                reason = DWELL_INCOMPLETE_AT_HORIZON
            else:
                # 曾经同时在带，但出带或断档终结了保持计时（末点孤立在带内亦然）
                reason = DWELL_INTERRUPTED

    return RecoveryAnalysis(
        event_ts=event_ts,
        horizon_end=horizon_end,
        baseline=baseline,
        post_event_sample_count=len(post),
        temp_tolerance=temp_tolerance,
        humidity_tolerance=humidity_tolerance,
        hold_seconds=hold_seconds,
        max_gap_seconds=max_gap_seconds,
        recovered=recovered,
        band_enter_ts=band_enter_ts,
        recovered_at=recovered_at,
        recovery_seconds=recovery_seconds,
        max_temperature_deviation=t_dev[0] if t_dev else None,
        max_temperature_deviation_ts=t_dev[1] if t_dev else None,
        max_humidity_deviation=h_dev[0] if h_dev else None,
        max_humidity_deviation_ts=h_dev[1] if h_dev else None,
        reason=reason,
    )
