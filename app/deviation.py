"""展柜同伴基线偏离查询（peer baseline deviation，纯计算，只读）。

以目标展柜为观测对象、2～10 个对照展柜为同伴基线，在统一的等距
网格上检测目标展柜相对对照中位数基线的持续偏离：

- 网格以查询起点 ``start`` 为原点，按 ``step_seconds`` 等距生成，
  时刻不晚于 ``end``（不足一个步长的尾部不补点）；
- 每个展柜（目标与对照）的读数视为分段线性曲线，仅在相邻读数间隔
  **不超过** ``max_gap_seconds`` 时线性插值（恰好等于仍连续）；
  落在覆盖范围外或断档内的网格点，该柜在该点无效；
- 每刻至少 **两个** 有效对照值才取**中位数**作为基线，否则该刻
  **中断判定**；目标值在该刻无效同样中断；
- 目标值与基线之差的绝对值**达到** ``deviation_threshold``
  （含等于）即标记偏离；
- 连续标记按相邻网格合并（网格等距，相邻时刻之差恒为步长），
  中断或未标记的时刻切断合并；仅返回持续时间
  （末标记时刻 − 首标记时刻）**≥** ``min_duration_seconds``
  （含等于）的时段；
- 每个时段给出最大偏离绝对值、峰值时刻（并列取最早）与方向
  （``up``：目标高于基线 / ``down``：目标低于基线）。

全部计算确定性，相同输入恒得相同输出；不写库、不改版本号。
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_left
from dataclasses import dataclass
from statistics import median
from typing import Protocol

TEMPERATURE = "temperature"
HUMIDITY = "humidity"

# 偏离方向
DIRECTION_UP = "up"  # 峰值处目标值高于基线
DIRECTION_DOWN = "down"  # 峰值处目标值低于基线

# 构成基线所需的最少有效对照值个数
MIN_BASELINE_PEERS = 2


class _SampleLike(Protocol):
    ts: dt.datetime
    temperature: float
    humidity: float


@dataclass(frozen=True)
class DeviationPeriod:
    """一段持续时间达标的偏离时段（起止均为被标记的网格时刻）。"""

    start_ts: dt.datetime
    end_ts: dt.datetime
    duration_seconds: float
    marked_count: int
    direction: str
    max_deviation: float
    peak_ts: dt.datetime


@dataclass(frozen=True)
class DeviationResult:
    """同伴基线偏离查询结果。

    ``evaluated_count`` 为目标有效且对照有效值不少于两个、实际完成
    偏离判定的网格点数；``marked_count`` 为被标记偏离的网格点总数。
    """

    dimension: str
    start: dt.datetime
    end: dt.datetime
    step_seconds: float
    max_gap_seconds: float
    deviation_threshold: float
    min_duration_seconds: float
    grid_point_count: int
    evaluated_count: int
    marked_count: int
    periods: tuple[DeviationPeriod, ...]


def _dimension_value(sample: _SampleLike, dimension: str) -> float:
    if dimension == TEMPERATURE:
        return sample.temperature
    if dimension == HUMIDITY:
        return sample.humidity
    raise ValueError(f"未知维度: {dimension}")  # pragma: no cover - 接口层已枚举校验


def _series(
    samples: list[_SampleLike], dimension: str
) -> tuple[list[dt.datetime], list[float]]:
    ordered = sorted(samples, key=lambda s: s.ts)
    return [s.ts for s in ordered], [_dimension_value(s, dimension) for s in ordered]


def _interpolate_at(
    times: list[dt.datetime],
    values: list[float],
    ts: dt.datetime,
    max_gap_seconds: float,
) -> float | None:
    """在分段线性曲线上求 ``ts`` 处的值。

    - ``ts`` 落在覆盖范围外返回 ``None``；
    - 所依赖的相邻读数间隔大于 ``max_gap_seconds``（断档）返回 ``None``；
    - 恰好落在读数时刻直接取该读数；断档判定含等于（等于 max_gap 连续）。
    """
    n = len(times)
    if n == 0 or ts < times[0] or ts > times[-1]:
        return None
    i = bisect_left(times, ts)
    if i < n and times[i] == ts:
        return values[i]
    # i >= 1（ts > times[0]）且 i-1, i 构成包围 ts 的相邻读数
    t0, t1 = times[i - 1], times[i]
    if (t1 - t0).total_seconds() > max_gap_seconds:
        return None
    v0, v1 = values[i - 1], values[i]
    frac = (ts - t0).total_seconds() / (t1 - t0).total_seconds()
    return v0 + (v1 - v0) * frac


def _build_grid(start: dt.datetime, end: dt.datetime, step_seconds: float) -> list[dt.datetime]:
    """以 ``start`` 为原点按固定步长生成不超过 ``end`` 的等距时刻（含起点）。"""
    if end < start:
        return []
    # 亚微秒步长在 datetime 精度下无意义，兜底为 1µs 避免零步长
    step_us = max(1, int(round(step_seconds * 1_000_000)))
    span_us = (end - start) // dt.timedelta(microseconds=1)
    n = span_us // step_us
    return [start + dt.timedelta(microseconds=step_us * k) for k in range(n + 1)]


def _grid_deviations(
    target_samples: list[_SampleLike],
    peer_samples: list[list[_SampleLike]],
    dimension: str,
    grid: list[dt.datetime],
    max_gap_seconds: float,
) -> list[float | None]:
    """逐网格时刻求目标值 − 对照中位数基线的偏离；该刻中断判定为 ``None``。"""
    tgt_times, tgt_values = _series(target_samples, dimension)
    peer_series = [_series(samples, dimension) for samples in peer_samples]

    deviations: list[float | None] = []
    for t in grid:
        target_value = _interpolate_at(tgt_times, tgt_values, t, max_gap_seconds)
        if target_value is None:
            deviations.append(None)
            continue
        peer_values = [
            value
            for times, values in peer_series
            if (value := _interpolate_at(times, values, t, max_gap_seconds)) is not None
        ]
        if len(peer_values) < MIN_BASELINE_PEERS:
            deviations.append(None)
            continue
        deviations.append(target_value - float(median(peer_values)))
    return deviations


def _merge_periods(
    grid: list[dt.datetime],
    deviations: list[float | None],
    deviation_threshold: float,
    min_duration_seconds: float,
) -> tuple[DeviationPeriod, ...]:
    """把连续达标的标记按相邻网格合并为时段，并按最短持续时间过滤。

    网格等距，相邻时刻之差恒为步长（不超过步长），故下标连续的标记
    即满足合并条件；中断（``None``）或未达标的时刻切断合并。峰值取
    时段内绝对偏离最大的网格时刻，并列取最早。
    """
    marked = [
        d is not None and abs(d) >= deviation_threshold for d in deviations
    ]
    periods: list[DeviationPeriod] = []
    n = len(grid)
    i = 0
    while i < n:
        if not marked[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and marked[j + 1]:
            j += 1
        duration = (grid[j] - grid[i]).total_seconds()
        if duration >= min_duration_seconds:
            peak = i
            for k in range(i + 1, j + 1):
                if abs(deviations[k]) > abs(deviations[peak]):  # type: ignore[arg-type]
                    peak = k
            peak_dev = deviations[peak]
            assert peak_dev is not None  # 被标记的时刻必有偏离值
            periods.append(
                DeviationPeriod(
                    start_ts=grid[i],
                    end_ts=grid[j],
                    duration_seconds=duration,
                    marked_count=j - i + 1,
                    direction=DIRECTION_UP if peak_dev > 0 else DIRECTION_DOWN,
                    max_deviation=abs(peak_dev),
                    peak_ts=grid[peak],
                )
            )
        i = j + 1
    return tuple(periods)


def compute_peer_baseline_deviation(
    target_samples: list[_SampleLike],
    peer_samples: list[list[_SampleLike]],
    dimension: str,
    start: dt.datetime,
    end: dt.datetime,
    step_seconds: float,
    max_gap_seconds: float,
    deviation_threshold: float,
    min_duration_seconds: float,
) -> DeviationResult:
    """在等距网格上检测目标展柜相对对照中位数基线的持续偏离时段。

    samples 无需预先排序；``start`` / ``end`` 已归一化为 UTC 且
    ``start <= end``（由接口层校验）。对照展柜没有读数时该柜在全部
    网格时刻无效；有效对照值不足两个的时刻中断判定，自然得到空结果。
    """
    grid = _build_grid(start, end, step_seconds)
    deviations = _grid_deviations(
        target_samples, peer_samples, dimension, grid, max_gap_seconds
    )
    periods = _merge_periods(
        grid, deviations, deviation_threshold, min_duration_seconds
    )
    return DeviationResult(
        dimension=dimension,
        start=start,
        end=end,
        step_seconds=step_seconds,
        max_gap_seconds=max_gap_seconds,
        deviation_threshold=deviation_threshold,
        min_duration_seconds=min_duration_seconds,
        grid_point_count=len(grid),
        evaluated_count=sum(d is not None for d in deviations),
        marked_count=sum(
            d is not None and abs(d) >= deviation_threshold for d in deviations
        ),
        periods=periods,
    )
