"""冷凝风险（condensation risk）计算（纯计算，只读）。

把展柜的空气温湿度读数视为**分段线性**曲线，用 Magnus 公式由空气
温度 T（°C）与相对湿度 RH（%RH）逐点计算露点 Td（°C）：

    γ  = ln(RH/100) + 17.62·T/(243.12 + T)
    Td = 243.12·γ/(17.62 − γ)

给定表面温度 Ts 下的**冷凝裕量** ``margin = Ts − Td``：裕量越小越
接近结露，``margin ≤ margin_threshold_c`` 视为风险（**含等于阈值**）。

- 在**相邻有效点**之间对冷凝裕量**线性插值**，精确求出进入与离开
  风险阈值的时刻，不做阶梯近似；
- 相邻读数间隔**大于** ``max_gap_seconds`` 时曲线断开（恰好等于
  ``max_gap`` 仍视为连续），未知区间不计风险；
- **湿度为 0 的读数无效**（``ln(0)`` 不定义）：相邻区间任一端点
  湿度为 0 时该区间必须断开，两侧曲线彼此独立；
- 仅返回持续时间 **≥** ``min_duration_seconds`` 的风险窗口；窗口
  含起止时间、最低裕量及其时刻（裕量沿线段线性，最低裕量必落在
  某个有效读数时刻，并列取最早出现者）；
- 断档两侧的风险窗口**绝不跨断档合并**。

全部计算确定性：相同输入恒得相同窗口序列（按开始时间排序）。
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Protocol

# Magnus 公式常数（Alduchov–Eskridge 常用系数）
MAGNUS_A = 17.62
MAGNUS_B = 243.12


class _SampleLike(Protocol):
    ts: dt.datetime
    temperature: float
    humidity: float


def dew_point(temperature_c: float, humidity_pct: float) -> float:
    """Magnus 公式露点（°C）。

    ``humidity_pct`` 必须严格为正；湿度为 0 时露点不定义，由调用方
    （:func:`compute_condensation_windows`）先行剔除。
    """
    gamma = math.log(humidity_pct / 100.0) + MAGNUS_A * temperature_c / (
        MAGNUS_B + temperature_c
    )
    return MAGNUS_B * gamma / (MAGNUS_A - gamma)


def condensation_margin(
    surface_temperature_c: float,
    air_temperature_c: float,
    humidity_pct: float,
) -> float:
    """冷凝裕量 = 给定表面温度 − Magnus 露点（°C）。"""
    return surface_temperature_c - dew_point(air_temperature_c, humidity_pct)


@dataclass(frozen=True)
class CondensationWindow:
    """一段连续的冷凝风险区间（裕量 ≤ 阈值）。"""

    start_ts: dt.datetime
    end_ts: dt.datetime
    duration_seconds: float
    min_margin: float  # 区间内最低冷凝裕量（°C，越负结露越严重）
    min_margin_ts: dt.datetime


@dataclass
class _RiskPiece:
    """单采样区间内切出的风险子区间（裕量沿该线段线性）。"""

    start_ts: dt.datetime
    end_ts: dt.datetime
    min_margin: float
    min_margin_ts: dt.datetime


def _cross_ts(t0: dt.datetime, t1: dt.datetime, frac: float) -> dt.datetime:
    """插值出交点时刻；端点分数直接返回原对象，保证相邻区间精确相接。"""
    if frac <= 0.0:
        return t0
    if frac >= 1.0:
        return t1
    return t0 + dt.timedelta(seconds=frac * (t1 - t0).total_seconds())


def _risk_piece(
    t0: dt.datetime,
    t1: dt.datetime,
    m0: float,
    m1: float,
    threshold: float,
) -> _RiskPiece | None:
    """单采样区间上的风险子区间（裕量 m 线性，风险条件 m ≤ threshold）。

    令越阈幅度 ``e = threshold − m``（e ≥ 0 即风险，**含等于**），
    返回风险部分占采样区间的端点分数并切出子区间；两端都严格安全
    （e < 0）时返回 ``None``。
    """
    e0 = threshold - m0
    e1 = threshold - m1
    if e0 < 0.0 and e1 < 0.0:
        return None
    if e0 >= 0.0:
        f0 = 0.0
    else:
        f0 = e0 / (e0 - e1)  # e=0 的交点分数
    if e1 >= 0.0:
        f1 = 1.0
    else:
        f1 = e0 / (e0 - e1)

    start_ts = _cross_ts(t0, t1, f0)
    end_ts = _cross_ts(t0, t1, f1)

    # 裕量线性：子区间最低裕量在端点；交点处裕量恰为阈值。
    # 并列时取更早时刻，保证确定性。
    candidates: list[tuple[float, dt.datetime]] = []
    if f0 == 0.0:
        candidates.append((m0, t0))
    else:
        candidates.append((threshold, start_ts))
    if f1 == 1.0:
        candidates.append((m1, t1))
    else:
        candidates.append((threshold, end_ts))
    min_margin, min_ts = min(candidates, key=lambda x: (x[0], x[1]))
    return _RiskPiece(
        start_ts=start_ts,
        end_ts=end_ts,
        min_margin=min_margin,
        min_margin_ts=min_ts,
    )


def compute_condensation_windows(
    samples: list[_SampleLike],
    surface_temperature_c: float,
    margin_threshold_c: float,
    max_gap_seconds: float,
    min_duration_seconds: float,
) -> list[CondensationWindow]:
    """计算冷凝风险窗口序列（按开始时间稳定排序）。

    samples 无需预先排序；同一展柜的读数应已按 (device_sn, ts) 合并。
    仅返回持续秒数 **≥** ``min_duration_seconds`` 的窗口；断档（采样
    间隔超限或湿度为 0）两侧的子区间绝不合并。
    """
    ordered = sorted(samples, key=lambda s: s.ts)

    # 逐点裕量；湿度为 0 的读数无效（露点不定义），标记为 None
    margins: list[float | None] = []
    for s in ordered:
        if s.humidity <= 0.0:
            margins.append(None)
        else:
            margins.append(
                condensation_margin(
                    surface_temperature_c, s.temperature, s.humidity
                )
            )

    pieces: list[_RiskPiece] = []
    for i in range(len(ordered) - 1):
        t0, t1 = ordered[i].ts, ordered[i + 1].ts
        span = (t1 - t0).total_seconds()
        if span <= 0.0 or span > max_gap_seconds:
            continue  # 采样间隔超过 max_gap：断开，未知区间不计风险
        m0, m1 = margins[i], margins[i + 1]
        if m0 is None or m1 is None:
            continue  # 任一端点湿度为 0：区间必须断开
        piece = _risk_piece(t0, t1, m0, m1, margin_threshold_c)
        if piece is not None:
            pieces.append(piece)

    pieces.sort(key=lambda p: (p.start_ts, p.end_ts))

    # 合并同一连续曲线段上相接的风险子区间；断档两侧的片在时间上
    # 互不接触（中间的采样区间被跳过），天然不会被合并。
    windows: list[CondensationWindow] = []
    for p in pieces:
        if windows and p.start_ts <= windows[-1].end_ts:
            cur = windows[-1]
            # dataclass frozen：替换为延长后的新窗口
            end_ts = max(cur.end_ts, p.end_ts)
            if p.min_margin < cur.min_margin:  # 并列保留更早出现者
                min_margin, min_ts = p.min_margin, p.min_margin_ts
            else:
                min_margin, min_ts = cur.min_margin, cur.min_margin_ts
            windows[-1] = CondensationWindow(
                start_ts=cur.start_ts,
                end_ts=end_ts,
                duration_seconds=(end_ts - cur.start_ts).total_seconds(),
                min_margin=min_margin,
                min_margin_ts=min_ts,
            )
        else:
            windows.append(
                CondensationWindow(
                    start_ts=p.start_ts,
                    end_ts=p.end_ts,
                    duration_seconds=(p.end_ts - p.start_ts).total_seconds(),
                    min_margin=p.min_margin,
                    min_margin_ts=p.min_margin_ts,
                )
            )

    return [w for w in windows if w.duration_seconds >= min_duration_seconds]
