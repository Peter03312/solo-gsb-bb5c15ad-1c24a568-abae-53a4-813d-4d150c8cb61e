"""展柜采集质量审计：基于邻点中位数与 MAD 的孤立异常值定位（纯计算，只读）。

对每个展柜**已合并读数**按时间排序后的序列做如下审计（全程不修改读数、
分段或版本号）：

1. **断档切段**：相邻读数间隔**大于** ``max_gap_seconds`` 时切断序列
   （恰好等于仍连续），邻点只在同一连续段内取；
2. **邻点窗**：每个待判点左侧、右侧各取**最多** ``window_size`` 个
   最近邻点（不含自身）；任一侧邻点**不足 2 个**时该点**不判**
   （段边界附近的点因此天然跳过）；
3. **稳健尺度**：温度、湿度分别基于邻点计算中位数 ``med`` 与
   ``MAD = median(|x − med|)``，尺度取
   ``scale = max(1.4826·MAD, min_delta)``——邻点完全一致（MAD=0）时
   由该维 ``min_delta`` 兜底，避免零尺度误报；
4. **标记**：原值与中位数的绝对偏差**严格大于** ``k·scale``
   （``|x − med| > k·scale``，等于不算）时标记该维；两维独立判定，
   至少一维命中的读数作为疑点返回。

疑点按时间升序排列；全部计算确定性：相同输入恒得相同结果。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from statistics import median
from typing import Protocol

TEMPERATURE = "temperature"
HUMIDITY = "humidity"

# 正态一致的 MAD 尺度常数
MAD_SCALE = 1.4826

# 每侧至少需要的邻点数；任一侧不足即不判
MIN_SIDE_NEIGHBORS = 2


class _SampleLike(Protocol):
    ts: dt.datetime
    temperature: float
    humidity: float


@dataclass(frozen=True)
class OutlierDimension:
    """某一命中的异常维度及其中位数与绝对偏差。"""

    dimension: str
    median: float
    deviation: float


@dataclass(frozen=True)
class ReadingOutlier:
    """疑点读数：原值 + 命中维度（温度在前）的统计量。"""

    ts: dt.datetime
    temperature: float
    humidity: float
    dimensions: tuple[OutlierDimension, ...]


@dataclass(frozen=True)
class ReadingOutlierResult:
    sample_count: int
    segment_count: int
    evaluated_count: int
    outliers: list[ReadingOutlier]


def _split_continuous(
    ordered: list[_SampleLike], max_gap_seconds: float
) -> list[list[_SampleLike]]:
    """按相邻读数间隔切断：间隔严格大于 max_gap_seconds 处开新段。"""
    segments: list[list[_SampleLike]] = []
    current: list[_SampleLike] = []
    prev_ts: dt.datetime | None = None
    for s in ordered:
        if (
            current
            and prev_ts is not None
            and (s.ts - prev_ts).total_seconds() > max_gap_seconds
        ):
            segments.append(current)
            current = []
        current.append(s)
        prev_ts = s.ts
    if current:
        segments.append(current)
    return segments


def _neighbor_stats(neighbors: list[_SampleLike], attr: str) -> tuple[float, float]:
    """邻点在某一维上的中位数与 MAD。"""
    values = [getattr(s, attr) for s in neighbors]
    med = float(median(values))
    mad = float(median(abs(v - med) for v in values))
    return med, mad


def compute_reading_outliers(
    samples: list[_SampleLike],
    *,
    window_size: int,
    k: float,
    min_temperature_delta: float,
    min_humidity_delta: float,
    max_gap_seconds: float,
) -> ReadingOutlierResult:
    """定位孤立异常读数（只读纯计算）。

    samples 无需预先排序；同一展柜的读数应已按 (device_sn, ts) 合并。
    参数合法性（``window_size ∈ [2, 50]``、各正数参数 > 0）由接口层
    校验，本函数假定入参合法。
    """
    ordered = sorted(samples, key=lambda s: s.ts)
    segments = _split_continuous(ordered, max_gap_seconds)

    dims_cfg = (
        (TEMPERATURE, "temperature", min_temperature_delta),
        (HUMIDITY, "humidity", min_humidity_delta),
    )

    outliers: list[ReadingOutlier] = []
    evaluated_count = 0
    for seg in segments:
        n = len(seg)
        for i in range(n):
            # 仅取本连续段内、左右各最多 window_size 个邻点（切片天然排除自身）
            left = seg[max(0, i - window_size):i]
            right = seg[i + 1:min(n, i + 1 + window_size)]
            if len(left) < MIN_SIDE_NEIGHBORS or len(right) < MIN_SIDE_NEIGHBORS:
                continue  # 任一侧不足 2 点：不判

            evaluated_count += 1
            neighbors = left + right
            point = seg[i]

            flagged: list[OutlierDimension] = []
            for name, attr, min_delta in dims_cfg:
                med, mad = _neighbor_stats(neighbors, attr)
                scale = max(MAD_SCALE * mad, min_delta)
                deviation = abs(getattr(point, attr) - med)
                if deviation > k * scale:  # 严格大于；等于阈值不算
                    flagged.append(
                        OutlierDimension(
                            dimension=name,
                            median=med,
                            deviation=float(deviation),
                        )
                    )
            if flagged:
                outliers.append(
                    ReadingOutlier(
                        ts=point.ts,
                        temperature=point.temperature,
                        humidity=point.humidity,
                        dimensions=tuple(flagged),
                    )
                )

    return ReadingOutlierResult(
        sample_count=len(ordered),
        segment_count=len(segments),
        evaluated_count=evaluated_count,
        outliers=outliers,
    )
