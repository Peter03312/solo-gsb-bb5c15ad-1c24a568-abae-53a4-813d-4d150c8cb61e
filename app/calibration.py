"""锚点参考值校正：分维偏移曲线的构建与求值（纯计算，只读）。

规则（全部确定性，相同输入严格产生相同输出）：

1. 每个维度（温度 / 湿度）**独立**计算偏移：

       偏移 = 锚点参考值 − 锚点所对齐读数的原始读数

2. 缺少该维参考值（``None``）的锚点不参与该维曲线；未对齐到读数
   的锚点两个维度都不参与（服务层只传入已对齐锚点）。
3. 多个锚点对齐到**同一读数**时，按锚点时间升序遍历，取**最后一个**
   有效参考值计算该读数处的偏移。
4. 偏移曲线定义在**整条时间线**上：相邻有效锚点（节点）之间按时间
   **线性插值**；首节点之前、末节点之后分别沿用最近节点的偏移
   （常数外推）。某维没有任何有效锚点时，该维偏移恒为 0。
5. 校正值 = 原始读数 + 该维偏移。调用方先基于整条时间线建立曲线，
   再按 start/end 过滤输出——窗口外的锚点仍参与窗口内读数的偏移。
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_left
from dataclasses import dataclass
from typing import Protocol

TEMPERATURE = "temperature"
HUMIDITY = "humidity"


class _SampleLike(Protocol):
    ts: dt.datetime
    temperature: float
    humidity: float


@dataclass(frozen=True)
class AlignedAnchor:
    """已对齐到读数的锚点及其参考值（服务层负责解析对齐读数）。"""

    anchor_ts: dt.datetime
    reading_ts: dt.datetime  # 对齐到的读数时间戳
    reading_temperature: float  # 对齐读数的当前原始值
    reading_humidity: float
    ref_temperature: float | None
    ref_humidity: float | None


@dataclass(frozen=True)
class CalibratedPoint:
    """单个读数的原值、两维偏移与校正值。"""

    ts: dt.datetime
    temperature: float
    humidity: float
    temperature_offset: float
    humidity_offset: float
    calibrated_temperature: float
    calibrated_humidity: float


def _dimension_knots(
    anchors: list[AlignedAnchor],
    dimension: str,
) -> list[tuple[dt.datetime, float]]:
    """单维偏移节点序列：``[(对齐读数时间戳, 偏移), ...]``，按时间升序。

    只纳入该维参考值非空的锚点；``anchors`` 必须已按锚点时间升序，
    同一读数被多个锚点对齐时后写覆盖，即取最后一个有效参考值。
    """
    by_reading: dict[dt.datetime, float] = {}
    for a in anchors:
        if dimension == TEMPERATURE:
            ref, raw = a.ref_temperature, a.reading_temperature
        elif dimension == HUMIDITY:
            ref, raw = a.ref_humidity, a.reading_humidity
        else:  # pragma: no cover - 内部调用方只传两个维度
            raise ValueError(f"未知维度: {dimension}")
        if ref is None:
            continue  # 缺少该维参考值的锚点不参与该维曲线
        by_reading[a.reading_ts] = ref - raw
    return sorted(by_reading.items())


class OffsetCurve:
    """单维分段线性偏移曲线（节点按时间升序，时间戳互不相同）。

    节点之间按时间线性插值；首节点之前、末节点之后沿用最近节点
    的偏移；没有节点时恒为 0。
    """

    def __init__(self, knots: list[tuple[dt.datetime, float]]):
        self._times = [k[0] for k in knots]
        self._values = [k[1] for k in knots]

    @property
    def knot_count(self) -> int:
        return len(self._times)

    def at(self, ts: dt.datetime) -> float:
        n = len(self._times)
        if n == 0:
            return 0.0
        if ts <= self._times[0]:
            return self._values[0]
        if ts >= self._times[-1]:
            return self._values[-1]
        i = bisect_left(self._times, ts)
        t0, t1 = self._times[i - 1], self._times[i]
        v0, v1 = self._values[i - 1], self._values[i]
        frac = (ts - t0).total_seconds() / (t1 - t0).total_seconds()
        return v0 + (v1 - v0) * frac


def calibrate_timeline(
    samples: list[_SampleLike],
    anchors: list[AlignedAnchor],
) -> list[CalibratedPoint]:
    """对整条时间线逐点计算两维偏移与校正值（输入顺序即输出顺序）。

    ``anchors`` 只需包含已对齐锚点；内部按锚点时间升序稳定排序，
    保证“同一读数取最后一个有效参考值”的规则与传入顺序无关。
    """
    ordered = sorted(anchors, key=lambda a: a.anchor_ts)
    temp_curve = OffsetCurve(_dimension_knots(ordered, TEMPERATURE))
    humidity_curve = OffsetCurve(_dimension_knots(ordered, HUMIDITY))

    out: list[CalibratedPoint] = []
    for s in samples:
        temp_offset = temp_curve.at(s.ts)
        humidity_offset = humidity_curve.at(s.ts)
        out.append(
            CalibratedPoint(
                ts=s.ts,
                temperature=s.temperature,
                humidity=s.humidity,
                temperature_offset=temp_offset,
                humidity_offset=humidity_offset,
                calibrated_temperature=s.temperature + temp_offset,
                calibrated_humidity=s.humidity + humidity_offset,
            )
        )
    return out
