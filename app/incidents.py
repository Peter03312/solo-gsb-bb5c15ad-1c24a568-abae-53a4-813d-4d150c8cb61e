"""跨展柜共模突变（common-mode incident）检测。

输入是各展柜**现有有效分段**的非基线段首（change boundary）：每一条
都是某个展柜一次突变的发生时刻、方向与幅度。本模块在全局范围内找出
"多个展柜在很短时间内相继突变" 的共模事件。

候选事件（连续候选窗）
----------------------

1. 把所有展柜落在查询范围内的段首按 ``(ts, device_sn)`` 全局排序；
2. 任意一段**连续段首** ``[i..j]``，只要最晚与最早边界之差不超过
   ``tolerance_seconds``（含等于），即构成一个候选窗；
3. 同一展柜在窗内可能出现多个段首，只保留**幅度最大**的一条，
   幅度并列取**时间更早**者；
4. 去重后涉及的 distinct device 数达到 ``min_devices`` 时候选才合法。

全局选择（加权区间调度，杜绝贪心链式归并）
------------------------------------------

枚举全部合法候选窗后，用动态规划选出一组**互不重叠**的事件
（后一事件的 ``start_ts`` 严格晚于前一事件的 ``end_ts``；同一时刻
的两簇突变视为同一事件，不允许拆开）。按以下次序依次最大化：

1. 入选事件的**设备数总和**；
2. 入选事件的**幅度总和**；
3. 入选事件按开始时间排列后的**起始时间序列**（逐项比较，越早越优；
   序列构成前缀并列时更长者优先，即宁拆成更多个更早的事件，也不
   归并成一个大事件）。

以上全部并列时，再以候选成员键的字典序兜底，保证相同输入恒得相同
输出，不含任何随机过程。
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_left
from dataclasses import dataclass

from .config import settings

DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
DIRECTION_FLAT = "flat"


@dataclass(frozen=True)
class Boundary:
    """某展柜非基线段的段首（一次突变边界）。"""

    device_sn: str
    ts: dt.datetime
    magnitude: float
    delta_temperature: float
    delta_humidity: float

    @property
    def key(self) -> tuple[dt.datetime, str]:
        return (self.ts, self.device_sn)


@dataclass(frozen=True)
class Incident:
    """一组跨展柜共模突变（成员来自不同展柜，时间跨度不超过容差）。"""

    start_ts: dt.datetime
    end_ts: dt.datetime
    members: tuple[Boundary, ...]
    device_count: int
    total_magnitude: float

    @property
    def duration_seconds(self) -> float:
        return (self.end_ts - self.start_ts).total_seconds()

    @property
    def event_key(self) -> tuple[tuple[dt.datetime, str], ...]:
        return tuple(b.key for b in self.members)


def boundary_direction(delta_temperature: float, delta_humidity: float) -> str:
    """边界方向：以温/湿两个缩放加权分量中占主导者的符号判定。

    温、湿分量并列时取温度（与其余并列规则一致，保证确定性）；
    主导分量恰为 0（如锚点强制切出的零跳变）时返回 ``flat``。
    """
    temp_component = (
        settings.TEMP_WEIGHT * (delta_temperature / settings.TEMP_SCALE) ** 2
    )
    hum_component = (
        settings.HUMIDITY_WEIGHT * (delta_humidity / settings.HUMIDITY_SCALE) ** 2
    )
    dominant = delta_humidity if hum_component > temp_component else delta_temperature
    if dominant > 0:
        return DIRECTION_UP
    if dominant < 0:
        return DIRECTION_DOWN
    return DIRECTION_FLAT


def _build_incident(members: set[tuple[dt.datetime, str]], by_key: dict) -> Incident:
    ordered = sorted((by_key[k] for k in members), key=lambda b: b.key)
    return Incident(
        start_ts=ordered[0].ts,
        end_ts=ordered[-1].ts,
        members=tuple(ordered),
        device_count=len({b.device_sn for b in ordered}),
        total_magnitude=sum(b.magnitude for b in ordered),
    )


def enumerate_incidents(
    boundaries: list[Boundary],
    tolerance_seconds: float,
    min_devices: int,
) -> list[Incident]:
    """枚举全部合法的连续候选窗（按成员集合去重）。"""
    ordered = sorted(boundaries, key=lambda b: b.key)
    by_key = {b.key: b for b in ordered}
    n = len(ordered)

    unique: dict[frozenset[tuple[dt.datetime, str]], Incident] = {}
    for i in range(n):
        # 窗 [i..j] 内每个展柜当前保留的边界；成员键集合随 j 增量维护
        chosen_by_device: dict[str, Boundary] = {}
        member_keys: set[tuple[dt.datetime, str]] = set()
        for j in range(i, n):
            span = (ordered[j].ts - ordered[i].ts).total_seconds()
            if span > tolerance_seconds:
                break  # 全局有序：之后的 j 只会更晚

            b = ordered[j]
            old = chosen_by_device.get(b.device_sn)
            changed = False
            if old is None:
                chosen_by_device[b.device_sn] = b
                member_keys.add(b.key)
                changed = True
            elif b.magnitude > old.magnitude:
                # 同展柜保留幅度更大的边界
                chosen_by_device[b.device_sn] = b
                member_keys.discard(old.key)
                member_keys.add(b.key)
                changed = True
            # 幅度并列：old 在更早的扫描位置（ts 更早），保持不变

            if changed and len(chosen_by_device) >= min_devices:
                fid = frozenset(member_keys)
                if fid not in unique:
                    unique[fid] = _build_incident(member_keys, by_key)

    return list(unique.values())


def select_incidents(incidents: list[Incident]) -> list[Incident]:
    """在全部合法候选窗上做加权区间调度，求全局最优的互不重叠事件集。

    优化目标按字典序比较：（设备数总和, 幅度总和, 起始时间序列, 候选键序列），
    末两项仅用于在物理量完全并列时保证结果确定。**序列比较采用"前缀相同时
    更长者优先"**：设备数与幅度总和相同的两种选法里，拆成更多个更早事件
    的方案优先——这与"贪心链式归并成大事件"恰好相反，确保全局拆分最优。
    候选按结束时间排序后，对每个候选用二分定位其最近的不重叠前驱，
    整体 O(m log m)。
    """
    if not incidents:
        return []

    candidates = sorted(
        incidents, key=lambda e: (e.end_ts, e.start_ts, e.event_key)
    )
    ends = [e.end_ts for e in candidates]
    m = len(candidates)

    # 下标 0 为空解；state[k] = 只考虑前 k 个候选时的最优解
    devices_sum = [0] * (m + 1)
    magnitude_sum = [0.0] * (m + 1)
    starts_seq: list[tuple[dt.datetime, ...]] = [()] * (m + 1)
    keys_seq: list[tuple] = [()] * (m + 1)
    chosen_here = [False] * m
    predecessor = [0] * m

    def _better(
        dev_a: int,
        mag_a: float,
        starts_a: tuple[dt.datetime, ...],
        keys_a: tuple,
        dev_b: int,
        mag_b: float,
        starts_b: tuple[dt.datetime, ...],
        keys_b: tuple,
    ) -> bool:
        """a 是否严格优于 b：设备数 → 幅度 → 起始序列 → 候选键序列。"""
        if dev_a != dev_b:
            return dev_a > dev_b
        if abs(mag_a - mag_b) > 1e-9:
            return mag_a > mag_b
        common = min(len(starts_a), len(starts_b))
        for i in range(common):
            if starts_a[i] != starts_b[i]:
                return starts_a[i] < starts_b[i]
        if len(starts_a) != len(starts_b):
            return len(starts_a) > len(starts_b)  # 前缀并列：更多更早事件优先
        common_k = min(len(keys_a), len(keys_b))
        for i in range(common_k):
            if keys_a[i] != keys_b[i]:
                return keys_a[i] < keys_b[i]
        return len(keys_a) > len(keys_b)

    for k, event in enumerate(candidates):
        pos = k + 1
        # 严格不重叠：前驱事件 end_ts < 本事件 start_ts
        p = bisect_left(ends, event.start_ts, 0, k) - 1
        pred_pos = p + 1

        take = (
            devices_sum[pred_pos] + event.device_count,
            magnitude_sum[pred_pos] + event.total_magnitude,
            starts_seq[pred_pos] + (event.start_ts,),
            keys_seq[pred_pos] + (event.event_key,),
        )
        skip = (
            devices_sum[pos - 1],
            magnitude_sum[pos - 1],
            starts_seq[pos - 1],
            keys_seq[pos - 1],
        )

        if _better(*take, *skip):
            devices_sum[pos], magnitude_sum[pos] = take[0], take[1]
            starts_seq[pos], keys_seq[pos] = take[2], take[3]
            chosen_here[k] = True
            predecessor[k] = pred_pos
        else:
            devices_sum[pos], magnitude_sum[pos] = skip[0], skip[1]
            starts_seq[pos], keys_seq[pos] = skip[2], skip[3]

    # 回溯
    selected: list[Incident] = []
    pos = m
    while pos > 0:
        k = pos - 1
        if chosen_here[k]:
            selected.append(candidates[k])
            pos = predecessor[k]
        else:
            pos -= 1

    selected.sort(key=lambda e: (e.start_ts, e.end_ts, e.event_key))
    return selected


def detect_incidents(
    boundaries: list[Boundary],
    tolerance_seconds: float,
    min_devices: int,
) -> list[Incident]:
    """枚举合法候选窗并做全局最优选择，返回按开始时间排序的共模事件。"""
    return select_incidents(
        enumerate_incidents(boundaries, tolerance_seconds, min_devices)
    )
