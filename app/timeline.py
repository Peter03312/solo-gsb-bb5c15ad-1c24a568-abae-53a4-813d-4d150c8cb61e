"""时间线降采样：强制点保留 + 比例配额 + 确定性 LTTB。

规则（全部确定性，相同输入严格产生相同输出）：

1. 强制点（去重）：
   - 过滤后范围的首点、尾点；
   - 范围内温度/湿度各自的全局极小、极大值（并列时取最早出现）；
   - 已对齐锚点所在读数；
   - 分段边界所在读数；
   一个读数可同时命中多条强制原因，原因以 :data:`REASON_ORDER` 的固定
   顺序随该点返回。
2. 强制点去重后若超过 ``max_points``，抛 :class:`ForcedPointsOverflow`
   并携带 ``minimum_required``，由上层返回 422，绝不静默丢点。
3. 其余名额按“相邻强制点之间候选（内部）点数”的比例分配到各区间：
   先取整，余数按分数部分降序派发，分数相同或不可比时区间起始时间更早者
   优先（稳定分派）。
4. 各区间在其两个强制端点之间独立做确定性 LTTB；配额不小于候选数时
   全部候选点原样保留。LTTB 使用温、湿度两个量纲（按配置缩放等权）
   的三角形面积之和，所有并列（时间跨度并列、面积并列、最近邻选择）
   一律取下标更小者，保证可重放。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .config import settings

# ---------- 保留原因（同时也是 reasons 数组的固定排列顺序） ----------

REASON_RANGE_START = "range_start"
REASON_RANGE_END = "range_end"
REASON_TEMP_MIN = "temp_min"
REASON_TEMP_MAX = "temp_max"
REASON_HUMIDITY_MIN = "humidity_min"
REASON_HUMIDITY_MAX = "humidity_max"
REASON_ALIGNED_ANCHOR = "aligned_anchor"
REASON_SEGMENT_BOUNDARY = "segment_boundary"
REASON_LTTB = "lttb"

REASON_ORDER: tuple[str, ...] = (
    REASON_RANGE_START,
    REASON_RANGE_END,
    REASON_TEMP_MIN,
    REASON_TEMP_MAX,
    REASON_HUMIDITY_MIN,
    REASON_HUMIDITY_MAX,
    REASON_ALIGNED_ANCHOR,
    REASON_SEGMENT_BOUNDARY,
    REASON_LTTB,
)
_REASON_RANK = {r: i for i, r in enumerate(REASON_ORDER)}


class ForcedPointsOverflow(Exception):
    """强制点去重后数量已超过 ``max_points``。"""

    def __init__(self, minimum_required: int, max_points: int):
        super().__init__(
            f"强制保留点 {minimum_required} 个，已超过 max_points={max_points}"
        )
        self.minimum_required = minimum_required
        self.max_points = max_points


@dataclass(frozen=True)
class TimelinePoint:
    """降采样输入的单个读数（序列必须已按时间升序、无重复时间戳）。"""

    ts: dt.datetime
    temperature: float
    humidity: float


def _ordered_reasons(reasons: set[str]) -> list[str]:
    return sorted(reasons, key=lambda r: _REASON_RANK[r])


def _apportion(candidates: list[int], budget: int) -> list[int]:
    """把 ``budget`` 个名额按候选点数比例分派到各区间（最大余数法）。

    先按比例取整，余数按分数部分降序派发；分数相同时区间下标更小
    （起始时间更早）者优先。预算大于候选总数时多余名额作废；候选
    总数为 0 或没有预算时返回全 0。
    """
    total = sum(candidates)
    n = len(candidates)
    if total == 0 or budget <= 0:
        return [0] * n
    budget = min(budget, total)

    quotas = [budget * c // total for c in candidates]
    remainder = budget - sum(quotas)
    if remainder:
        # 分数部分（整数余数）降序；并列时下标升序（区间起始时间升序）
        order = sorted(range(n), key=lambda i: (-(budget * candidates[i] % total), i))
        for i in order:
            if remainder == 0:
                break
            quotas[i] += 1
            remainder -= 1
    return quotas


def _triangle_area_2d(
    p: tuple[float, float, float],
    q: tuple[float, float, float],
    r: tuple[float, float, float],
) -> tuple[float, float]:
    """(x, 缩放温度, 缩放湿度) 三点在温度/湿度平面上的两倍有向面积绝对值。"""
    px, pt, ph = p
    qx, qt, qh = q
    rx, rt, rh = r
    return (
        abs((qx - px) * (rt - qt) - (qt - pt) * (rx - qx)),
        abs((qx - px) * (rh - qh) - (qh - ph) * (rx - qx)),
    )


def _lttb_interval(
    points: list[TimelinePoint],
    lo: int,
    hi: int,
    quota: int,
) -> list[int]:
    """在强制端点 ``points[lo]``、``points[hi]`` 之间做确定性 LTTB。

    两端点必选；``quota`` 为内部（不含端点）保留点数，且满足
    ``0 < quota < hi - lo - 1``（配额足够的区间由调用方直接全保留）。

    内部点按经典 LTTB 切成 ``quota`` 个桶（多出的点并入最后一个
    中间桶）；选点时以“下一桶的平均点”（最后一个桶则以尾部强制点）
    为锚，比较温、湿度两个平面上三角形两倍面积之和，全部并列取
    下标更小者。
    """
    interior_count = hi - lo - 1

    t0 = points[lo].ts

    def coord(i: int) -> tuple[float, float, float]:
        p = points[i]
        return (
            (p.ts - t0).total_seconds(),
            p.temperature / settings.TEMP_SCALE,
            p.humidity / settings.HUMIDITY_SCALE,
        )

    def average(members: list[int]) -> tuple[float, float, float]:
        return (
            sum((points[i].ts - t0).total_seconds() for i in members) / len(members),
            sum(points[i].temperature for i in members) / len(members) / settings.TEMP_SCALE,
            sum(points[i].humidity for i in members) / len(members) / settings.HUMIDITY_SCALE,
        )

    # 分桶：bucket_of[j] 为内部第 j 点（全局 lo+1+j）所属桶号
    bucket_of: list[int] = [0] * interior_count
    for j in range(interior_count):
        b = ((j + 1) * quota) // interior_count
        bucket_of[j] = min(b, quota - 1)
    buckets = [
        [lo + 1 + j for j, bb in enumerate(bucket_of) if bb == b]
        for b in range(quota)
    ]

    selected: list[int] = [lo]
    a = lo
    for b, members in enumerate(buckets):
        if b == quota - 1:
            anchor = coord(hi)
        else:
            anchor = average(buckets[b + 1])

        best_i = -1
        best_score = -1.0
        for i in members:  # members 已按下标升序：并列时首个即最小下标
            at_t, at_h = _triangle_area_2d(coord(a), coord(i), anchor)
            score = at_t + at_h
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        a = best_i

    selected.append(hi)
    return selected


def downsample_timeline(
    points: list[TimelinePoint],
    forced: dict[int, set[str]],
    max_points: int,
) -> list[tuple[int, list[str]]]:
    """按规则降采样，返回 ``[(全局下标, [保留原因, ...]), ...]``（下标升序）。

    ``forced`` 给出每个强制点的原因集合；其下标必须对应 ``points``。
    当去重后的强制点数量超过 ``max_points`` 时抛
    :class:`ForcedPointsOverflow`。
    """
    n = len(points)
    forced_idx = sorted(forced)
    forced_count = len(forced_idx)

    if forced_count > max_points:
        raise ForcedPointsOverflow(forced_count, max_points)
    if n == 0:
        return []
    if n <= max_points:
        return [
            (i, _ordered_reasons(forced.get(i, {REASON_LTTB})))
            for i in range(n)
        ]

    budget = max_points - forced_count

    # 相邻强制点之间的区间（含端点），按候选内部点数比例分配名额
    spans: list[tuple[int, int]] = []
    for k in range(len(forced_idx) - 1):
        spans.append((forced_idx[k], forced_idx[k + 1]))
    candidates = [hi - lo - 1 for lo, hi in spans]
    quotas = _apportion(candidates, budget)

    selected: dict[int, set[str]] = {i: set(forced[i]) for i in forced_idx}
    for (lo, hi), quota in zip(spans, quotas):
        interior = hi - lo - 1
        if interior == 0:
            continue
        if quota >= interior:
            for i in range(lo + 1, hi):
                selected[i] = {REASON_LTTB}
        elif quota <= 0:
            continue
        else:
            for i in _lttb_interval(points, lo, hi, quota):
                if i not in selected:
                    selected[i] = {REASON_LTTB}

    return [(i, _ordered_reasons(reasons)) for i, reasons in sorted(selected.items())]
