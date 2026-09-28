"""确定性变点分段。

采用带惩罚项的最优分段（精确动态规划，等价于一维 PELT 在穷举
剪枝关闭时的结果）：

    最小化  sum_segments Cost(seg) + beta * 段数

其中 Cost 为温湿度（各自量纲缩放后）按权重合成的段内平方误差和
(SSE)，锚点作为强制边界把时间线切成若干区间，区间内部独立做
带最小段长约束的 DP。所有输入相同时输出严格一致，不含随机过程。
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

from .config import settings

_EPS = 1e-12


@dataclass(frozen=True)
class Sample:
    ts: dt.datetime
    temperature: float
    humidity: float
    report_count: int = 1


@dataclass
class Segment:
    index: int
    start_ts: dt.datetime
    end_ts: dt.datetime
    sample_count: int
    mean_temperature: float
    mean_humidity: float
    temperature_std: float
    humidity_std: float
    # 相对上一段的跳变（第一段恒为 0）
    delta_temperature: float
    delta_humidity: float
    # 缩放加权后的欧氏变化幅度
    magnitude: float
    # baseline / high / medium / low
    confidence: str
    start_anchored: bool
    end_anchored: bool
    cost: float


@dataclass
class SegmentationResult:
    device_sn: str
    segments: list[Segment] = field(default_factory=list)
    sample_count: int = 0
    anchor_count: int = 0
    total_cost: float = 0.0
    penalty: float = settings.PENALTY
    # 未能对齐到任何读数（或落在端点）的锚点时间戳
    unmatched_anchors: list[dt.datetime] = field(default_factory=list)


class _Prefix:
    """单一一维特征的前缀统计，区间 SSE 可 O(1) 取得。"""

    def __init__(self, values: list[float]):
        n = len(values)
        self.s: list[float] = [0.0] * (n + 1)
        self.sq: list[float] = [0.0] * (n + 1)
        for i, v in enumerate(values):
            self.s[i + 1] = self.s[i] + v
            self.sq[i + 1] = self.sq[i] + v * v

    def sse(self, a: int, b: int) -> float:
        length = b - a
        if length <= 0:
            return 0.0
        total = self.sq[b] - self.sq[a]
        mean_sum = self.s[b] - self.s[a]
        return max(0.0, total - mean_sum * mean_sum / length)

    def mean(self, a: int, b: int) -> float:
        return (self.s[b] - self.s[a]) / (b - a)

    def pstd(self, a: int, b: int) -> float:
        """物理单位的总体标准差。"""
        length = b - a
        if length <= 1:
            return 0.0
        return math.sqrt(max(0.0, self.sse(a, b) / length))


def _snap_anchors(
    samples: list[Sample], anchor_ts: list[dt.datetime], tolerance: float
) -> tuple[list[int], list[dt.datetime]]:
    """把锚点时间戳对齐为最近的合法边界下标（1..n-1）。"""
    n = len(samples)
    boundary_set: set[int] = set()
    unmatched: list[dt.datetime] = []
    times = [s.ts for s in samples]

    for anchor in sorted(anchor_ts):
        if n < 2:
            unmatched.append(anchor)
            continue
        # 单调可二分，但展柜读数规模很小，直接扫描更直观
        best_k, best_diff = -1, math.inf
        for k in range(1, n):
            diff = abs((times[k] - anchor).total_seconds())
            if diff < best_diff:
                best_diff, best_k = diff, k
        if best_diff <= tolerance:
            boundary_set.add(best_k)
        else:
            unmatched.append(anchor)

    return sorted(boundary_set), unmatched


def _partition_block(
    a: int,
    b: int,
    pt: _Prefix,
    ph: _Prefix,
    beta: float,
    min_size: int,
) -> tuple[list[tuple[int, int]], float]:
    """对区间 [a, b) 做带惩罚与最小段长的精确 DP，返回 (段下标区间, 总代价含惩罚)。"""
    m = b - a
    if m == 0:
        return [], 0.0

    # 小区间放宽最小段长，保证锚点强制切开后仍能给出结果
    block_min = min(min_size, m)

    def cost(la: int, lb: int) -> float:
        return (
            settings.TEMP_WEIGHT * pt.sse(a + la, a + lb)
            + settings.HUMIDITY_WEIGHT * ph.sse(a + la, a + lb)
        )

    inf = math.inf
    f: list[float] = [inf] * (m + 1)
    prev: list[int] = [-1] * (m + 1)
    f[0] = 0.0

    for j in range(block_min, m + 1):
        best, best_t = inf, -1
        # t = 本段起点（局部下标）；第一段允许 t=0
        t_lo = 0
        t_hi = j - block_min
        for t in range(t_lo, t_hi + 1):
            if f[t] == inf:
                continue
            candidate = f[t] + cost(t, j) + beta
            if candidate < best - _EPS:
                best, best_t = candidate, t
            elif abs(candidate - best) <= _EPS and t > best_t:
                # 代价相同：取更少段（更晚的切点），保证确定性与稳定性
                best_t = t
        f[j], prev[j] = best, best_t

    # 回溯
    cuts: list[tuple[int, int]] = []
    j = m
    while j > 0:
        t = prev[j]
        cuts.append((a + t, a + j))
        j = t
    cuts.reverse()
    return cuts, f[m]


def _confidence(
    prev_seg: tuple[int, int] | None,
    cur_seg: tuple[int, int],
    pt: _Prefix,
    ph: _Prefix,
    is_anchor_boundary: bool,
) -> tuple[float, float, float, str, float]:
    """计算相对上一段的跳变与置信标记。"""
    a, b = cur_seg
    mean_t = pt.mean(a, b)
    mean_h = ph.mean(a, b)

    if prev_seg is None:
        return mean_t, mean_h, 0.0, "baseline", 0.0

    pa, pb = prev_seg
    prev_mean_t = pt.mean(pa, pb)
    prev_mean_h = ph.mean(pa, pb)
    delta_t = mean_t - prev_mean_t
    delta_h = mean_h - prev_mean_h

    magnitude = math.sqrt(
        settings.TEMP_WEIGHT * (delta_t / settings.TEMP_SCALE) ** 2
        + settings.HUMIDITY_WEIGHT * (delta_h / settings.HUMIDITY_SCALE) ** 2
    )

    # 相邻两段合并残差作为局部噪声估计（缩放加权空间）
    df = max(1, (pb - pa) + (b - a) - 2)
    resid = (
        settings.TEMP_WEIGHT
        * (pt.sse(pa, pb) + pt.sse(a, b))
        / settings.TEMP_SCALE**2
        + settings.HUMIDITY_WEIGHT
        * (ph.sse(pa, pb) + ph.sse(a, b))
        / settings.HUMIDITY_SCALE**2
    )
    noise = math.sqrt(max(0.0, resid) / df)
    score = magnitude / max(noise, _EPS)

    n_prev, n_cur = pb - pa, b - a
    if is_anchor_boundary:
        # 锚点是人工校准确认的真实变点，直接判高置信
        confidence = "high"
    elif (
        n_prev >= settings.CONF_HIGH_MIN_SAMPLES
        and n_cur >= settings.CONF_HIGH_MIN_SAMPLES
        and score >= settings.CONF_HIGH_SIGMAS
        and magnitude >= settings.CONF_MIN_MAGNITUDE
    ):
        confidence = "high"
    elif score >= settings.CONF_MEDIUM_SIGMAS and magnitude >= settings.CONF_MIN_MAGNITUDE:
        confidence = "medium"
    else:
        confidence = "low"

    return mean_t, mean_h, magnitude, confidence, noise


def segment_samples(
    device_sn: str,
    samples: list[Sample],
    anchor_ts: list[dt.datetime] | None = None,
    penalty: float | None = None,
    min_segment_size: int | None = None,
    anchor_tolerance: float | None = None,
) -> SegmentationResult:
    """对单个展柜的读数序列执行确定性变点分段。

    samples 无需预先排序，内部按时间戳升序处理；同时间戳样本不应
    出现（重复上报应先经合并）。
    """
    beta = settings.PENALTY if penalty is None else penalty
    min_size = settings.MIN_SEGMENT_SIZE if min_segment_size is None else min_segment_size
    tolerance = settings.ANCHOR_TOLERANCE if anchor_tolerance is None else anchor_tolerance

    ordered = sorted(samples, key=lambda s: s.ts)
    n = len(ordered)
    result = SegmentationResult(device_sn=device_sn, sample_count=n, penalty=beta)

    if n == 0:
        return result

    temps = [s.temperature for s in ordered]
    hums = [s.humidity for s in ordered]
    pt = _Prefix(temps)
    ph = _Prefix(hums)

    boundaries, unmatched = _snap_anchors(ordered, sorted(anchor_ts or []), tolerance)
    result.anchor_count = len(boundaries)
    result.unmatched_anchors = unmatched

    # 以强制边界划分区间
    blocks: list[tuple[int, int]] = []
    start = 0
    for k in boundaries:
        if start < k:
            blocks.append((start, k))
        start = k
    if start < n:
        blocks.append((start, n))

    raw_segments: list[tuple[int, int]] = []
    total_cost = 0.0
    for ba, bb in blocks:
        cuts, block_cost = _partition_block(ba, bb, pt, ph, beta, min_size)
        raw_segments.extend(cuts)
        total_cost += block_cost
    result.total_cost = total_cost

    boundary_flags = {k: True for k in boundaries}
    for idx, (a, b) in enumerate(raw_segments):
        prev_seg = raw_segments[idx - 1] if idx > 0 else None
        mean_t, mean_h, magnitude, confidence, _noise = _confidence(
            prev_seg, (a, b), pt, ph, is_anchor_boundary=a in boundary_flags
        )
        prev_b = raw_segments[idx - 1][1] if idx > 0 else b
        seg = Segment(
            index=idx,
            start_ts=ordered[a].ts,
            end_ts=ordered[b - 1].ts,
            sample_count=b - a,
            mean_temperature=mean_t,
            mean_humidity=mean_h,
            temperature_std=pt.pstd(a, b),
            humidity_std=ph.pstd(a, b),
            delta_temperature=mean_t - pt.mean(raw_segments[idx - 1][0], prev_b) if idx > 0 else 0.0,
            delta_humidity=mean_h - ph.mean(raw_segments[idx - 1][0], prev_b) if idx > 0 else 0.0,
            magnitude=magnitude,
            confidence=confidence,
            start_anchored=a in boundary_flags,
            end_anchored=b in boundary_flags,
            cost=(
                settings.TEMP_WEIGHT * pt.sse(a, b)
                + settings.HUMIDITY_WEIGHT * ph.sse(a, b)
            ),
        )
        result.segments.append(seg)

    return result
