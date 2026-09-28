"""跨展柜环境传播分析（cross-case propagation，纯计算，只读）。

把两个展柜（source / target）在重叠时段内的同一维（温度 / 湿度）
读数各自视为**分段线性**曲线，在统一的**固定步长网格**上插值，然后枚举
一组正负时滞（lag，秒），用**成对有效点**计算皮尔逊相关系数：

- 网格点落在某展柜覆盖范围之外、或其所依赖的相邻读数间隔**大于**
  ``max_gap_seconds`` 时，该展柜在该点无效（恰好等于 ``max_gap``
  仍视为连续）；
- 时滞单位为 ``step_seconds`` 的整数倍，范围
  ``[-max_lag_seconds, +max_lag_seconds]``；
- **正时滞表示 source 领先 target**：比较 source 在 ``t`` 与 target 在
  ``t + lag`` 的值（在网格上即 source[i] 与 target[i+k] 配对）；
- 只统计两侧同时有效的**成对点**；成对点数不足或任一序列为常量时，
  该时滞**不可用**（``unavailable``），绝不产生 NaN。

择优规则（依次）：相关绝对值更大 → 有效成对点数更多 → 时滞绝对值
更小 → 时滞值更小。全部计算确定性，相同输入恒得相同输出。

时间参数边界：``step_seconds`` / ``max_lag_seconds`` 先按微秒四舍五入
取整为**生效值**，网格、时滞秒数与步长回显均使用该生效步长；非有限
（NaN / ±inf）时间参数、或取整后为 0 的正步长（小于半微秒）抛
:class:`PropagationParamsError`。在分配网格与时滞明细之前，按生效步长
预估重叠网格点数（≤ 10000）、时滞行数（≤ 201）及两者乘积（配对
工作量 ≤ 1 000 000），超限同样抛 :class:`PropagationParamsError`
（接口层转为 422；恰好等于上限仍可计算）。
"""

from __future__ import annotations

import datetime as dt
import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Protocol

TEMPERATURE = "temperature"
HUMIDITY = "humidity"

# 传播方向
SOURCE_TO_TARGET = "source_to_target"  # 最佳时滞 > 0：source 领先 target
TARGET_TO_SOURCE = "target_to_source"  # 最佳时滞 < 0：target 领先 source
SYNCHRONOUS = "synchronous"  # 最佳时滞 = 0：同步
UNAVAILABLE = "unavailable"  # 没有任何可用时滞

# 不可用原因
REASON_NO_OVERLAP = "no_overlap"  # 两柜（在查询窗内）无重叠时段
REASON_INSUFFICIENT = "insufficient_samples"  # 成对有效点少于 2
REASON_CONSTANT = "constant_series"  # 任一成对序列方差为 0

# 皮尔逊相关最少成对点数
MIN_PAIRS = 2

# 资源上限：在分配网格与时滞明细之前按生效步长预估，超限抛
# PropagationParamsError（接口层转 422）；恰好等于上限仍可计算
MAX_GRID_POINTS = 10_000  # 重叠时段网格点数上限
MAX_LAG_ROWS = 201  # 枚举时滞行数上限（2⌊max_lag/step⌋+1）
MAX_PAIR_WORK = 1_000_000  # 网格点数 × 时滞行数（配对工作量）上限


class PropagationParamsError(ValueError):
    """传播分析时间参数非法或超出资源上限（接口层转为 422）。

    ``parameter`` 指出触发拒绝的参数或限量维度；``limit`` / ``actual``
    在超限场景给出上限值与分配前的实际预估值。
    """

    def __init__(
        self,
        message: str,
        *,
        parameter: str,
        limit: int | None = None,
        actual: int | None = None,
    ) -> None:
        super().__init__(message)
        self.parameter = parameter
        self.limit = limit
        self.actual = actual


class _SampleLike(Protocol):
    ts: dt.datetime
    temperature: float
    humidity: float


@dataclass(frozen=True)
class LagCorrelation:
    """单个时滞的相关计算结果。

    ``available=False`` 时 ``correlation`` 为 ``None``（绝不产生 NaN），
    ``reason`` 给出不可用原因。
    """

    lag_seconds: float
    available: bool
    pair_count: int
    correlation: float | None
    reason: str | None


@dataclass(frozen=True)
class PropagationResult:
    """跨展柜传播分析结果。

    lags 按时滞升序排列（含 0）；``best_*`` 在没有任何可用时滞时为
    ``None`` / :data:`UNAVAILABLE`。
    """

    dimension: str
    start: dt.datetime
    end: dt.datetime
    overlap_start: dt.datetime | None
    overlap_end: dt.datetime | None
    step_seconds: float
    max_lag_seconds: float
    max_gap_seconds: float
    grid_point_count: int
    source_valid_count: int
    target_valid_count: int
    lags: tuple[LagCorrelation, ...]
    best_lag_seconds: float | None
    best_correlation: float | None
    direction: str


def _dimension_value(sample: _SampleLike, dimension: str) -> float:
    if dimension == TEMPERATURE:
        return sample.temperature
    if dimension == HUMIDITY:
        return sample.humidity
    raise ValueError(f"未知维度: {dimension}")  # pragma: no cover - 接口层已枚举校验


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


def normalize_time_params(
    step_seconds: float, max_lag_seconds: float, max_gap_seconds: float
) -> tuple[int, int]:
    """校验时间参数并把步长 / 最大时滞统一到微秒精度。

    返回 ``(step_us, max_lag_us)``——后续网格生成与时滞枚举的唯一依据
    （生效步长）。拒绝条件（抛 :class:`PropagationParamsError`，接口层
    转 422）：

    - 任一参数非有限（NaN / ±inf）、步长非正、最大时滞为负、最大间隔非正；
    - 正步长按微秒四舍五入取整后为 0（小于半微秒）——否则网格取整会除零。
    """
    for name, value in (
        ("step_seconds", step_seconds),
        ("max_lag_seconds", max_lag_seconds),
        ("max_gap_seconds", max_gap_seconds),
    ):
        if not math.isfinite(value):
            raise PropagationParamsError(
                f"{name} 必须为有限数值（收到 {value!r}）", parameter=name
            )
    if step_seconds <= 0.0:
        raise PropagationParamsError("step_seconds 必须为正", parameter="step_seconds")
    if max_lag_seconds < 0.0:
        raise PropagationParamsError(
            "max_lag_seconds 必须非负", parameter="max_lag_seconds"
        )
    if max_gap_seconds <= 0.0:
        raise PropagationParamsError(
            "max_gap_seconds 必须为正", parameter="max_gap_seconds"
        )
    try:
        step_us = int(round(step_seconds * 1_000_000))
    except OverflowError:
        raise PropagationParamsError(
            "step_seconds 超出可表示范围", parameter="step_seconds"
        ) from None
    try:
        max_lag_us = int(round(max_lag_seconds * 1_000_000))
    except OverflowError:
        raise PropagationParamsError(
            "max_lag_seconds 超出可表示范围", parameter="max_lag_seconds"
        ) from None
    if step_us <= 0:
        raise PropagationParamsError(
            "step_seconds 取整到微秒后为 0（正步长不得小于半微秒）",
            parameter="step_seconds",
        )
    return step_us, max_lag_us


def _build_grid(start: dt.datetime, end: dt.datetime, step_us: int) -> list[dt.datetime]:
    """从 ``start`` 起按固定微秒步长生成不超过 ``end`` 的等距网格（含两端边界）。"""
    if end < start:
        return []
    span_us = (end - start) // dt.timedelta(microseconds=1)
    n = span_us // step_us
    return [start + dt.timedelta(microseconds=step_us * k) for k in range(n + 1)]


def _grid_point_count(start: dt.datetime, end: dt.datetime, step_us: int) -> int:
    """不分配网格，仅计算将生成的网格点数（与 :func:`_build_grid` 同一公式）。"""
    if end < start:
        return 0
    span_us = (end - start) // dt.timedelta(microseconds=1)
    return span_us // step_us + 1


def _series(
    samples: list[_SampleLike], dimension: str
) -> tuple[list[dt.datetime], list[float]]:
    ordered = sorted(samples, key=lambda s: s.ts)
    return [s.ts for s in ordered], [_dimension_value(s, dimension) for s in ordered]


def _pearson(xs: list[float], ys: list[float]) -> tuple[float | None, str | None]:
    """成对序列的皮尔逊相关；样本不足或任一序列常量时不可用（不产生 NaN）。"""
    n = len(xs)
    if n < MIN_PAIRS:
        return None, REASON_INSUFFICIENT
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) * (x - mx) for x in xs)
    syy = sum((y - my) * (y - my) for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None, REASON_CONSTANT
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = sxy / math.sqrt(sxx * syy)
    # 数值误差兜底：相关系数必落在 [-1, 1]
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    return r, None


def _lag_row_count(max_lag_us: int, step_us: int) -> int:
    """时滞枚举的行数：``2⌊max_lag/step⌋ + 1``（与 :func:`_lag_range` 同公式）。"""
    return 2 * (max_lag_us // step_us) + 1


def _check_limits(grid_point_count: int, lag_row_count: int) -> None:
    """分配前的规模上限校验（按生效步长预估；恰好等于上限放行）。"""
    if grid_point_count > MAX_GRID_POINTS:
        raise PropagationParamsError(
            f"重叠时段网格点数 {grid_point_count} 超过上限 {MAX_GRID_POINTS}",
            parameter="grid_point_count",
            limit=MAX_GRID_POINTS,
            actual=grid_point_count,
        )
    if lag_row_count > MAX_LAG_ROWS:
        raise PropagationParamsError(
            f"枚举时滞行数 {lag_row_count} 超过上限 {MAX_LAG_ROWS}",
            parameter="lag_row_count",
            limit=MAX_LAG_ROWS,
            actual=lag_row_count,
        )
    pair_work = grid_point_count * lag_row_count
    if pair_work > MAX_PAIR_WORK:
        raise PropagationParamsError(
            f"网格点数 × 时滞行数的配对工作量 {pair_work} 超过上限 {MAX_PAIR_WORK}",
            parameter="pair_work",
            limit=MAX_PAIR_WORK,
            actual=pair_work,
        )


def _lag_range(max_lag_us: int, step_us: int) -> range:
    """枚举 ±⌊max_lag / step⌋ 内的整数步（含 0）；max_lag < step 时仅 0。"""
    kmax = max_lag_us // step_us
    return range(-kmax, kmax + 1)


def _select_best(
    lags: list[LagCorrelation],
) -> tuple[float | None, float | None]:
    """按 |r|↓、成对点数↓、|lag|↓、lag↓ 择优；无可用时滞返回 (None, None)。"""
    best: LagCorrelation | None = None
    best_key: tuple[float, int, float, float] | None = None
    for item in lags:
        if not item.available or item.correlation is None:
            continue
        key = (
            -abs(item.correlation),
            -item.pair_count,
            abs(item.lag_seconds),
            item.lag_seconds,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = item
    if best is None:
        return None, None
    return best.lag_seconds, best.correlation


def compute_propagation(
    source_samples: list[_SampleLike],
    target_samples: list[_SampleLike],
    dimension: str,
    start: dt.datetime,
    end: dt.datetime,
    step_seconds: float,
    max_lag_seconds: float,
    max_gap_seconds: float,
) -> PropagationResult:
    """计算两展柜在重叠时段上的分维时滞相关与传播方向。

    samples 无需预先排序；``start`` / ``end`` 已归一化为 UTC 且
    ``start <= end``（由接口层校验）。步长与最大时滞先经
    :func:`normalize_time_params` 统一到微秒精度——网格、时滞秒数与
    ``step_seconds`` 回显均使用该生效步长；非法时间参数或预估规模
    超限（网格点数 / 时滞行数 / 配对工作量）抛
    :class:`PropagationParamsError`，且在分配任何网格 / 明细之前抛出。
    """
    step_us, max_lag_us = normalize_time_params(
        step_seconds, max_lag_seconds, max_gap_seconds
    )
    effective_step_seconds = step_us / 1_000_000

    src_times, src_values = _series(source_samples, dimension)
    tgt_times, tgt_values = _series(target_samples, dimension)

    overlap_start = overlap_end = None
    planned_grid_points = 0

    if src_times and tgt_times:
        lo = max(start, src_times[0], tgt_times[0])
        hi = min(end, src_times[-1], tgt_times[-1])
        if lo <= hi:
            overlap_start, overlap_end = lo, hi
            planned_grid_points = _grid_point_count(lo, hi, step_us)

    # 分配前上限校验：时滞行数不依赖是否存在重叠（无重叠仍会生成全部明细），
    # 故两项预估与配对工作量都在此处一次性校验
    lag_row_count = _lag_row_count(max_lag_us, step_us)
    _check_limits(planned_grid_points, lag_row_count)

    grid: list[dt.datetime] = []
    source_grid: list[float | None] = []
    target_grid: list[float | None] = []
    if overlap_start is not None and overlap_end is not None:
        grid = _build_grid(overlap_start, overlap_end, step_us)
        source_grid = [
            _interpolate_at(src_times, src_values, t, max_gap_seconds)
            for t in grid
        ]
        target_grid = [
            _interpolate_at(tgt_times, tgt_values, t, max_gap_seconds)
            for t in grid
        ]

    # 网格等距：source[i] 与 target[i+k] 即 source(t) 与 target(t + k·step)；
    # 正时滞 k>0 表示 source 领先 target
    results: list[LagCorrelation] = []
    n = len(grid)
    for k in _lag_range(max_lag_us, step_us):
        # 先乘微秒整数再换秒：保证时滞为生效步长的精确整数倍
        lag_seconds = (k * step_us) / 1_000_000
        xs: list[float] = []
        ys: list[float] = []
        if n:
            i_lo = max(0, -k)
            i_hi = min(n, n - k)
            for i in range(i_lo, i_hi):
                sv = source_grid[i]
                tv = target_grid[i + k]
                if sv is None or tv is None:
                    continue
                xs.append(sv)
                ys.append(tv)

        if not n:
            correlation, reason = None, REASON_NO_OVERLAP
        else:
            correlation, reason = _pearson(xs, ys)
        results.append(
            LagCorrelation(
                lag_seconds=lag_seconds,
                available=correlation is not None,
                pair_count=len(xs),
                correlation=correlation,
                reason=reason,
            )
        )

    best_lag, best_r = _select_best(results)
    if best_lag is None:
        direction = UNAVAILABLE
    elif best_lag > 0:
        direction = SOURCE_TO_TARGET
    elif best_lag < 0:
        direction = TARGET_TO_SOURCE
    else:
        direction = SYNCHRONOUS

    return PropagationResult(
        dimension=dimension,
        start=start,
        end=end,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        step_seconds=effective_step_seconds,
        max_lag_seconds=max_lag_seconds,
        max_gap_seconds=max_gap_seconds,
        grid_point_count=n,
        source_valid_count=sum(v is not None for v in source_grid),
        target_valid_count=sum(v is not None for v in target_grid),
        lags=tuple(results),
        best_lag_seconds=best_lag,
        best_correlation=best_r,
        direction=direction,
    )
