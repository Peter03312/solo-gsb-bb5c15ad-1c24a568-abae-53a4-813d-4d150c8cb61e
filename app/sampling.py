"""采样覆盖率审计：周期应采时隙 + 容差内一对一匹配（纯计算，只读）。

以查询起点为**首个应采时刻**，按固定周期生成**不晚于终点**的应采时隙
（``t_i = start + i·period``，``t_i ≤ end``；起点不晚于终点时至少一个
时隙）。再取**容差扩展范围** ``[start − tolerance, 末时隙 + tolerance]``
内的已合并读数，按时间顺序做**一对一**贪心匹配：

- 读数可覆盖时隙 ``t_i`` 当且仅当 ``|读数时刻 − t_i| ≤ tolerance``
  （**含等于**）；
- 每条读数最多覆盖一个时隙，每个时隙最多接收一条读数；
- 每条读数优先匹配其容差窗口内**最早**（时刻最小）的未覆盖时隙。

连续的缺测时隙合并为缺口，返回缺口起止时隙时刻与缺测时隙数。

时间统一规则（避免浮点秒逐次构造带来的重复时隙与越界运算）：

- 周期 / 容差先**四舍五入到整数微秒**，之后所有运算只使用该生效值；
- 生效周期不足 1 微秒（正周期取整后为 0）拒绝（接口层转 422）；
- 应采时隙数**先按查询窗口算出**（``span_us // period_us + 1``），
  最多 :data:`MAX_EXPECTED_SLOTS` 个，超限拒绝（接口层转 422），
  先算数、不预分配；
- 周期大于窗口（含大到微秒整数无法表示的周期）时只生成起点一个时隙，
  绝不试算越界的下一时隙（极大正周期下旧实现会在 ``start + period``
  处溢出）。

匹配、覆盖率与连续缺口均引用同一生效时隙序列，全程确定性：相同输入
恒得相同输出；不修改任何输入。
"""

from __future__ import annotations

import datetime as dt
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Protocol

MICROSECONDS_PER_SECOND = 1_000_000
#: 应采时隙数硬上限：先算数、超限即 422，不构造时隙列表
MAX_EXPECTED_SLOTS = 100_000

# timedelta 的微秒表示范围（约 ±270 万年），超出即无法做时间位移
_TIMEDELTA_MAX_US = dt.timedelta.max // dt.timedelta(microseconds=1)


class SamplingParamsError(ValueError):
    """采样覆盖率入参非法（接口层统一转 422）。"""

    def __init__(
        self,
        message: str,
        *,
        parameter: str,
        limit: int | None = None,
        actual: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.parameter = parameter
        self.limit = limit
        self.actual = actual


class _SampleLike(Protocol):
    ts: dt.datetime


@dataclass(frozen=True)
class CoverageGap:
    """连续缺测时隙合并得到的缺口。"""

    start_ts: dt.datetime  # 缺口内首个缺测时隙的时刻
    end_ts: dt.datetime  # 缺口内末个缺测时隙的时刻
    missing_slots: int  # 缺口内连续缺测的时隙数


@dataclass(frozen=True)
class SamplingCoverageResult:
    expected_count: int  # 应采时隙数
    covered_count: int  # 已覆盖时隙数
    coverage: float  # 覆盖率 = covered / expected（expected 为 0 时取 0.0）
    gaps: list[CoverageGap]


@dataclass(frozen=True)
class ResolvedSamplingParams:
    """校验并归一化后的生效参数（时隙序列的唯一依据）。

    ``period_us`` 为 ``None`` 表示周期大到整数微秒也无法表示，此时
    :attr:`expected_count` 恒为 1（只生成起点时隙），周期不参与任何
    时间运算。``tolerance_us`` 为 ``None`` 表示容差超出 timedelta
    可表示范围，匹配窗口按 datetime 上下界饱和裁剪。
    """

    period_us: int | None
    tolerance_us: int | None
    expected_count: int
    period_effective_seconds: float  # 回显：生效周期（无法表示时为原值）
    tolerance_effective_seconds: float  # 回显：生效容差（无法表示时为原值）


def _round_microseconds(seconds: float) -> int | None:
    """秒值四舍五入为整数微秒；超出 timedelta 可表示范围时返回 ``None``。

    调用方需已确认 ``seconds`` 有限（NaN / ±inf 不进入本函数）。
    """
    try:
        micros = int(round(seconds * MICROSECONDS_PER_SECOND))
    except OverflowError:
        return None
    if micros > _TIMEDELTA_MAX_US:
        return None
    return micros


def resolve_sampling_params(
    start: dt.datetime,
    end: dt.datetime,
    period_seconds: float,
    tolerance_seconds: float,
) -> ResolvedSamplingParams:
    """校验入参并把周期 / 容差统一为整数微秒，先按窗口算出应采时隙数。

    抛 :class:`SamplingParamsError`（接口层转 422）的情形：

    - ``start > end``；
    - ``period_seconds`` 非有限或非正；``tolerance_seconds`` 非有限或为负；
    - 容差不小于周期（先按原值判，再按微秒生效值判，含等于）；
    - 生效周期不足 1 微秒（正周期取整后为 0）；
    - 应采时隙数超过 :data:`MAX_EXPECTED_SLOTS`。

    周期大于窗口（或无法用整数微秒表示）时只保留起点一个时隙，且不做
    任何可能越界的周期时间运算。
    """
    if start > end:
        raise SamplingParamsError("start 不能晚于 end", parameter="start")
    if not math.isfinite(period_seconds) or period_seconds <= 0.0:
        raise SamplingParamsError(
            "period_seconds 必须为正的有限数值", parameter="period_seconds"
        )
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0.0:
        raise SamplingParamsError(
            "tolerance_seconds 必须为非负的有限数值", parameter="tolerance_seconds"
        )
    # 原值先判一次：覆盖取整后无法比较的极大值组合
    if tolerance_seconds >= period_seconds:
        raise SamplingParamsError(
            "tolerance_seconds 必须小于 period_seconds",
            parameter="tolerance_seconds",
        )

    period_us = _round_microseconds(period_seconds)
    tolerance_us = _round_microseconds(tolerance_seconds)

    if period_us is not None and period_us <= 0:
        raise SamplingParamsError(
            "period_seconds 四舍五入到整数微秒后不足 1 微秒"
            "（生效周期至少为 1 微秒）",
            parameter="period_seconds",
        )
    # 生效值再判一次：例如周期 1.0000004s 与容差 1.0000006s 取整后相等
    if period_us is not None and tolerance_us is not None:
        if tolerance_us >= period_us:
            raise SamplingParamsError(
                f"生效容差（{tolerance_us} 微秒）必须严格小于生效周期"
                f"（{period_us} 微秒）",
                parameter="tolerance_seconds",
            )

    span_us = (end - start) // dt.timedelta(microseconds=1)

    # 周期大于窗口（含无法用整数微秒表示的极大周期）：只有起点时隙。
    # 先于此分支返回，避免 start + period 等越界时间运算（旧实现 500）。
    if period_us is None or period_us > span_us:
        return ResolvedSamplingParams(
            period_us=None,
            tolerance_us=tolerance_us,
            expected_count=1,
            period_effective_seconds=period_seconds,
            tolerance_effective_seconds=(
                tolerance_seconds
                if tolerance_us is None
                else tolerance_us / MICROSECONDS_PER_SECOND
            ),
        )

    # 先按窗口算时隙数量：i·period_us ≤ span_us 的 i 共 span//period + 1 个
    expected_count = span_us // period_us + 1
    if expected_count > MAX_EXPECTED_SLOTS:
        raise SamplingParamsError(
            f"应采时隙数 {expected_count} 超过上限 {MAX_EXPECTED_SLOTS}",
            parameter="expected_count",
            limit=MAX_EXPECTED_SLOTS,
            actual=expected_count,
        )

    return ResolvedSamplingParams(
        period_us=period_us,
        tolerance_us=tolerance_us,
        expected_count=expected_count,
        period_effective_seconds=period_us / MICROSECONDS_PER_SECOND,
        tolerance_effective_seconds=(
            tolerance_seconds
            if tolerance_us is None
            else tolerance_us / MICROSECONDS_PER_SECOND
        ),
    )


def generate_slots(
    start: dt.datetime, resolved: ResolvedSamplingParams
) -> list[dt.datetime]:
    """按生效参数生成应采时隙（升序）。

    数量与时刻完全由 :func:`resolve_sampling_params` 的结果决定：周期
    大于窗口时仅 ``[start]``；否则 ``start + i·period_us``
    （``i = 0 … expected_count−1``），整数微秒运算，不重复、不越界。
    """
    if resolved.period_us is None:
        return [start]
    period_us = resolved.period_us
    return [
        start + dt.timedelta(microseconds=period_us * i)
        for i in range(resolved.expected_count)
    ]


_AWARE_MIN = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
_AWARE_MAX = dt.datetime.max.replace(tzinfo=dt.timezone.utc)


def _shift_clamp(ts: dt.datetime, delta: dt.timedelta) -> dt.datetime:
    """时间位移；越出 datetime 可表示范围时饱和裁剪到上下界。"""
    try:
        return ts + delta
    except OverflowError:
        return _AWARE_MAX if delta > dt.timedelta(0) else _AWARE_MIN


def compute_sampling_coverage(
    samples: list[_SampleLike],
    *,
    start: dt.datetime,
    end: dt.datetime,
    resolved: ResolvedSamplingParams,
) -> SamplingCoverageResult:
    """周期时隙的采样覆盖率与连续缺测缺口（只读纯计算）。

    参数合法性由 :func:`resolve_sampling_params` 与接口层保证，本函数
    只消费已归一化的 ``resolved``。samples 无需预先排序、无需预先按
    范围过滤；只读取容差扩展范围内的读数，范围外读数不参与匹配。
    匹配、覆盖率与缺口共用 :func:`generate_slots` 生成的同一份时隙。
    """
    slots = generate_slots(start, resolved)
    expected = resolved.expected_count

    tolerance: dt.timedelta | None = (
        None
        if resolved.tolerance_us is None
        else dt.timedelta(microseconds=resolved.tolerance_us)
    )
    # 容差扩展范围：覆盖全部时隙匹配窗口的最小区间（大容差按上下界饱和）
    if tolerance is None:
        # 容差大到不可表示：任一读数都在唯一起点时隙的容差窗口内
        lo, hi = _AWARE_MIN, _AWARE_MAX
    else:
        lo = _shift_clamp(slots[0], -tolerance)
        hi = _shift_clamp(slots[-1], tolerance)
    readings = sorted(s.ts for s in samples if lo <= s.ts <= hi)

    covered = [False] * expected
    covered_count = 0
    for ts in readings:
        # 该读数可覆盖的时隙下标区间 [left, right)
        if tolerance is None:
            # 容差窗口饱和覆盖整个 datetime 范围：所有时隙都可被该读数覆盖
            left, right = 0, expected
        else:
            left = bisect_left(slots, _shift_clamp(ts, -tolerance))
            right = bisect_right(slots, _shift_clamp(ts, tolerance))
        for i in range(left, right):  # 优先最早的可覆盖时隙
            if not covered[i]:
                covered[i] = True
                covered_count += 1
                break

    gaps: list[CoverageGap] = []
    i = 0
    while i < expected:
        if covered[i]:
            i += 1
            continue
        j = i
        while j + 1 < expected and not covered[j + 1]:
            j += 1
        gaps.append(
            CoverageGap(
                start_ts=slots[i], end_ts=slots[j], missing_slots=j - i + 1
            )
        )
        i = j + 1

    return SamplingCoverageResult(
        expected_count=expected,
        covered_count=covered_count,
        coverage=covered_count / expected,
        gaps=gaps,
    )
