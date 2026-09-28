"""时间线降采样算法单测：配额分派与确定性 LTTB。"""

from __future__ import annotations

import datetime as dt

import pytest

from app.timeline import (
    REASON_ALIGNED_ANCHOR,
    REASON_HUMIDITY_MAX,
    REASON_HUMIDITY_MIN,
    REASON_LTTB,
    REASON_RANGE_END,
    REASON_RANGE_START,
    REASON_SEGMENT_BOUNDARY,
    REASON_TEMP_MAX,
    REASON_TEMP_MIN,
    ForcedPointsOverflow,
    TimelinePoint,
    _apportion,
    downsample_timeline,
)

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


def _point(i: int, t: float, h: float) -> TimelinePoint:
    return TimelinePoint(BASE + dt.timedelta(minutes=10 * i), t, h)


def _series(vals: list[tuple[float, float]]) -> list[TimelinePoint]:
    return [_point(i, t, h) for i, (t, h) in enumerate(vals)]


# ---------- 名额比例分派（最大余数法，余数稳定分派） ----------


def test_apportion_proportional():
    assert _apportion([3, 0, 3], 2) == [1, 0, 1]
    assert _apportion([1, 3], 2) == [1, 1]
    assert _apportion([5, 5, 5], 4) == [2, 1, 1]
    # 候选数为 0 的区间不参与余数分派
    assert _apportion([0, 4], 2) == [0, 2]
    assert _apportion([10], 0) == [0]
    assert _apportion([], 3) == []


def test_apportion_remainder_tie_prefers_earlier_interval():
    # 2,2 分 3 个名额：各 1.5，余数 1 个，并列时起始更早的区间得到
    assert _apportion([2, 2], 3) == [2, 1]
    # 3,3,3 分 5 个名额：各 1，余数 2 个，前两个区间先得
    assert _apportion([3, 3, 3], 5) == [2, 2, 1]


def test_apportion_never_exceeds_candidates():
    # [1, 100] 分 50：比例 0.495 / 49.5，余数按分数派给第二区间
    assert _apportion([1, 100], 50) == [0, 50]
    # 预算极大时受候选数封顶，多余名额作废
    assert _apportion([2, 3], 100) == [2, 3]


# ---------- 强制点与预算 ----------


def test_empty_series():
    assert downsample_timeline([], {}, 10) == []


def test_all_returned_when_within_budget():
    points = _series([(20.0, 55.0), (21.0, 56.0), (19.0, 50.0)])
    chosen = downsample_timeline(points, {0: {REASON_RANGE_START}, 2: {REASON_RANGE_END}}, 10)
    assert [i for i, _ in chosen] == [0, 1, 2]
    assert chosen[1][1] == [REASON_LTTB]


def test_forced_overflow_raises_minimum_required():
    points = _series([(20.0 + i, 55.0) for i in range(6)])
    forced = {
        0: {REASON_RANGE_START},
        5: {REASON_RANGE_END},
        1: {REASON_TEMP_MIN},
        2: {REASON_TEMP_MAX},
    }
    with pytest.raises(ForcedPointsOverflow) as ei:
        downsample_timeline(points, forced, 3)
    assert ei.value.minimum_required == 4
    assert ei.value.max_points == 3
    # 恰好等于下限可以通过
    chosen = downsample_timeline(points, forced, 4)
    assert len(chosen) == 4


def test_forced_reasons_merged_in_fixed_order():
    points = _series([(20.0, 55.0), (20.0, 55.0)])
    forced = {
        0: {
            REASON_SEGMENT_BOUNDARY,
            REASON_ALIGNED_ANCHOR,
            REASON_TEMP_MIN,
            REASON_TEMP_MAX,
            REASON_HUMIDITY_MIN,
            REASON_HUMIDITY_MAX,
            REASON_RANGE_START,
        }
    }
    chosen = downsample_timeline(points, forced, 10)
    assert chosen[0][1] == [
        REASON_RANGE_START,
        REASON_TEMP_MIN,
        REASON_TEMP_MAX,
        REASON_HUMIDITY_MIN,
        REASON_HUMIDITY_MAX,
        REASON_ALIGNED_ANCHOR,
        REASON_SEGMENT_BOUNDARY,
    ]


# ---------- LTTB 区间选点 ----------


def test_lttb_keeps_endpoints_and_fills_quota():
    # 单调 V 型：端点固定，内部用一个明显尖峰吸引 LTTB
    vals = [(20.0, 55.0), (20.0, 55.0), (30.0, 80.0), (20.0, 55.0),
            (20.0, 55.0), (10.0, 30.0), (20.0, 55.0)]
    points = _series(vals)
    forced = {0: {REASON_RANGE_START}, 6: {REASON_RANGE_END}}
    chosen = downsample_timeline(points, forced, 4)
    idxs = [i for i, _ in chosen]
    assert idxs[0] == 0 and idxs[-1] == 6
    assert len(idxs) == 4 and idxs == sorted(set(idxs))
    # 两个极端点（2、5）构成最大三角形，必被选中
    assert set(idxs) >= {0, 2, 5, 6}


def test_lttb_deterministic_under_ties():
    # 完全平坦：所有面积并列，应稳定选择（并列取桶内最小下标）且两次一致
    points = _series([(20.0, 55.0)] * 10)
    forced = {0: {REASON_RANGE_START}, 9: {REASON_RANGE_END}}
    first = downsample_timeline(points, forced, 5)
    second = downsample_timeline(points, forced, 5)
    assert first == second
    idxs = [i for i, _ in first]
    assert idxs == sorted(set(idxs))
    assert idxs[0] == 0 and idxs[-1] == 9
    assert len(idxs) == 5


def test_quota_split_across_intervals_proportional():
    # 强制点把序列切成 2 + 6 两个候选区间；4 个候选名额按 1:3 分派
    points = _series([(20.0 + i * 0.1, 55.0) for i in range(10)])
    forced = {
        0: {REASON_RANGE_START},
        2: {REASON_ALIGNED_ANCHOR},
        9: {REASON_RANGE_END},
    }
    chosen = downsample_timeline(points, forced, 7)
    idxs = {i for i, _ in chosen}
    assert len(chosen) == 7
    assert {0, 2, 9} <= idxs
    # 第一区间内部仅 1 个候选点（下标 1），比例分派后必保留
    assert 1 in idxs
    # 第二区间保留 3 个内部点
    second_kept = [i for i in idxs if 3 <= i <= 8]
    assert len(second_kept) == 3


def test_output_sorted_and_reasons_tagged():
    vals = [(20.0, 55.0), (21.0, 56.0), (28.0, 72.0), (19.0, 54.0),
            (20.0, 55.0), (20.5, 60.0), (20.2, 55.0), (20.1, 40.0),
            (20.3, 55.0), (20.0, 80.0), (20.0, 55.0), (17.0, 52.0)]
    points = _series(vals)
    forced = {
        0: {REASON_RANGE_START},
        11: {REASON_RANGE_END},
        3: {REASON_TEMP_MIN},
        2: {REASON_TEMP_MAX},
        7: {REASON_HUMIDITY_MIN},
        9: {REASON_HUMIDITY_MAX},
        5: {REASON_ALIGNED_ANCHOR, REASON_SEGMENT_BOUNDARY},
    }
    chosen = downsample_timeline(points, forced, 9)
    idxs = [i for i, _ in chosen]
    assert idxs == sorted(idxs)
    assert len(chosen) == 9
    for fi in forced:
        assert fi in idxs
    # 非强制点只带 lttb 原因
    for i, reasons in chosen:
        if i not in forced:
            assert reasons == [REASON_LTTB]
