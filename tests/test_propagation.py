"""跨展柜环境传播分析：时滞相关纯算法单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import math
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine, init_db
from app.main import app
from app.propagation import (
    MAX_GRID_POINTS,
    MAX_LAG_ROWS,
    MAX_PAIR_WORK,
    REASON_CONSTANT,
    REASON_INSUFFICIENT,
    REASON_NO_OVERLAP,
    LagCorrelation,
    PropagationParamsError,
    compute_propagation,
)
from app.segmentation import Sample

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def _sample(minute: float, t: float, h: float = 50.0) -> Sample:
    return Sample(BASE + dt.timedelta(seconds=60 * minute), t, h)


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


# 非对称伪随机波形：平移后只有正确时滞相关为 1，线性斜坡不会产生多峰退化
WAVE = [20.0, 22.0, 21.0, 25.0, 24.0, 28.0, 27.0, 30.0, 26.0, 23.0, 21.0, 24.0]


def _wave_src():
    return [_sample(i, WAVE[i]) for i in range(12)]


def _compute(src, tgt, dimension="temperature", *, start=BASE, end=None,
             step=60.0, max_lag=300.0, max_gap=600.0):
    return compute_propagation(
        src,
        tgt,
        dimension,
        start=start,
        end=end if end is not None else BASE + dt.timedelta(seconds=60 * 13),
        step_seconds=step,
        max_lag_seconds=max_lag,
        max_gap_seconds=max_gap,
    )


# ---------- 纯算法：时滞方向 ----------


def test_positive_lag_means_source_leads_target():
    # target 取值整体后移 2 分钟：source 在 t 的值 == target 在 t+120s 的值
    src = _wave_src()
    tgt = [_sample(i + 2, WAVE[i]) for i in range(12)]
    res = _compute(src, tgt)
    assert res.best_lag_seconds == 120.0
    assert res.best_correlation == pytest.approx(1.0, abs=1e-9)
    assert res.direction == "source_to_target"
    # 最佳明细行与汇总一致
    best_row = next(l for l in res.lags if l.lag_seconds == 120.0)
    assert best_row.correlation == pytest.approx(1.0, abs=1e-9)


def test_negative_lag_means_target_leads_source():
    src = [_sample(i + 2, WAVE[i]) for i in range(12)]
    tgt = _wave_src()
    res = _compute(src, tgt)
    assert res.best_lag_seconds == -120.0
    assert res.direction == "target_to_source"


def test_zero_lag_is_synchronous():
    src = _wave_src()
    res = _compute(src, src, end=BASE + dt.timedelta(seconds=60 * 11), max_lag=120.0)
    assert res.best_lag_seconds == 0.0
    assert res.direction == "synchronous"
    assert res.best_correlation == pytest.approx(1.0, abs=1e-12)


def test_humidity_dimension_uses_humidity_values():
    hum = [40.0, 44.0, 41.0, 47.0, 43.0, 50.0, 42.0, 46.0]
    src = [_sample(i, 20.0, hum[i]) for i in range(8)]
    tgt = [_sample(i + 1, 20.0, hum[i]) for i in range(8)]
    res = _compute(
        src,
        tgt,
        "humidity",
        end=BASE + dt.timedelta(seconds=60 * 8),
        max_lag=180.0,
    )
    assert res.best_lag_seconds == 60.0
    assert res.direction == "source_to_target"


# ---------- 纯算法：网格、断档、有效性 ----------


def test_grid_clipped_to_overlap_and_floored():
    src = _wave_src()  # 覆盖 0..11 分钟
    tgt = [_sample(i + 2, WAVE[i]) for i in range(12)]  # 覆盖 2..13 分钟
    res = _compute(src, tgt)
    # 重叠 [2min, 11min]：10 个 60s 网格点
    assert res.overlap_start == BASE + dt.timedelta(seconds=120)
    assert res.overlap_end == BASE + dt.timedelta(seconds=660)
    assert res.grid_point_count == 10
    assert res.source_valid_count == 10
    assert res.target_valid_count == 10


def test_grid_does_not_extend_past_end_when_step_does_not_divide_window():
    res = _compute(
        _wave_src(),
        _wave_src(),
        end=BASE + dt.timedelta(seconds=630),  # 10.5 分钟
        step=240.0,  # 0,4,8 分钟；下一个 12 分钟超出
        max_lag=0.0,
    )
    assert res.grid_point_count == 3


def test_gap_larger_than_max_gap_invalidates_interior_points():
    src = _wave_src()
    # target 在 1→11 分钟有 10 分钟断档；max_gap=300
    tgt = [_sample(0, 20.0), _sample(1, 22.0), _sample(11, 21.0), _sample(12, 25.0)]
    res = _compute(src, tgt, end=BASE + dt.timedelta(seconds=60 * 12), max_lag=0.0, max_gap=300.0)
    # 重叠裁剪到 source 覆盖 [0,11]：网格 0..11；target 仅在读数点 0,1,11 有效
    assert res.grid_point_count == 12
    assert res.target_valid_count == 3
    lag0 = res.lags[0]
    assert lag0.pair_count == 3  # 0,1,11 两侧都有效


def test_gap_equal_to_max_gap_is_still_continuous():
    samples = [_sample(0, 20.0), _sample(10, 30.0)]
    res = _compute(
        samples,
        samples,
        end=BASE + dt.timedelta(seconds=600),
        step=300.0,
        max_lag=0.0,
        max_gap=600.0,  # 恰好等于读数间隔
    )
    assert res.target_valid_count == 3
    assert res.lags[0].available is True
    assert res.lags[0].correlation == pytest.approx(1.0, abs=1e-12)


# ---------- 纯算法：不可用场景绝不产生 NaN ----------


def test_no_overlap_marks_all_lags_unavailable():
    src = _wave_src()
    far = [_sample(60, 1.0), _sample(61, 2.0)]
    res = _compute(src, far, max_lag=120.0)
    assert res.direction == "unavailable"
    assert res.overlap_start is None
    assert res.overlap_end is None
    assert res.grid_point_count == 0
    assert res.best_lag_seconds is None
    assert res.best_correlation is None
    assert res.lags
    for lag in res.lags:
        assert lag.available is False
        assert lag.correlation is None
        assert lag.reason == REASON_NO_OVERLAP


def test_constant_series_is_unavailable_not_nan():
    const = [_sample(i, 20.0) for i in range(12)]
    res = _compute(const, const, end=BASE + dt.timedelta(seconds=60 * 11), max_lag=120.0)
    assert res.direction == "unavailable"
    for lag in res.lags:
        assert lag.available is False
        assert lag.correlation is None
        assert lag.reason == REASON_CONSTANT


def test_one_side_constant_pairs_is_constant_reason():
    src = _wave_src()
    tgt = [_sample(i, 50.0) for i in range(12)]
    res = _compute(src, tgt, end=BASE + dt.timedelta(seconds=60 * 11))
    assert all(not lag.available and lag.reason == REASON_CONSTANT for lag in res.lags)
    assert res.direction == "unavailable"


def test_insufficient_pairs_when_only_one_point_valid():
    # 两侧都只有一个读数点：网格仅 1 点，无法计算相关
    src = [_sample(0, 20.0)]
    tgt = [_sample(0, 21.0)]
    res = _compute(src, tgt, end=BASE, max_lag=0.0)
    lag0 = res.lags[0]
    assert lag0.pair_count == 1
    assert lag0.available is False
    assert lag0.correlation is None
    assert lag0.reason == REASON_INSUFFICIENT
    assert res.direction == "unavailable"


def test_gap_endpoints_two_nonconstant_points_still_correlate():
    # 读数间隔 660s > 600：中间插值点无效，但两个读数端点本身仍有效 → 2 对可计算
    src = [_sample(0, 20.0), _sample(11, 24.0)]
    tgt = [_sample(0, 21.0), _sample(11, 25.0)]
    res = _compute(src, tgt, end=BASE + dt.timedelta(seconds=60 * 11), max_lag=0.0)
    lag0 = res.lags[0]
    assert lag0.pair_count == 2
    assert lag0.available is True
    assert lag0.correlation == pytest.approx(1.0)


def test_no_nan_anywhere():
    scenarios = [
        (_wave_src(), [_sample(i + 2, WAVE[i]) for i in range(12)]),
        (_wave_src(), [_sample(60, 1.0), _sample(61, 2.0)]),
        ([_sample(i, 20.0) for i in range(12)], [_sample(i, 20.0) for i in range(12)]),
    ]
    for src, tgt in scenarios:
        res = _compute(src, tgt)
        for lag in res.lags:
            assert lag.correlation is None or not math.isnan(lag.correlation)
        assert res.best_correlation is None or not math.isnan(res.best_correlation)


# ---------- 纯算法：择优次序 ----------


def test_select_best_abs_correlation_first():
    rows = [
        LagCorrelation(0.0, True, 9, 0.9, None),
        LagCorrelation(120.0, True, 4, -0.95, None),
    ]
    from app.propagation import _select_best

    assert _select_best(rows) == (120.0, -0.95)


def test_select_best_pair_count_then_lag_then_sign():
    from app.propagation import _select_best

    # |r| 相同：成对点数更多
    assert _select_best([
        LagCorrelation(0.0, True, 4, 0.9, None),
        LagCorrelation(60.0, True, 5, 0.9, None),
    ]) == (60.0, 0.9)
    # |r|、点数相同：|lag| 更小
    assert _select_best([
        LagCorrelation(-60.0, True, 5, 0.9, None),
        LagCorrelation(0.0, True, 5, 0.9, None),
        LagCorrelation(60.0, True, 5, 0.9, None),
    ]) == (0.0, 0.9)
    # 全部相同仅符号相反：时滞值更小（负者）
    assert _select_best([
        LagCorrelation(-60.0, True, 5, 0.9, None),
        LagCorrelation(60.0, True, 5, 0.9, None),
    ]) == (-60.0, 0.9)
    # 无可用
    assert _select_best([LagCorrelation(0.0, False, 1, None, REASON_INSUFFICIENT)]) == (
        None,
        None,
    )


def test_lags_enumerated_ascending_and_bounded():
    src = _wave_src()
    res = _compute(src, src, end=BASE + dt.timedelta(seconds=60 * 11), max_lag=300.0)
    lags = [lag.lag_seconds for lag in res.lags]
    assert lags == sorted(lags)
    assert lags[0] == -300.0
    assert lags[-1] == 300.0
    assert 0.0 in lags

    # max_lag < step：仅枚举零时滞
    res_small = _compute(
        src, src, end=BASE + dt.timedelta(seconds=60 * 11), step=600.0, max_lag=100.0
    )
    assert [lag.lag_seconds for lag in res_small.lags] == [0.0]


def test_unsorted_input_deterministic():
    src = _wave_src()
    tgt = [_sample(i + 2, WAVE[i]) for i in range(12)]
    r1 = _compute(src, tgt)
    r2 = _compute(list(reversed(src)), list(reversed(tgt)))
    assert [ (l.lag_seconds, l.correlation, l.pair_count) for l in r1.lags] == [
        (l.lag_seconds, l.correlation, l.pair_count) for l in r2.lags
    ]
    assert r1.best_lag_seconds == r2.best_lag_seconds


# ---------- 时间离散边界与资源上限（纯算法） ----------


def _span_samples(span_seconds: float, base_t: float = 20.0):
    """只在窗口两端各放一个读数的序列（配合大 max_gap 覆盖整段）。"""
    return [
        Sample(BASE, base_t, 50.0),
        Sample(BASE + dt.timedelta(seconds=span_seconds), base_t + 1.0, 50.0),
    ]


def test_sub_microsecond_step_rounding_to_zero_rejected():
    src = _wave_src()
    # 小于半微秒的正步长取整为 0：拒绝（修复前会网格除零）
    with pytest.raises(PropagationParamsError) as ei:
        _compute(src, src, step=0.0000001)
    assert ei.value.parameter == "step_seconds"
    # 恰好半微秒：round-half-even 取整为 0 → 同样拒绝
    with pytest.raises(PropagationParamsError):
        _compute(src, src, step=0.0000005)
    # 非正步长直调纯算法同样拒绝（接口层由 pydantic 拦）
    with pytest.raises(PropagationParamsError):
        _compute(src, src, step=0.0)
    with pytest.raises(PropagationParamsError):
        _compute(src, src, step=-1.0)


def test_non_finite_time_params_rejected():
    src = _wave_src()
    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(PropagationParamsError):
            _compute(src, src, step=bad)
        with pytest.raises(PropagationParamsError):
            _compute(src, src, max_lag=bad)
        with pytest.raises(PropagationParamsError):
            _compute(src, src, max_gap=bad)


def test_step_quantized_to_microseconds_and_echoed():
    # 60.0000004 秒取整为整 60 秒：回显、网格、时滞都按生效步长
    res = _compute(_wave_src(), _wave_src(), step=60.0000004, max_lag=120.0)
    assert res.step_seconds == 60.0
    assert res.grid_point_count == 12  # 重叠 [0, 11min]，与 step=60 完全一致
    assert [l.lag_seconds for l in res.lags] == [-120.0, -60.0, 0.0, 60.0, 120.0]

    # 1.5 微秒取整为 2 微秒（4µs 跨度 → 0/2/4µs 三点）
    tiny = [
        Sample(BASE, 20.0, 50.0),
        Sample(BASE + dt.timedelta(microseconds=4), 24.0, 50.0),
    ]
    tiny_res = compute_propagation(
        tiny, tiny, "temperature",
        start=BASE, end=BASE + dt.timedelta(microseconds=4),
        step_seconds=0.0000015, max_lag_seconds=0.0, max_gap_seconds=600.0,
    )
    assert tiny_res.step_seconds == pytest.approx(0.000002)
    assert tiny_res.grid_point_count == 3
    assert tiny_res.lags[0].lag_seconds == 0.0


def test_lag_seconds_are_exact_multiples_of_effective_step():
    # 取整前带微秒零头；时滞秒数不得带出浮点漂移
    res = _compute(_wave_src(), _wave_src(), step=60.0000004, max_lag=300.0)
    assert [l.lag_seconds for l in res.lags] == [
        -300.0, -240.0, -180.0, -120.0, -60.0,
        0.0, 60.0, 120.0, 180.0, 240.0, 300.0,
    ]


def test_grid_point_limit_boundary_values():
    # 恰好 MAX_GRID_POINTS 个网格点：可计算
    res = _compute(
        _span_samples(9999.0), _span_samples(9999.0),
        end=BASE + dt.timedelta(seconds=9999.0),
        step=1.0, max_lag=0.0, max_gap=20000.0,
    )
    assert res.grid_point_count == MAX_GRID_POINTS == 10000

    # 再多一个点：分配前拒绝
    with pytest.raises(PropagationParamsError) as ei:
        _compute(
            _span_samples(10000.0), _span_samples(10000.0),
            end=BASE + dt.timedelta(seconds=10000.0),
            step=1.0, max_lag=0.0, max_gap=20000.0,
        )
    assert ei.value.parameter == "grid_point_count"
    assert (ei.value.limit, ei.value.actual) == (10000, 10001)


def test_lag_row_limit_boundary_values():
    end = BASE + dt.timedelta(seconds=660)
    # 2*100+1 = 201 行：可计算
    res = _compute(_wave_src(), _wave_src(), end=end, step=1.0, max_lag=100.0)
    assert len(res.lags) == MAX_LAG_ROWS == 201

    # 203 行：拒绝
    with pytest.raises(PropagationParamsError) as ei:
        _compute(_wave_src(), _wave_src(), end=end, step=1.0, max_lag=101.0)
    assert ei.value.parameter == "lag_row_count"
    assert (ei.value.limit, ei.value.actual) == (201, 203)


def test_lag_row_limit_applies_even_without_overlap():
    # 无重叠时仍会枚举全部时滞明细：大时滞同样受 201 行上限保护
    far = [_sample(60, 1.0), _sample(61, 2.0)]
    with pytest.raises(PropagationParamsError) as ei:
        _compute(_wave_src(), far, step=1.0, max_lag=101.0)
    assert ei.value.parameter == "lag_row_count"


def test_pair_work_limit_boundary_values():
    # 8000 网格点 × 125 时滞行 = 1,000,000：恰好等于上限，可计算
    res = _compute(
        _span_samples(7999.0), _span_samples(7999.0),
        end=BASE + dt.timedelta(seconds=7999.0),
        step=1.0, max_lag=62.0, max_gap=20000.0,
    )
    assert res.grid_point_count == 8000
    assert len(res.lags) == 125
    assert 8000 * 125 == MAX_PAIR_WORK == 1_000_000

    # 8001 × 125 = 1,000,125 > 上限：分配前拒绝（网格点数本身未超 10000）
    with pytest.raises(PropagationParamsError) as ei:
        _compute(
            _span_samples(8000.0), _span_samples(8000.0),
            end=BASE + dt.timedelta(seconds=8000.0),
            step=1.0, max_lag=62.0, max_gap=20000.0,
        )
    assert ei.value.parameter == "pair_work"
    assert (ei.value.limit, ei.value.actual) == (1_000_000, 1_000_125)


def test_limits_enforced_before_allocation_is_fast():
    # 大窗口 × 小步长本会生成上亿网格点：必须在校验阶段立即失败
    import time

    t0 = time.perf_counter()
    with pytest.raises(PropagationParamsError):
        _compute(
            _span_samples(100_000.0), _span_samples(100_000.0),
            end=BASE + dt.timedelta(seconds=100_000.0),
            step=0.001, max_lag=0.0, max_gap=200_000.0,
        )
    assert time.perf_counter() - t0 < 1.0


# ---------- 接口集成 ----------


def _ingest(client, sn, rows):
    r = client.post("/api/v1/readings/batch", json={"readings": rows})
    assert r.status_code == 200, r.text


def _wave_readings(sn, offset_minutes=0):
    return [
        _reading(
            sn,
            (BASE + dt.timedelta(seconds=60 * (i + offset_minutes))).isoformat().replace(
                "+00:00", "Z"
            ),
            WAVE[i],
            50.0,
        )
        for i in range(12)
    ]


BASE_PARAMS = {
    "start": "2026-09-01T00:00:00Z",
    "end": "2026-09-01T00:13:00Z",
    "step_seconds": 60,
    "max_lag_seconds": 300,
    "max_gap_seconds": 600,
}


def test_endpoint_detects_positive_lag_and_direction(client):
    _ingest(client, "CASE-S", _wave_readings("CASE-S", 0))
    _ingest(client, "CASE-T", _wave_readings("CASE-T", 2))
    r = client.get(
        "/api/v1/propagation",
        params={**BASE_PARAMS, "source_sn": "CASE-S", "target_sn": "CASE-T",
                "dimension": "temperature"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source_sn"] == "CASE-S"
    assert body["target_sn"] == "CASE-T"
    assert body["dimension"] == "temperature"
    assert body["best_lag_seconds"] == 120.0
    assert body["best_correlation"] == pytest.approx(1.0, abs=1e-9)
    assert body["direction"] == "source_to_target"
    assert body["grid_point_count"] == 10
    lags = body["lags"]
    assert [l["lag_seconds"] for l in lags] == sorted(l["lag_seconds"] for l in lags)
    best = next(l for l in lags if l["lag_seconds"] == 120.0)
    assert best["available"] is True
    # 不存在 NaN（JSON 中 NaN 不合法；响应可正常解析即满足）
    assert all(l["correlation"] is None or isinstance(l["correlation"], (int, float)) for l in lags)


def test_endpoint_negative_lag(client):
    _ingest(client, "CASE-S2", _wave_readings("CASE-S2", 2))
    _ingest(client, "CASE-T2", _wave_readings("CASE-T2", 0))
    r = client.get(
        "/api/v1/propagation",
        params={**BASE_PARAMS, "source_sn": "CASE-S2", "target_sn": "CASE-T2",
                "dimension": "temperature"},
    )
    body = r.json()
    assert body["best_lag_seconds"] == -120.0
    assert body["direction"] == "target_to_source"


def test_endpoint_no_overlap_is_unavailable_200(client):
    _ingest(client, "CASE-A", [
        _reading("CASE-A", "2026-09-01T00:00:00Z", 20.0, 50.0),
        _reading("CASE-A", "2026-09-01T00:10:00Z", 24.0, 50.0),
    ])
    _ingest(client, "CASE-B", [
        _reading("CASE-B", "2026-09-02T00:00:00Z", 20.0, 50.0),
        _reading("CASE-B", "2026-09-02T00:10:00Z", 24.0, 50.0),
    ])
    r = client.get(
        "/api/v1/propagation",
        params={
            "source_sn": "CASE-A", "target_sn": "CASE-B", "dimension": "temperature",
            "start": "2026-09-01T00:00:00Z", "end": "2026-09-03T00:00:00Z",
            "step_seconds": 60, "max_lag_seconds": 300, "max_gap_seconds": 600,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["direction"] == "unavailable"
    assert body["best_lag_seconds"] is None
    assert body["best_correlation"] is None
    assert body["overlap_start"] is None
    assert body["overlap_end"] is None
    assert body["grid_point_count"] == 0
    assert all(l["available"] is False and l["correlation"] is None for l in body["lags"])


def test_endpoint_unknown_device_404(client):
    _ingest(client, "CASE-KNOWN", _wave_readings("CASE-KNOWN", 0))
    r = client.get(
        "/api/v1/propagation",
        params={**BASE_PARAMS, "source_sn": "CASE-KNOWN", "target_sn": "CASE-GHOST",
                "dimension": "temperature"},
    )
    assert r.status_code == 404
    r = client.get(
        "/api/v1/propagation",
        params={**BASE_PARAMS, "source_sn": "CASE-GHOST", "target_sn": "CASE-KNOWN",
                "dimension": "temperature"},
    )
    assert r.status_code == 404


def test_endpoint_validation_errors(client):
    _ingest(client, "CASE-V1", _wave_readings("CASE-V1", 0))
    _ingest(client, "CASE-V2", _wave_readings("CASE-V2", 0))
    base = {
        "source_sn": "CASE-V1", "target_sn": "CASE-V2", "dimension": "temperature",
        **BASE_PARAMS,
    }

    def status(**over):
        p = {**base, **over}
        return client.get("/api/v1/propagation", params=p).status_code

    # source 与 target 相同
    assert status(source_sn="CASE-V1", target_sn="CASE-V1") == 422
    # start 晚于 end
    assert status(start="2026-09-02T00:00:00Z", end="2026-09-01T00:00:00Z") == 422
    # step 非正
    assert status(step_seconds=0) == 422
    assert status(step_seconds=-10) == 422
    # max_lag 负
    assert status(max_lag_seconds=-1) == 422
    # max_gap 非正
    assert status(max_gap_seconds=0) == 422
    # 非法维度
    assert status(dimension="pressure") == 422
    # 缺少必填参数
    assert client.get("/api/v1/propagation", params={"source_sn": "CASE-V1"}).status_code == 422


def test_endpoint_max_gap_controls_invalidation(client):
    # target 在中间断档 20 分钟；max_gap=600 时断档内点无效
    rows = [
        _reading("CASE-G1", "2026-09-01T00:%02d:00Z" % i, WAVE[i], 50.0)
        for i in range(12)
    ]
    _ingest(client, "CASE-G1", rows)
    target_rows = [
        _reading("CASE-G2", "2026-09-01T00:00:00Z", 20.0, 50.0),
        _reading("CASE-G2", "2026-09-01T00:01:00Z", 22.0, 50.0),
        _reading("CASE-G2", "2026-09-01T00:21:00Z", 21.0, 50.0),
        _reading("CASE-G2", "2026-09-01T00:22:00Z", 24.0, 50.0),
    ]
    _ingest(client, "CASE-G2", target_rows)
    params = {
        "source_sn": "CASE-G1", "target_sn": "CASE-G2", "dimension": "temperature",
        "start": "2026-09-01T00:00:00Z", "end": "2026-09-01T00:11:00Z",
        "step_seconds": 60, "max_lag_seconds": 0, "max_gap_seconds": 600,
    }
    body = client.get("/api/v1/propagation", params=params).json()
    # target 有效点：0,1 分钟（11 分钟超出 target 覆盖前的断档，落在 1→21 间隔内）
    assert body["target_valid_count"] == 2

    # 放宽 max_gap 到 2000 秒：断档内点恢复有效
    params["max_gap_seconds"] = 2000
    body2 = client.get("/api/v1/propagation", params=params).json()
    assert body2["target_valid_count"] == body2["grid_point_count"]


def test_endpoint_deterministic(client):
    _ingest(client, "CASE-D1", _wave_readings("CASE-D1", 0))
    _ingest(client, "CASE-D2", _wave_readings("CASE-D2", 2))
    params = {**BASE_PARAMS, "source_sn": "CASE-D1", "target_sn": "CASE-D2",
              "dimension": "temperature"}
    first = client.get("/api/v1/propagation", params=params).json()
    second = client.get("/api/v1/propagation", params=params).json()
    assert first == second


# ---------- 接口集成：时间离散边界与资源上限 ----------


def test_endpoint_rejects_sub_microsecond_step(client):
    _ingest(client, "CASE-M1", _wave_readings("CASE-M1", 0))
    _ingest(client, "CASE-M2", _wave_readings("CASE-M2", 0))
    params = {
        "source_sn": "CASE-M1", "target_sn": "CASE-M2", "dimension": "temperature",
        **BASE_PARAMS, "step_seconds": 0.0000001,
    }
    r = client.get("/api/v1/propagation", params=params)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["parameter"] == "step_seconds"
    assert "微秒" in detail["message"]


def test_endpoint_time_param_validation_precedes_404(client):
    # 未知展柜本应 404，但时间参数非法时先判 422
    params = {
        "source_sn": "CASE-GHOST1", "target_sn": "CASE-GHOST2",
        "dimension": "temperature", **BASE_PARAMS, "step_seconds": 0.0000001,
    }
    assert client.get("/api/v1/propagation", params=params).status_code == 422


def test_endpoint_rejects_non_finite_time_params(client):
    _ingest(client, "CASE-N1", _wave_readings("CASE-N1", 0))
    _ingest(client, "CASE-N2", _wave_readings("CASE-N2", 0))
    base = {
        "source_sn": "CASE-N1", "target_sn": "CASE-N2",
        "dimension": "temperature", **BASE_PARAMS,
    }
    for key in ("step_seconds", "max_lag_seconds", "max_gap_seconds"):
        for bad in ("inf", "-inf", "nan"):
            r = client.get("/api/v1/propagation", params={**base, key: bad})
            assert r.status_code == 422, (key, bad, r.text)
    # 放行 pydantic 的 +inf 步长由接口层拒绝，且 detail 指明参数
    r = client.get("/api/v1/propagation", params={**base, "step_seconds": "inf"})
    assert r.status_code == 422
    assert r.json()["detail"]["parameter"] == "step_seconds"


def test_endpoint_effective_step_echo_and_grid(client):
    _ingest(client, "CASE-E1", _wave_readings("CASE-E1", 0))
    _ingest(client, "CASE-E2", _wave_readings("CASE-E2", 2))
    r = client.get(
        "/api/v1/propagation",
        params={
            **BASE_PARAMS, "source_sn": "CASE-E1", "target_sn": "CASE-E2",
            "dimension": "temperature", "step_seconds": 60.0000004,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # 回显与各档时滞都使用取整后的生效步长 60s
    assert body["step_seconds"] == 60.0
    assert body["grid_point_count"] == 10
    assert body["best_lag_seconds"] == 120.0
    assert [l["lag_seconds"] for l in body["lags"]] == [
        -300.0, -240.0, -180.0, -120.0, -60.0,
        0.0, 60.0, 120.0, 180.0, 240.0, 300.0,
    ]


def test_endpoint_grid_point_limit(client):
    # 两端读数 + 大 max_gap 覆盖整段，无需上报上万条读数
    _ingest(client, "CASE-W1", [
        _reading("CASE-W1", "2026-09-01T00:00:00Z", 20.0, 50.0),
        _reading("CASE-W1", "2026-09-01T02:46:40Z", 24.0, 50.0),  # +10000s
    ])
    _ingest(client, "CASE-W2", [
        _reading("CASE-W2", "2026-09-01T00:00:00Z", 21.0, 50.0),
        _reading("CASE-W2", "2026-09-01T02:46:40Z", 25.0, 50.0),
    ])
    base = {
        "source_sn": "CASE-W1", "target_sn": "CASE-W2", "dimension": "temperature",
        "start": "2026-09-01T00:00:00Z", "end": "2026-09-01T02:46:40Z",
        "step_seconds": 1, "max_lag_seconds": 0, "max_gap_seconds": 20000,
    }
    # 10001 点 > 10000：422 且 detail 带 limit/actual
    r = client.get("/api/v1/propagation", params=base)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail == {
        "message": "重叠时段网格点数 10001 超过上限 10000",
        "parameter": "grid_point_count", "limit": 10000, "actual": 10001,
    }
    # 缩短 1 秒：恰好 10000 点，可计算
    r = client.get(
        "/api/v1/propagation",
        params={**base, "end": "2026-09-01T02:46:39Z"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["grid_point_count"] == 10000


def test_endpoint_lag_row_limit(client):
    _ingest(client, "CASE-L1", _wave_readings("CASE-L1", 0))
    _ingest(client, "CASE-L2", _wave_readings("CASE-L2", 0))
    base = {
        "source_sn": "CASE-L1", "target_sn": "CASE-L2", "dimension": "temperature",
        "start": "2026-09-01T00:00:00Z", "end": "2026-09-01T00:11:00Z",
        "step_seconds": 1, "max_gap_seconds": 600,
    }
    # 2*100+1 = 201 行：可计算
    r = client.get("/api/v1/propagation", params={**base, "max_lag_seconds": 100})
    assert r.status_code == 200, r.text
    assert len(r.json()["lags"]) == 201
    # 203 行：422
    r = client.get("/api/v1/propagation", params={**base, "max_lag_seconds": 101})
    assert r.status_code == 422
    assert r.json()["detail"] == {
        "message": "枚举时滞行数 203 超过上限 201",
        "parameter": "lag_row_count", "limit": 201, "actual": 203,
    }


def test_endpoint_pair_work_limit(client):
    _ingest(client, "CASE-P1", [
        _reading("CASE-P1", "2026-09-01T00:00:00Z", 20.0, 50.0),
        _reading("CASE-P1", "2026-09-01T02:13:20Z", 24.0, 50.0),  # +8000s
    ])
    _ingest(client, "CASE-P2", [
        _reading("CASE-P2", "2026-09-01T00:00:00Z", 21.0, 50.0),
        _reading("CASE-P2", "2026-09-01T02:13:20Z", 25.0, 50.0),
    ])
    base = {
        "source_sn": "CASE-P1", "target_sn": "CASE-P2", "dimension": "temperature",
        "start": "2026-09-01T00:00:00Z", "end": "2026-09-01T02:13:20Z",
        "step_seconds": 1, "max_lag_seconds": 62, "max_gap_seconds": 20000,
    }
    # 8001 点 × 125 行 = 1,000,125 > 1,000,000：422
    r = client.get("/api/v1/propagation", params=base)
    assert r.status_code == 422
    assert r.json()["detail"] == {
        "message": "网格点数 × 时滞行数的配对工作量 1000125 超过上限 1000000",
        "parameter": "pair_work", "limit": 1000000, "actual": 1000125,
    }
    # 缩短 1 秒：8000 × 125 = 恰好 1,000,000，可计算
    r = client.get(
        "/api/v1/propagation",
        params={**base, "end": "2026-09-01T02:13:19Z"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grid_point_count"] == 8000
    assert len(body["lags"]) == 125
    # 原有语义不受影响：两柜都是线性斜坡，零时滞相关为 1
    lag0 = next(l for l in body["lags"] if l["lag_seconds"] == 0.0)
    assert lag0["correlation"] == pytest.approx(1.0, abs=1e-9)
