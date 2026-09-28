"""累计越界剂量预算：窗末累计量、剩余额度与预算耗尽时刻反解。

纯算法测试覆盖：线性插值规则切出的越界片段面积、在线段内按越界
幅度线性变化反解耗尽时刻、零预算在首次越界起点耗尽、断档不跨区间
累计；接口集成测试覆盖 422 校验组合、404 与空范围。
"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine, init_db
from app.excursions import (
    HUMIDITY,
    TEMPERATURE,
    Thresholds,
    compute_exposure_budget,
)
from app.main import app
from app.segmentation import Sample

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def _sample(minutes: float, t: float, h: float = 50.0) -> Sample:
    return Sample(BASE + dt.timedelta(minutes=minutes), t, h)


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def _budget(samples, *, thresholds, budgets, max_gap=600.0):
    out = compute_exposure_budget(samples, thresholds, budgets, max_gap)
    return {d.dimension: d for d in out.dimensions}


# ---------- 纯算法：窗末累计量与剩余额度 ----------


def test_cumulative_dose_and_remaining():
    # 20→30→20°C，上限 25：两个三角形，合计 1500 °C·s
    res = _budget(
        [_sample(0, 20.0), _sample(10, 30.0), _sample(20, 20.0)],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 2000.0},
    )
    t = res[TEMPERATURE]
    assert t.cumulative_dose == pytest.approx(1500.0)
    assert t.remaining == pytest.approx(500.0)
    assert t.exhausted_ts is None


def test_remaining_negative_when_overspent():
    res = _budget(
        [_sample(0, 28.0), _sample(10, 28.0)],  # 矩形 3°C × 600s = 1800
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 1000.0},
    )
    t = res[TEMPERATURE]
    assert t.cumulative_dose == pytest.approx(1800.0)
    assert t.remaining == pytest.approx(-800.0)
    assert t.exhausted_ts is not None


def test_no_violation_keeps_full_budget():
    res = _budget(
        [_sample(0, 20.0), _sample(10, 24.0)],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 500.0},
    )
    t = res[TEMPERATURE]
    assert t.cumulative_dose == 0.0
    assert t.remaining == pytest.approx(500.0)
    assert t.exhausted_ts is None


# ---------- 耗尽时刻反解 ----------


def test_exhaustion_inside_linear_ramp_triangle():
    # 20→30°C，上限 25：越界自第 5 分钟起，幅度线性 0→5，
    # 片段 [5min,10min] 内 e(u)=5·u/300（以秒计），A=0.5·k·x²。
    # 预算 600 = 0.5 × (5/300) × x² → x = sqrt(72000) ≈ 268.328s
    res = _budget(
        [_sample(0, 20.0), _sample(10, 30.0)],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 600.0},
    )
    t = res[TEMPERATURE]
    x = (72000.0) ** 0.5
    expected = BASE + dt.timedelta(minutes=5, seconds=x)
    assert t.exhausted_ts == pytest.approx(expected, abs=dt.timedelta(microseconds=1))
    # 耗尽时刻处累计恰为预算，窗末累计为整片面积 750
    assert t.cumulative_dose == pytest.approx(750.0)


def test_exhaustion_inside_trapezoid():
    # 28→34°C，上限 25：全片越界，幅度 3→9 线性，时长 600s。
    # A(x) = 3·x + 0.5·(6/600)·x²，预算 1800：
    # x² + 600·x − 360000 = 0 → x = -300 + sqrt(450000) ≈ 370.82s
    res = _budget(
        [_sample(0, 28.0), _sample(10, 34.0)],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 1800.0},
    )
    t = res[TEMPERATURE]
    x = -300.0 + 450000.0 ** 0.5
    expected = BASE + dt.timedelta(seconds=x)
    assert t.exhausted_ts == pytest.approx(expected, abs=dt.timedelta(microseconds=1))
    # 整片面积 0.5×(3+9)×600 = 3600
    assert t.cumulative_dose == pytest.approx(3600.0)


def test_exhaustion_at_piece_end_is_exact_crossing():
    # 第一段三角面积恰好 750：预算 750 时耗尽时刻应精确等于段末读数
    res = _budget(
        [_sample(0, 20.0), _sample(10, 30.0)],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 750.0},
    )
    assert res[TEMPERATURE].exhausted_ts == BASE + dt.timedelta(minutes=10)


def test_exhaustion_in_second_piece():
    # 两片三角各 750；预算 1000 → 在第二片 [10min,15min]（幅度 5→0
    # 的下降沿）内反解，剩余 250：A(x) = 5·x − x²/120 = 250
    # → x = 300 − sqrt(60000)
    res = _budget(
        [_sample(0, 20.0), _sample(10, 30.0), _sample(20, 20.0)],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 1000.0},
    )
    x = 300.0 - 60000.0 ** 0.5
    expected = BASE + dt.timedelta(minutes=10, seconds=x)
    assert res[TEMPERATURE].exhausted_ts == pytest.approx(
        expected, abs=dt.timedelta(microseconds=1)
    )


def test_budget_not_exhausted_null_ts():
    res = _budget(
        [_sample(0, 20.0), _sample(10, 30.0)],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 10000.0},
    )
    assert res[TEMPERATURE].exhausted_ts is None


# ---------- 零预算 ----------


def test_zero_budget_exhausts_at_first_violation_start():
    res = _budget(
        [_sample(0, 20.0), _sample(10, 30.0)],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 0.0},
    )
    t = res[TEMPERATURE]
    # 首次越界起点即线性插值交点（第 5 分钟），而非读数时刻
    assert t.exhausted_ts == BASE + dt.timedelta(minutes=5)
    assert t.remaining == pytest.approx(-750.0)


def test_zero_budget_no_violation_not_exhausted():
    res = _budget(
        [_sample(0, 20.0), _sample(10, 24.0)],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 0.0},
    )
    assert res[TEMPERATURE].exhausted_ts is None
    assert res[TEMPERATURE].remaining == 0.0


def test_zero_budget_after_gap_starts_at_next_piece():
    # 0→10 分钟不越界；10→300 分钟断档；300→310 分钟 20→30 越界。
    # 零预算应在断档之后的首个越界起点（305min）耗尽
    res = _budget(
        [
            _sample(0, 20.0),
            _sample(10, 20.0),
            _sample(300, 20.0),
            _sample(310, 30.0),
        ],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 0.0},
        max_gap=600.0,
    )
    assert res[TEMPERATURE].exhausted_ts == BASE + dt.timedelta(minutes=305)


# ---------- 断档 ----------


def test_gap_not_accumulated_across_break():
    # 0→10 分钟一片三角 750；10→300 断档；300→310 矩形 5×600=3000。
    # 预算 1000 在断档之后才耗尽，且累计不含任何断档面积
    res = _budget(
        [
            _sample(0, 20.0),
            _sample(10, 30.0),
            _sample(300, 30.0),
            _sample(310, 30.0),
        ],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 1000.0},
        max_gap=600.0,
    )
    t = res[TEMPERATURE]
    assert t.cumulative_dose == pytest.approx(3750.0)
    # 第一段仅累计 750 < 1000；第二段矩形内还需 250 → 250/5 = 50s
    assert t.exhausted_ts == BASE + dt.timedelta(minutes=300, seconds=50)


def test_gap_equal_max_gap_still_accumulates():
    # 相邻间隔恰好 600s 不断开：300min→310min 与前段连续
    res = _budget(
        [
            _sample(0, 20.0),
            _sample(10, 30.0),
            _sample(300, 30.0),
            _sample(310, 30.0),
        ],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 50000.0},
        max_gap=17400.0,
    )
    # 断档被视为连续：0..310min 全程按线性曲线累计
    assert res[TEMPERATURE].cumulative_dose > 0.0
    assert res[TEMPERATURE].exhausted_ts is not None


def test_isolated_spike_contributes_nothing():
    res = _budget(
        [
            _sample(0, 20.0),
            _sample(10, 20.0),
            _sample(300, 40.0),  # 两侧均断档
            _sample(600, 20.0),
        ],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 100.0},
    )
    t = res[TEMPERATURE]
    assert t.cumulative_dose == 0.0
    assert t.exhausted_ts is None
    assert t.remaining == pytest.approx(100.0)


# ---------- 维度与上下限 ----------


def test_both_dimensions_independent():
    res = _budget(
        [_sample(0, 30.0, 50.0), _sample(10, 20.0, 30.0)],
        thresholds=Thresholds(temp_max=25.0, humidity_min=40.0),
        budgets={TEMPERATURE: 750.0, HUMIDITY: 0.0},
    )
    assert set(res) == {TEMPERATURE, HUMIDITY}
    # 温度越界 [0,5min] 三角面积 750 → 预算恰好耗尽于阈值交点（第 5 分钟）
    assert res[TEMPERATURE].exhausted_ts == BASE + dt.timedelta(minutes=5)
    assert res[TEMPERATURE].remaining == pytest.approx(0.0)
    # 湿度零预算 → 首次越界起点（中点）
    assert res[HUMIDITY].exhausted_ts == BASE + dt.timedelta(minutes=5)
    # 返回顺序 temperature 在前
    assert list(res) == [TEMPERATURE, HUMIDITY]


def test_lower_branch_inverse_solution():
    # 20→10°C，下限 15：中点跌破，幅度 0→5，预算 375 为半片三角
    res = _budget(
        [_sample(0, 20.0), _sample(10, 10.0)],
        thresholds=Thresholds(temp_min=15.0),
        budgets={TEMPERATURE: 375.0},
    )
    # 375 = 0.5 × (5/300) × x² → x = sqrt(45000)
    x = 45000.0 ** 0.5
    expected = BASE + dt.timedelta(minutes=5, seconds=x)
    assert res[TEMPERATURE].exhausted_ts == pytest.approx(
        expected, abs=dt.timedelta(microseconds=1)
    )


def test_unsorted_input_deterministic_budget():
    samples = [
        _sample(0, 20.0),
        _sample(10, 30.0),
        _sample(20, 20.0),
    ]
    a = _budget(
        samples, thresholds=Thresholds(temp_max=25.0), budgets={TEMPERATURE: 800.0}
    )
    b = _budget(
        samples[::-1],
        thresholds=Thresholds(temp_max=25.0),
        budgets={TEMPERATURE: 800.0},
    )
    assert a[TEMPERATURE] == b[TEMPERATURE]


# ---------- 接口集成 ----------


def _ingest(client, sn, rows):
    r = client.post("/api/v1/readings/batch", json={"readings": rows})
    assert r.status_code == 200, r.text


def test_exposure_budget_endpoint_basic(client):
    _ingest(client, "CASE-B1", [
        _reading("CASE-B1", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-B1", "2026-09-01T08:10:00+08:00", 30.0, 50.0),
        _reading("CASE-B1", "2026-09-01T08:20:00+08:00", 20.0, 50.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-B1/exposure-budget",
        params={"temp_max": 25, "temp_budget": 1000},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "CASE-B1"
    assert body["temp_max"] == 25.0
    assert body["temp_min"] is None
    assert body["temp_budget"] == 1000.0
    assert body["humidity_budget"] is None
    assert body["max_gap_seconds"] == 600.0
    assert body["sample_count"] == 3
    assert len(body["dimensions"]) == 1

    dim = body["dimensions"][0]
    assert dim["dimension"] == "temperature"
    assert dim["budget"] == 1000.0
    assert dim["cumulative_dose"] == pytest.approx(1500.0)
    assert dim["remaining"] == pytest.approx(-500.0)
    assert dim["budget_exhausted"] is True
    # 预算 1000 > 第一片 750 → 第二片（幅度 5→0 的下降沿）内反解：
    # 5·x − x²/120 = 250 → x = 300 − sqrt(60000)
    x = 300.0 - 60000.0 ** 0.5
    expected = _ts("2026-09-01T08:10:00+08:00") + dt.timedelta(seconds=x)
    assert _ts(dim["exhausted_ts"]) == pytest.approx(
        expected.astimezone(dt.timezone.utc), abs=dt.timedelta(microseconds=1)
    )


def test_exposure_budget_zero_budget_via_api(client):
    _ingest(client, "CASE-B2", [
        _reading("CASE-B2", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-B2", "2026-09-01T08:10:00+08:00", 30.0, 50.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-B2/exposure-budget",
        params={"temp_max": 25, "temp_budget": 0},
    )
    assert r.status_code == 200, r.text
    dim = r.json()["dimensions"][0]
    assert dim["budget_exhausted"] is True
    # 首次越界起点：08:05（+08:00）→ 00:05 UTC
    assert _ts(dim["exhausted_ts"]) == _ts("2026-09-01T00:05:00+00:00")


def test_exposure_budget_not_exhausted_null_ts(client):
    _ingest(client, "CASE-B3", [
        _reading("CASE-B3", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-B3", "2026-09-01T08:10:00+08:00", 24.0, 50.0),
    ])
    body = client.get(
        "/api/v1/cases/CASE-B3/exposure-budget",
        params={"temp_max": 25, "temp_budget": 100},
    ).json()
    dim = body["dimensions"][0]
    assert dim["cumulative_dose"] == 0.0
    assert dim["remaining"] == pytest.approx(100.0)
    assert dim["budget_exhausted"] is False
    assert dim["exhausted_ts"] is None


def test_exposure_budget_gap_not_accumulated(client):
    _ingest(client, "CASE-B4", [
        _reading("CASE-B4", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-B4", "2026-09-01T08:10:00+08:00", 30.0, 50.0),
        _reading("CASE-B4", "2026-09-01T13:00:00+08:00", 30.0, 50.0),
        _reading("CASE-B4", "2026-09-01T13:10:00+08:00", 30.0, 50.0),
    ])
    body = client.get(
        "/api/v1/cases/CASE-B4/exposure-budget",
        params={"temp_max": 25, "temp_budget": 1000},
    ).json()
    dims = {d["dimension"]: d for d in body["dimensions"]}
    # 750（三角）+ 3000（矩形），断档区间无面积
    assert dims["temperature"]["cumulative_dose"] == pytest.approx(3750.0)
    # 耗尽发生在 13:00 之后 50s（矩形 5°C/s → 250/5）
    assert _ts(dims["temperature"]["exhausted_ts"]) == _ts(
        "2026-09-01T05:00:50+00:00"
    )


def test_exposure_budget_both_dimensions_via_api(client):
    _ingest(client, "CASE-B5", [
        _reading("CASE-B5", "2026-09-01T08:00:00+08:00", 30.0, 70.0),
        _reading("CASE-B5", "2026-09-01T08:10:00+08:00", 20.0, 50.0),
    ])
    body = client.get(
        "/api/v1/cases/CASE-B5/exposure-budget",
        params={
            "temp_max": 25,
            "humidity_max": 60,
            "temp_budget": 0,
            "humidity_budget": 100000,
        },
    ).json()
    dims = {d["dimension"]: d for d in body["dimensions"]}
    assert [d["dimension"] for d in body["dimensions"]] == [
        "temperature",
        "humidity",
    ]
    # 温度 30→20（上限 25）自窗口起点即越界：零预算在起点耗尽
    assert _ts(dims["temperature"]["exhausted_ts"]) == _ts(
        "2026-09-01T00:00:00+00:00"
    )
    # 湿度越界 [0,5min] 三角面积 0.5×10×300 = 1500，预算远未耗尽
    assert dims["humidity"]["cumulative_dose"] == pytest.approx(1500.0)
    assert dims["humidity"]["exhausted_ts"] is None
    assert dims["humidity"]["remaining"] == pytest.approx(98500.0)


def test_exposure_budget_window_filter(client):
    _ingest(client, "CASE-B6", [
        _reading("CASE-B6", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-B6", "2026-09-01T08:10:00+08:00", 30.0, 50.0),
        _reading("CASE-B6", "2026-09-01T08:20:00+08:00", 20.0, 50.0),
    ])
    # 只取后两点：窗内曲线 30→20，单片三角 750
    body = client.get(
        "/api/v1/cases/CASE-B6/exposure-budget",
        params={
            "temp_max": 25,
            "temp_budget": 750,
            "start": "2026-09-01T08:10:00+08:00",
        },
    ).json()
    assert body["sample_count"] == 2
    dim = body["dimensions"][0]
    assert dim["cumulative_dose"] == pytest.approx(750.0)
    assert dim["remaining"] == pytest.approx(0.0)
    # 耗尽于窗内段末 08:15
    assert _ts(dim["exhausted_ts"]) == _ts("2026-09-01T00:15:00+00:00")


def test_exposure_budget_empty_range(client):
    _ingest(client, "CASE-B7", [
        _reading("CASE-B7", "2026-09-01T08:00:00+08:00", 30.0, 70.0),
        _reading("CASE-B7", "2026-09-01T08:10:00+08:00", 30.0, 70.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-B7/exposure-budget",
        params={
            "temp_max": 25,
            "temp_budget": 100,
            "start": "2030-01-01T00:00:00Z",
            "end": "2030-02-01T00:00:00Z",
        },
    )
    assert r.status_code == 200
    dim = r.json()["dimensions"][0]
    assert dim["cumulative_dose"] == 0.0
    assert dim["remaining"] == pytest.approx(100.0)
    assert dim["exhausted_ts"] is None
    assert dim["budget_exhausted"] is False


def test_exposure_budget_unknown_case_404(client):
    r = client.get(
        "/api/v1/cases/NOPE/exposure-budget",
        params={"temp_max": 25, "temp_budget": 100},
    )
    assert r.status_code == 404


@pytest.mark.parametrize(
    "params",
    [
        # 未提供任何“阈值 + 预算”组合
        {},
        # 有阈值无预算
        {"temp_max": 25},
        {"humidity_min": 40},
        # 有预算无阈值
        {"temp_budget": 100},
        {"humidity_budget": 100},
        # 预算为负（Query 层 ge=0）
        {"temp_max": 25, "temp_budget": -1},
        {"humidity_max": 60, "humidity_budget": -0.01},
        # start 晚于 end
        {
            "temp_max": 25,
            "temp_budget": 100,
            "start": "2030-02-01T00:00:00Z",
            "end": "2030-01-01T00:00:00Z",
        },
        # 同维下限大于上限
        {"temp_min": 30, "temp_max": 20, "temp_budget": 100},
        {"humidity_min": 80, "humidity_max": 40, "humidity_budget": 100},
        # max_gap 非正
        {"temp_max": 25, "temp_budget": 100, "max_gap": 0},
        # 阈值超量程
        {"temp_max": 150, "temp_budget": 100},
        {"humidity_max": 120, "humidity_budget": 100},
    ],
)
def test_exposure_budget_validation_422(client, params):
    url = "/api/v1/cases/CASE-X/exposure-budget"
    r = client.get(url, params=params)
    assert r.status_code == 422, r.text


def test_exposure_budget_deterministic(client):
    rows = [
        _reading("CASE-B8", f"2026-09-01T08:{i:02d}:00+08:00", 20.0 + (i % 3) * 4, 50.0)
        for i in range(0, 60, 10)
    ]
    _ingest(client, "CASE-B8", rows)
    params = {
        "temp_max": 25,
        "humidity_min": 45,
        "temp_budget": 300,
        "humidity_budget": 0,
        "max_gap": 900,
    }
    first = client.get(
        "/api/v1/cases/CASE-B8/exposure-budget", params=params
    ).json()
    second = client.get(
        "/api/v1/cases/CASE-B8/exposure-budget", params=params
    ).json()
    assert first == second
