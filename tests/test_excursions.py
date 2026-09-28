"""越界剂量：插值交点/剂量积分单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine, init_db
from app.excursions import Thresholds, compute_excursions
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


# ---------- 纯算法：线性插值交点与剂量积分 ----------


def test_crossing_exact_interpolation():
    # 20°C → 30°C 线性上升，上限 25°C：恰在中点越界
    events = compute_excursions(
        [_sample(0, 20.0), _sample(10, 30.0)],
        Thresholds(temp_max=25.0),
        max_gap_seconds=600.0,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.start_ts == BASE + dt.timedelta(minutes=5)
    assert ev.end_ts == BASE + dt.timedelta(minutes=10)
    assert ev.duration_seconds == 300.0
    assert ev.dimensions == ("temperature",)
    # 三角形面积：0.5 × 5°C × 300s
    assert ev.temp_dose == pytest.approx(750.0)
    assert ev.humidity_dose == 0.0
    assert ev.peak_temperature == 30.0
    assert ev.peak_temperature_ts == BASE + dt.timedelta(minutes=10)
    assert ev.peak_humidity is None
    assert ev.peak_humidity_ts is None


def test_constant_excess_rectangle_dose():
    events = compute_excursions(
        [_sample(0, 28.0), _sample(10, 28.0)],
        Thresholds(temp_max=25.0),
        max_gap_seconds=600.0,
    )
    assert len(events) == 1
    assert events[0].duration_seconds == 600.0
    # 矩形面积：3°C × 600s
    assert events[0].temp_dose == pytest.approx(1800.0)
    assert events[0].peak_temperature == 28.0


def test_humidity_below_min():
    events = compute_excursions(
        [_sample(0, 20.0, 50.0), _sample(10, 20.0, 30.0)],
        Thresholds(humidity_min=40.0),
        max_gap_seconds=600.0,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.dimensions == ("humidity",)
    assert ev.start_ts == BASE + dt.timedelta(minutes=5)
    assert ev.humidity_dose == pytest.approx(0.5 * 10.0 * 300.0)
    assert ev.peak_humidity == 30.0
    assert ev.peak_temperature is None


def test_gap_breaks_curve_and_unknown_excluded():
    # 0→10 分钟连续（恰好等于 max_gap 不断开）；10→300 分钟超过 max_gap 断开；
    # 孤立的 300 分钟高读数两侧曲线未知，不产生事件
    events = compute_excursions(
        [_sample(0, 20.0), _sample(10, 30.0), _sample(300, 34.0)],
        Thresholds(temp_max=25.0),
        max_gap_seconds=600.0,
    )
    assert len(events) == 1
    assert events[0].start_ts == BASE + dt.timedelta(minutes=5)
    assert events[0].end_ts == BASE + dt.timedelta(minutes=10)
    assert events[0].temp_dose == pytest.approx(750.0)


def test_gap_equal_max_gap_stays_connected():
    events = compute_excursions(
        [_sample(0, 30.0), _sample(10, 30.0)],
        Thresholds(temp_max=25.0),
        max_gap_seconds=600.0,
    )
    assert len(events) == 1
    assert events[0].duration_seconds == 600.0
    # 间隔稍超 max_gap 则断开：两个孤立点，无事件
    none = compute_excursions(
        [_sample(0, 30.0), _sample(10, 30.0)],
        Thresholds(temp_max=25.0),
        max_gap_seconds=599.0,
    )
    assert none == []


def test_merge_across_dimensions():
    # 温度越界 [0, 25min]，湿度越界 [5min, 30min]：并集连续 → 单个事件
    samples = [
        _sample(0, 30.0, 50.0),
        _sample(10, 30.0, 70.0),
        _sample(20, 30.0, 70.0),
        _sample(30, 20.0, 70.0),
    ]
    events = compute_excursions(
        samples,
        Thresholds(temp_max=25.0, humidity_max=60.0),
        max_gap_seconds=600.0,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.dimensions == ("temperature", "humidity")
    assert ev.start_ts == BASE
    assert ev.end_ts == BASE + dt.timedelta(minutes=30)
    assert ev.duration_seconds == 1800.0
    # 温度：5×1200（[0,20min] 矩形）+ 0.5×5×300（[20,25min] 三角）
    assert ev.temp_dose == pytest.approx(6750.0)
    # 湿度：0.5×10×300（[5,10min] 三角）+ 10×1200（[10,30min] 矩形）
    assert ev.humidity_dose == pytest.approx(13500.0)
    # 峰值并列时取最早出现
    assert ev.peak_temperature == 30.0
    assert ev.peak_temperature_ts == BASE
    assert ev.peak_humidity == 70.0
    assert ev.peak_humidity_ts == BASE + dt.timedelta(minutes=10)


def test_separate_events_sorted_by_start():
    samples = [
        _sample(0, 30.0),
        _sample(10, 30.0),
        _sample(20, 20.0),
        _sample(30, 20.0),
        _sample(40, 30.0),
        _sample(50, 30.0),
    ]
    events = compute_excursions(samples, Thresholds(temp_max=25.0), 600.0)
    assert len(events) == 2
    assert events[0].start_ts < events[1].start_ts
    assert events[0].end_ts == BASE + dt.timedelta(minutes=15)
    assert events[1].start_ts == BASE + dt.timedelta(minutes=35)
    for ev in events:
        # 矩形 5×600 + 三角 0.5×5×300
        assert ev.temp_dose == pytest.approx(3750.0)


def test_touching_at_threshold_merges():
    # 中间读数恰好等于阈值：两侧区间在该点相接，合并为一个事件
    events = compute_excursions(
        [_sample(0, 30.0), _sample(10, 25.0), _sample(20, 30.0)],
        Thresholds(temp_max=25.0),
        max_gap_seconds=600.0,
    )
    assert len(events) == 1
    assert events[0].duration_seconds == 1200.0
    assert events[0].temp_dose == pytest.approx(2 * 0.5 * 5.0 * 600.0)


def test_both_min_max_same_dimension():
    # 上限 24、下限 18：先冲高再跌破，两个独立事件
    samples = [
        _sample(0, 20.0),
        _sample(10, 26.0),
        _sample(20, 20.0),
        _sample(30, 16.0),
        _sample(40, 20.0),
    ]
    events = compute_excursions(
        samples, Thresholds(temp_min=18.0, temp_max=24.0), 600.0
    )
    assert len(events) == 2
    high, low = events
    assert high.start_ts == BASE + dt.timedelta(seconds=400)
    assert high.end_ts == BASE + dt.timedelta(seconds=800)
    assert high.temp_dose == pytest.approx(400.0)
    assert high.peak_temperature == 26.0
    assert low.start_ts == BASE + dt.timedelta(seconds=1500)
    assert low.end_ts == BASE + dt.timedelta(seconds=2100)
    assert low.temp_dose == pytest.approx(600.0)
    assert low.peak_temperature == 16.0


def test_no_violation_no_events():
    assert compute_excursions(
        [_sample(0, 20.0), _sample(10, 24.0)],
        Thresholds(temp_max=25.0),
        600.0,
    ) == []
    # 单点无法构成区间
    assert compute_excursions([_sample(0, 40.0)], Thresholds(temp_max=25.0), 600.0) == []


def test_unsorted_input_deterministic():
    samples = [
        _sample(0, 30.0),
        _sample(10, 30.0),
        _sample(20, 20.0),
        _sample(30, 20.0),
        _sample(40, 30.0),
        _sample(50, 30.0),
    ]
    ordered = compute_excursions(samples, Thresholds(temp_max=25.0), 600.0)
    shuffled = compute_excursions(samples[::-1], Thresholds(temp_max=25.0), 600.0)
    assert ordered == shuffled


# ---------- 接口集成 ----------


def _ingest(client, sn, rows):
    r = client.post("/api/v1/readings/batch", json={"readings": rows})
    assert r.status_code == 200, r.text


def test_excursions_endpoint_basic(client):
    _ingest(client, "CASE-E1", [
        _reading("CASE-E1", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-E1", "2026-09-01T08:10:00+08:00", 30.0, 50.0),
        _reading("CASE-E1", "2026-09-01T08:20:00+08:00", 20.0, 50.0),
    ])
    r = client.get("/api/v1/cases/CASE-E1/excursions", params={"temp_max": 25})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "CASE-E1"
    assert body["temp_max"] == 25.0
    assert body["temp_min"] is None
    assert body["max_gap_seconds"] == 600.0
    assert body["sample_count"] == 3
    assert body["event_count"] == 1

    ev = body["events"][0]
    assert ev["dimensions"] == ["temperature"]
    assert _ts(ev["start_ts"]) == _ts("2026-09-01T00:05:00+00:00")
    assert _ts(ev["end_ts"]) == _ts("2026-09-01T00:15:00+00:00")
    assert ev["duration_seconds"] == pytest.approx(600.0)
    # 两个三角形：2 × 0.5 × 5°C × 300s
    assert ev["temp_dose"] == pytest.approx(1500.0)
    assert ev["humidity_dose"] == 0.0
    assert ev["peak_temperature"] == 30.0
    assert _ts(ev["peak_temperature_ts"]) == _ts("2026-09-01T00:10:00+00:00")
    assert ev["peak_humidity"] is None


def test_excursions_combined_event_via_api(client):
    _ingest(client, "CASE-E2", [
        _reading("CASE-E2", "2026-09-01T08:00:00+08:00", 30.0, 50.0),
        _reading("CASE-E2", "2026-09-01T08:10:00+08:00", 30.0, 70.0),
        _reading("CASE-E2", "2026-09-01T08:20:00+08:00", 30.0, 70.0),
        _reading("CASE-E2", "2026-09-01T08:30:00+08:00", 20.0, 70.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-E2/excursions",
        params={"temp_max": 25, "humidity_max": 60},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["event_count"] == 1
    ev = body["events"][0]
    assert ev["dimensions"] == ["temperature", "humidity"]
    assert ev["duration_seconds"] == pytest.approx(1800.0)
    assert ev["temp_dose"] == pytest.approx(6750.0)
    assert ev["humidity_dose"] == pytest.approx(13500.0)
    assert ev["peak_temperature"] == 30.0
    assert ev["peak_humidity"] == 70.0


def test_excursions_max_gap_param_controls_break(client):
    rows = [
        _reading("CASE-E3", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-E3", "2026-09-01T08:10:00+08:00", 30.0, 50.0),
        _reading("CASE-E3", "2026-09-01T10:00:00+08:00", 34.0, 50.0),
    ]
    _ingest(client, "CASE-E3", rows)

    # 默认 max_gap=600：08:10→10:00 断开，事件在 08:10 结束
    body = client.get(
        "/api/v1/cases/CASE-E3/excursions", params={"temp_max": 25}
    ).json()
    assert body["event_count"] == 1
    assert _ts(body["events"][0]["end_ts"]) == _ts("2026-09-01T00:10:00+00:00")
    assert body["events"][0]["temp_dose"] == pytest.approx(750.0)

    # max_gap=7200：曲线延续到 10:00，剂量计入整段梯形
    body = client.get(
        "/api/v1/cases/CASE-E3/excursions",
        params={"temp_max": 25, "max_gap": 7200},
    ).json()
    assert body["event_count"] == 1
    ev = body["events"][0]
    assert _ts(ev["end_ts"]) == _ts("2026-09-01T02:00:00+00:00")
    # 750 + 0.5×(5+9)×6600
    assert ev["temp_dose"] == pytest.approx(46950.0)
    assert ev["peak_temperature"] == 34.0


def test_excursions_isolated_spike_no_event(client):
    _ingest(client, "CASE-E4", [
        _reading("CASE-E4", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-E4", "2026-09-01T08:10:00+08:00", 20.0, 50.0),
        _reading("CASE-E4", "2026-09-01T12:00:00+08:00", 40.0, 50.0),
        _reading("CASE-E4", "2026-09-01T13:00:00+08:00", 20.0, 50.0),
        _reading("CASE-E4", "2026-09-01T13:10:00+08:00", 20.0, 50.0),
    ])
    body = client.get(
        "/api/v1/cases/CASE-E4/excursions", params={"temp_max": 25}
    ).json()
    # 12:00 的尖峰两侧间隔均超 max_gap，曲线未知 → 无事件
    assert body["event_count"] == 0
    assert body["events"] == []


def test_excursions_start_end_filter_scopes_samples(client):
    _ingest(client, "CASE-E5", [
        _reading("CASE-E5", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-E5", "2026-09-01T08:10:00+08:00", 30.0, 50.0),
        _reading("CASE-E5", "2026-09-01T08:20:00+08:00", 20.0, 50.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-E5/excursions",
        params={"temp_max": 25, "start": "2026-09-01T08:10:00+08:00"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_count"] == 2
    assert body["start"].startswith("2026-09-01T00:10:00")
    ev = body["events"][0]
    # 窗口内曲线仅从 08:10 开始：事件 [08:10, 08:15]
    assert _ts(ev["start_ts"]) == _ts("2026-09-01T00:10:00+00:00")
    assert ev["temp_dose"] == pytest.approx(750.0)


def test_excursions_empty_range_returns_empty(client):
    _ingest(client, "CASE-E6", [
        _reading("CASE-E6", "2026-09-01T08:00:00+08:00", 30.0, 70.0),
        _reading("CASE-E6", "2026-09-01T08:10:00+08:00", 30.0, 70.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-E6/excursions",
        params={
            "temp_max": 25,
            "start": "2030-01-01T00:00:00Z",
            "end": "2030-02-01T00:00:00Z",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["events"] == []
    assert body["event_count"] == 0
    assert body["sample_count"] == 0


def test_excursions_unknown_case_404(client):
    r = client.get("/api/v1/cases/NOPE/excursions", params={"temp_max": 25})
    assert r.status_code == 404


def test_excursions_validation_errors(client):
    _ingest(client, "CASE-E7", [
        _reading("CASE-E7", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-E7", "2026-09-01T08:10:00+08:00", 20.0, 50.0),
    ])
    url = "/api/v1/cases/CASE-E7/excursions"
    # 至少配置一个上下限
    assert client.get(url).status_code == 422
    # 上下限自相矛盾
    assert client.get(url, params={"temp_min": 30, "temp_max": 20}).status_code == 422
    assert (
        client.get(url, params={"humidity_min": 80, "humidity_max": 40}).status_code
        == 422
    )
    # start 晚于 end
    assert (
        client.get(
            url,
            params={
                "temp_max": 25,
                "start": "2030-01-01T00:00:00Z",
                "end": "2029-01-01T00:00:00Z",
            },
        ).status_code
        == 422
    )
    # max_gap 必须为正
    assert client.get(url, params={"temp_max": 25, "max_gap": 0}).status_code == 422
    assert client.get(url, params={"temp_max": 25, "max_gap": -5}).status_code == 422
    # 阈值超出量程
    assert client.get(url, params={"temp_max": 150}).status_code == 422
    assert client.get(url, params={"humidity_max": 120}).status_code == 422


def test_excursions_deterministic(client):
    rows = [
        _reading("CASE-E8", f"2026-09-01T08:{i:02d}:00+08:00", 20.0 + (i % 3) * 4, 50.0)
        for i in range(0, 60, 10)
    ]
    _ingest(client, "CASE-E8", rows)
    params = {"temp_max": 25, "humidity_min": 45, "max_gap": 900}
    first = client.get("/api/v1/cases/CASE-E8/excursions", params=params).json()
    second = client.get("/api/v1/cases/CASE-E8/excursions", params=params).json()
    assert first == second
    starts = [e["start_ts"] for e in first["events"]]
    assert starts == sorted(starts)
