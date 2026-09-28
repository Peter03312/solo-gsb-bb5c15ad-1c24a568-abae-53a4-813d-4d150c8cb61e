"""冷凝风险：Magnus 露点 / 裕量插值窗口单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.condensation import compute_condensation_windows, dew_point
from app.database import Base, engine, init_db
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


def _sample(minutes: float, t: float, h: float = 100.0) -> Sample:
    return Sample(BASE + dt.timedelta(minutes=minutes), t, h)


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


# ---------- Magnus 公式 ----------


def test_dew_point_saturation_equals_temperature():
    # RH=100% 时露点等于空气温度
    assert dew_point(20.0, 100.0) == pytest.approx(20.0)
    assert dew_point(0.0, 100.0) == pytest.approx(0.0)
    assert dew_point(-10.0, 100.0) == pytest.approx(-10.0)


def test_dew_point_known_value():
    # 20°C / 50%RH 的露点约 9.27°C（Magnus 常用表值 9.27）
    assert dew_point(20.0, 50.0) == pytest.approx(9.27, abs=0.02)
    # 30°C / 80%RH 的露点约 26.17°C
    assert dew_point(30.0, 80.0) == pytest.approx(26.17, abs=0.02)


def test_dew_point_monotonic():
    # 同温度下湿度越高露点越高；同湿度下温度越高露点越高
    assert dew_point(20.0, 30.0) < dew_point(20.0, 60.0) < dew_point(20.0, 90.0)
    assert dew_point(10.0, 60.0) < dew_point(20.0, 60.0) < dew_point(30.0, 60.0)


# ---------- 纯算法：线性插值窗口 ----------


def test_entire_interval_at_risk():
    # surface=12, RH=100% → margin=12−T；m0=−8, m10=−18，全程风险
    windows = compute_condensation_windows(
        [_sample(0, 20.0), _sample(10, 30.0)],
        surface_temperature_c=12.0,
        margin_threshold_c=2.0,
        max_gap_seconds=600.0,
        min_duration_seconds=0.0,
    )
    assert len(windows) == 1
    w = windows[0]
    assert w.start_ts == BASE
    assert w.end_ts == BASE + dt.timedelta(minutes=10)
    assert w.duration_seconds == 600.0
    assert w.min_margin == pytest.approx(-18.0)
    assert w.min_margin_ts == BASE + dt.timedelta(minutes=10)


def test_threshold_crossing_exact_midpoint():
    # surface=10, RH=100% → margin=10−T：m0=−10, m10=10, m20=−10；
    # 阈值 0：风险窗口 [0,5min] 与 [15min,20min]，交点恰在中点
    windows = compute_condensation_windows(
        [_sample(0, 20.0), _sample(10, 0.0), _sample(20, 20.0)],
        10.0, 0.0, 600.0, 0.0,
    )
    assert len(windows) == 2
    first, second = windows
    assert first.start_ts == BASE
    assert first.end_ts == BASE + dt.timedelta(minutes=5)
    assert first.duration_seconds == 300.0
    assert first.min_margin == pytest.approx(-10.0)
    assert first.min_margin_ts == BASE
    assert second.start_ts == BASE + dt.timedelta(minutes=15)
    assert second.end_ts == BASE + dt.timedelta(minutes=20)
    assert second.min_margin_ts == BASE + dt.timedelta(minutes=20)


def test_margin_equal_threshold_counts_as_risk():
    # 裕量恰好等于阈值（含等于）即风险：m0=2 → 风险从 t0 起
    windows = compute_condensation_windows(
        [_sample(0, 20.0), _sample(10, 30.0)], 12.0, 2.0, 600.0, 0.0
    )
    assert len(windows) == 1
    assert windows[0].start_ts == BASE


def test_touching_at_threshold_merges_into_one_window():
    # m: −10 → 0 → −10，阈值 0：两子区间在中间读数（恰为阈值）相接，合并
    windows = compute_condensation_windows(
        [_sample(0, 20.0), _sample(10, 10.0), _sample(20, 20.0)],
        10.0, 0.0, 600.0, 0.0,
    )
    assert len(windows) == 1
    w = windows[0]
    assert w.start_ts == BASE
    assert w.end_ts == BASE + dt.timedelta(minutes=20)
    assert w.duration_seconds == 1200.0
    assert w.min_margin == pytest.approx(-10.0)
    # 最低裕量并列取最早
    assert w.min_margin_ts == BASE


def test_min_margin_at_interior_reading():
    # m: 0 → −5 → 0（threshold 0 含等于）：最低裕量在中间读数
    windows = compute_condensation_windows(
        [_sample(0, 10.0), _sample(10, 15.0), _sample(20, 10.0)],
        10.0, 0.0, 600.0, 0.0,
    )
    assert len(windows) == 1
    w = windows[0]
    assert w.start_ts == BASE
    assert w.end_ts == BASE + dt.timedelta(minutes=20)
    assert w.min_margin == pytest.approx(-5.0)
    assert w.min_margin_ts == BASE + dt.timedelta(minutes=10)


def test_gap_breaks_windows():
    # 0→10 连续；10→30 间隔 20min 超过 max_gap=600：两侧窗口不合并
    windows = compute_condensation_windows(
        [_sample(0, 20.0), _sample(10, 30.0), _sample(30, 30.0)],
        12.0, 2.0, 600.0, 0.0,
    )
    assert len(windows) == 1
    assert windows[0].end_ts == BASE + dt.timedelta(minutes=10)


def test_gap_equal_max_gap_stays_connected():
    windows = compute_condensation_windows(
        [_sample(0, 30.0), _sample(10, 30.0)], 12.0, 2.0, 600.0, 0.0
    )
    assert len(windows) == 1
    assert windows[0].duration_seconds == 600.0
    # 间隔稍超 max_gap：孤立点无相邻有效区间，无窗口
    assert compute_condensation_windows(
        [_sample(0, 30.0), _sample(10, 30.0)], 12.0, 2.0, 599.0, 0.0
    ) == []


def test_zero_humidity_breaks_curve():
    # 中间读数湿度为 0：其两侧区间必须断开，即便时间间隔未超 max_gap
    windows = compute_condensation_windows(
        [_sample(0, 20.0, 100.0), _sample(10, 30.0, 100.0),
         _sample(20, 30.0, 0.0), _sample(30, 30.0, 100.0)],
        12.0, 2.0, 600.0, 0.0,
    )
    assert len(windows) == 1
    w = windows[0]
    assert w.start_ts == BASE
    assert w.end_ts == BASE + dt.timedelta(minutes=10)


def test_zero_humidity_isolated_point_no_window():
    # 仅一个湿度为 0 的读数：露点不定义，无窗口
    assert compute_condensation_windows(
        [_sample(0, 20.0, 0.0)], 12.0, 2.0, 600.0, 0.0
    ) == []


def test_min_duration_filters_short_windows():
    # 风险窗口 [0,5min] 与 [15,20min]：min_duration=300 恰好保留（含等于），
    # min_duration=301 全部过滤
    samples = [_sample(0, 20.0), _sample(10, 0.0), _sample(20, 20.0)]
    kept = compute_condensation_windows(samples, 10.0, 0.0, 600.0, 300.0)
    assert len(kept) == 2
    filtered = compute_condensation_windows(samples, 10.0, 0.0, 600.0, 301.0)
    assert filtered == []


def test_no_risk_no_windows():
    # 裕量全程远高于阈值
    assert compute_condensation_windows(
        [_sample(0, 10.0), _sample(10, 10.0)], 30.0, 0.0, 600.0, 0.0
    ) == []
    # 单点无法构成区间
    assert compute_condensation_windows(
        [_sample(0, 40.0)], 0.0, 2.0, 600.0, 0.0
    ) == []


def test_windows_sorted_and_unsorted_input_deterministic():
    # m: −10,10,−10,10,−10（threshold 0）：风险子区间 [0,5]、[15,20]、
    # [20,25] 在 t=20 相接合并为 [15,25]，再加 [35,40] → 共 3 个窗口
    samples = [
        _sample(0, 20.0), _sample(10, 0.0), _sample(20, 20.0),
        _sample(30, 0.0), _sample(40, 20.0),
    ]
    ordered = compute_condensation_windows(samples, 10.0, 0.0, 600.0, 0.0)
    shuffled = compute_condensation_windows(samples[::-1], 10.0, 0.0, 600.0, 0.0)
    assert ordered == shuffled
    assert len(ordered) == 3
    assert [w.start_ts for w in ordered] == sorted(w.start_ts for w in ordered)
    assert ordered[0].start_ts == BASE
    assert ordered[0].end_ts == BASE + dt.timedelta(minutes=5)
    assert ordered[1].start_ts == BASE + dt.timedelta(minutes=15)
    assert ordered[1].end_ts == BASE + dt.timedelta(minutes=25)
    assert ordered[1].duration_seconds == 600.0
    assert ordered[2].start_ts == BASE + dt.timedelta(minutes=35)
    assert ordered[2].end_ts == BASE + dt.timedelta(minutes=40)


# ---------- 接口集成 ----------


def _ingest(client, sn, rows):
    r = client.post("/api/v1/readings/batch", json={"readings": rows})
    assert r.status_code == 200, r.text


def test_condensation_endpoint_basic(client):
    _ingest(client, "CASE-C1", [
        _reading("CASE-C1", "2026-09-01T08:00:00+08:00", 20.0, 100.0),
        _reading("CASE-C1", "2026-09-01T08:10:00+08:00", 0.0, 100.0),
        _reading("CASE-C1", "2026-09-01T08:20:00+08:00", 20.0, 100.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-C1/condensation-risk",
        params={"surface_temperature_c": 10, "margin_threshold_c": 0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "CASE-C1"
    assert body["surface_temperature_c"] == 10.0
    assert body["margin_threshold_c"] == 0.0
    assert body["max_gap_seconds"] == 600.0
    assert body["min_duration_seconds"] == 0.0
    assert body["sample_count"] == 3
    assert body["window_count"] == 2

    first = body["windows"][0]
    assert _ts(first["start_ts"]) == _ts("2026-09-01T00:00:00+00:00")
    assert _ts(first["end_ts"]) == _ts("2026-09-01T00:05:00+00:00")
    assert first["duration_seconds"] == pytest.approx(300.0)
    assert first["min_margin"] == pytest.approx(-10.0)
    assert _ts(first["min_margin_ts"]) == _ts("2026-09-01T00:00:00+00:00")

    second = body["windows"][1]
    assert _ts(second["start_ts"]) == _ts("2026-09-01T00:15:00+00:00")
    assert _ts(second["end_ts"]) == _ts("2026-09-01T00:20:00+00:00")
    assert _ts(second["min_margin_ts"]) == _ts("2026-09-01T00:20:00+00:00")


def test_condensation_min_duration_param(client):
    _ingest(client, "CASE-C2", [
        _reading("CASE-C2", "2026-09-01T08:00:00+08:00", 20.0, 100.0),
        _reading("CASE-C2", "2026-09-01T08:10:00+08:00", 0.0, 100.0),
        _reading("CASE-C2", "2026-09-01T08:20:00+08:00", 20.0, 100.0),
    ])
    url = "/api/v1/cases/CASE-C2/condensation-risk"
    params = {"surface_temperature_c": 10, "margin_threshold_c": 0,
              "min_duration_seconds": 400}
    body = client.get(url, params=params).json()
    assert body["window_count"] == 0
    assert body["windows"] == []
    # 恰好等于最短持续：保留
    params["min_duration_seconds"] = 300
    body = client.get(url, params=params).json()
    assert body["window_count"] == 2


def test_condensation_max_gap_and_zero_humidity_via_api(client):
    _ingest(client, "CASE-C3", [
        _reading("CASE-C3", "2026-09-01T08:00:00+08:00", 20.0, 100.0),
        _reading("CASE-C3", "2026-09-01T08:10:00+08:00", 30.0, 100.0),
        _reading("CASE-C3", "2026-09-01T08:20:00+08:00", 30.0, 0.0),
        _reading("CASE-C3", "2026-09-01T08:30:00+08:00", 30.0, 100.0),
        _reading("CASE-C3", "2026-09-01T10:00:00+08:00", 30.0, 100.0),
    ])
    url = "/api/v1/cases/CASE-C3/condensation-risk"
    params = {"surface_temperature_c": 12, "margin_threshold_c": 2}
    body = client.get(url, params=params).json()
    # 仅 [08:00, 08:10] 成窗：08:20 湿度 0 断开；08:30→10:00 超 max_gap
    assert body["window_count"] == 1
    assert _ts(body["windows"][0]["end_ts"]) == _ts("2026-09-01T00:10:00+00:00")
    assert body["sample_count"] == 5  # 含湿度 0 的无效读数

    # 放宽 max_gap：08:30→10:00 连续，新增第二个窗口（仍不跨 08:20）
    body = client.get(url, params={**params, "max_gap_seconds": 7200}).json()
    assert body["window_count"] == 2
    assert _ts(body["windows"][1]["start_ts"]) == _ts("2026-09-01T00:30:00+00:00")


def test_condensation_start_end_filter(client):
    _ingest(client, "CASE-C4", [
        _reading("CASE-C4", "2026-09-01T08:00:00+08:00", 20.0, 100.0),
        _reading("CASE-C4", "2026-09-01T08:10:00+08:00", 0.0, 100.0),
        _reading("CASE-C4", "2026-09-01T08:20:00+08:00", 20.0, 100.0),
    ])
    body = client.get(
        "/api/v1/cases/CASE-C4/condensation-risk",
        params={
            "surface_temperature_c": 10,
            "margin_threshold_c": 0,
            "start": "2026-09-01T08:10:00+08:00",
        },
    ).json()
    assert body["sample_count"] == 2
    assert body["window_count"] == 1
    # 曲线从 08:10 起：风险仅在 [08:15, 08:20]
    w = body["windows"][0]
    assert _ts(w["start_ts"]) == _ts("2026-09-01T00:15:00+00:00")
    assert _ts(w["end_ts"]) == _ts("2026-09-01T00:20:00+00:00")
    assert body["start"].startswith("2026-09-01T00:10:00")


def test_condensation_empty_range_returns_empty(client):
    _ingest(client, "CASE-C5", [
        _reading("CASE-C5", "2026-09-01T08:00:00+08:00", 20.0, 100.0),
        _reading("CASE-C5", "2026-09-01T08:10:00+08:00", 30.0, 100.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-C5/condensation-risk",
        params={
            "surface_temperature_c": 12,
            "margin_threshold_c": 2,
            "start": "2030-01-01T00:00:00Z",
            "end": "2030-02-01T00:00:00Z",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["windows"] == []
    assert body["window_count"] == 0
    assert body["sample_count"] == 0


def test_condensation_unknown_case_404(client):
    r = client.get(
        "/api/v1/cases/NOPE/condensation-risk",
        params={"surface_temperature_c": 12, "margin_threshold_c": 2},
    )
    assert r.status_code == 404


def test_condensation_validation_errors(client):
    _ingest(client, "CASE-C6", [
        _reading("CASE-C6", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-C6", "2026-09-01T08:10:00+08:00", 20.0, 50.0),
    ])
    url = "/api/v1/cases/CASE-C6/condensation-risk"
    # 缺少必填参数
    assert client.get(url, params={"margin_threshold_c": 2}).status_code == 422
    assert client.get(url, params={"surface_temperature_c": 12}).status_code == 422
    # start 晚于 end
    assert client.get(url, params={
        "surface_temperature_c": 12,
        "margin_threshold_c": 2,
        "start": "2030-01-01T00:00:00Z",
        "end": "2029-01-01T00:00:00Z",
    }).status_code == 422
    # max_gap_seconds 必须为正
    assert client.get(url, params={
        "surface_temperature_c": 12, "margin_threshold_c": 2,
        "max_gap_seconds": 0,
    }).status_code == 422
    assert client.get(url, params={
        "surface_temperature_c": 12, "margin_threshold_c": 2,
        "max_gap_seconds": -5,
    }).status_code == 422
    # min_duration_seconds 不可为负
    assert client.get(url, params={
        "surface_temperature_c": 12, "margin_threshold_c": 2,
        "min_duration_seconds": -1,
    }).status_code == 422
    # 表面温度超出量程
    assert client.get(url, params={
        "surface_temperature_c": 150, "margin_threshold_c": 2,
    }).status_code == 422


def test_condensation_deterministic(client):
    rows = [
        _reading("CASE-C7", f"2026-09-01T08:{i:02d}:00+08:00",
                 10.0 + (i % 3) * 8.0, 90.0)
        for i in range(0, 60, 10)
    ]
    _ingest(client, "CASE-C7", rows)
    params = {"surface_temperature_c": 15, "margin_threshold_c": 1,
              "max_gap_seconds": 900}
    first = client.get(
        "/api/v1/cases/CASE-C7/condensation-risk", params=params
    ).json()
    second = client.get(
        "/api/v1/cases/CASE-C7/condensation-risk", params=params
    ).json()
    assert first == second
    starts = [w["start_ts"] for w in first["windows"]]
    assert starts == sorted(starts)
