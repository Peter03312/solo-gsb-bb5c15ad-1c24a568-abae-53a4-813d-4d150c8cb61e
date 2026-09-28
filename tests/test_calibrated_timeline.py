"""锚点参考值校正时间线：分维偏移曲线单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.calibration import AlignedAnchor, OffsetCurve, calibrate_timeline
from app.database import (
    Base,
    CaseState,
    Reading,
    SegmentSnapshot,
    SessionLocal,
    engine,
    init_db,
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


@pytest.fixture()
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _sample(minutes: float, t: float, h: float = 50.0) -> Sample:
    return Sample(BASE + dt.timedelta(minutes=minutes), t, h)


def _anchor(
    anchor_min: float,
    reading_min: float,
    t: float,
    h: float,
    ref_t: float | None,
    ref_h: float | None,
) -> AlignedAnchor:
    return AlignedAnchor(
        anchor_ts=BASE + dt.timedelta(minutes=anchor_min),
        reading_ts=BASE + dt.timedelta(minutes=reading_min),
        reading_temperature=t,
        reading_humidity=h,
        ref_temperature=ref_t,
        ref_humidity=ref_h,
    )


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


# ---------- 纯算法：分维偏移曲线 ----------


def test_offset_curve_interpolates_and_extrapolates():
    # 节点 (0min, +2.0) 与 (30min, -1.0)：之间线性插值，两端常数外推
    curve = OffsetCurve(
        [(BASE, 2.0), (BASE + dt.timedelta(minutes=30), -1.0)]
    )
    assert curve.at(BASE - dt.timedelta(minutes=5)) == 2.0  # 首节点之前
    assert curve.at(BASE) == 2.0
    assert curve.at(BASE + dt.timedelta(minutes=10)) == pytest.approx(1.0)
    assert curve.at(BASE + dt.timedelta(minutes=20)) == pytest.approx(0.0)
    assert curve.at(BASE + dt.timedelta(minutes=30)) == -1.0
    assert curve.at(BASE + dt.timedelta(minutes=45)) == -1.0  # 末节点之后


def test_offset_curve_empty_is_zero():
    curve = OffsetCurve([])
    assert curve.at(BASE) == 0.0
    assert curve.knot_count == 0


def test_calibrate_dimensions_independent():
    # 锚点只带温度参考值：湿度维没有有效锚点，偏移恒为 0
    samples = [_sample(0, 20.0, 50.0), _sample(10, 21.0, 51.0)]
    anchors = [_anchor(0, 0, 20.0, 50.0, ref_t=22.5, ref_h=None)]
    out = calibrate_timeline(samples, anchors)
    assert [p.temperature_offset for p in out] == [2.5, 2.5]
    assert [p.humidity_offset for p in out] == [0.0, 0.0]
    assert out[1].calibrated_temperature == pytest.approx(23.5)
    assert out[1].calibrated_humidity == pytest.approx(51.0)


def test_calibrate_missing_dimension_anchor_skipped_per_dimension():
    # 两个锚点：A 只有温度参考、B 只有湿度参考 → 两维各自独立成曲线
    samples = [_sample(0, 20.0, 50.0), _sample(30, 23.0, 53.0)]
    anchors = [
        _anchor(0, 0, 20.0, 50.0, ref_t=21.0, ref_h=None),
        _anchor(30, 30, 23.0, 53.0, ref_t=None, ref_h=58.0),
    ]
    out = calibrate_timeline(samples, anchors)
    # 温度维只有 A：全程 +1.0；湿度维只有 B：全程 +5.0
    assert [p.temperature_offset for p in out] == [1.0, 1.0]
    assert [p.humidity_offset for p in out] == [5.0, 5.0]


def test_calibrate_multiple_anchors_same_reading_last_wins():
    # 两个锚点对齐同一读数：按锚点时间升序取最后一个有效参考值
    samples = [_sample(0, 20.0, 50.0)]
    anchors = [
        _anchor(5, 0, 20.0, 50.0, ref_t=25.0, ref_h=70.0),  # 更晚 → 生效
        _anchor(1, 0, 20.0, 50.0, ref_t=21.0, ref_h=55.0),
    ]
    out = calibrate_timeline(samples, anchors)
    assert out[0].temperature_offset == pytest.approx(5.0)
    assert out[0].humidity_offset == pytest.approx(20.0)


def test_calibrate_later_anchor_without_dimension_ref_does_not_override():
    # 更晚的锚点缺少该维参考值：不参与该维曲线，较早锚点仍然生效
    samples = [_sample(0, 20.0, 50.0)]
    anchors = [
        _anchor(1, 0, 20.0, 50.0, ref_t=21.0, ref_h=55.0),
        _anchor(5, 0, 20.0, 50.0, ref_t=None, ref_h=70.0),
    ]
    out = calibrate_timeline(samples, anchors)
    assert out[0].temperature_offset == pytest.approx(1.0)  # 来自较早锚点
    assert out[0].humidity_offset == pytest.approx(20.0)  # 来自更晚锚点


def test_calibrate_no_anchors_offsets_zero():
    samples = [_sample(0, 20.0, 50.0), _sample(10, 21.0, 51.0)]
    out = calibrate_timeline(samples, [])
    for p in out:
        assert p.temperature_offset == 0.0
        assert p.humidity_offset == 0.0
        assert p.calibrated_temperature == p.temperature
        assert p.calibrated_humidity == p.humidity


# ---------- 接口集成 ----------


def _post_readings(client, sn="CASE-CAL"):
    """6 个读数（每 10 分钟一点），温度 20→23 缓升，湿度 50→53。"""
    payload = [
        _reading(
            sn,
            f"2026-09-01T08:{i * 10:02d}:00+08:00",
            20.0 + 0.6 * i,
            50.0 + 0.6 * i,
        )
        for i in range(6)
    ]
    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    assert r.status_code == 200, r.text
    return payload


def test_calibrated_timeline_basic_interpolation(client):
    _post_readings(client)
    # 锚点对齐 08:00（UTC 00:00，读数 20.0/50.0）与 08:50（读数 23.0/53.0）
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:00:00+08:00",
        "ref_temperature": 22.0, "ref_humidity": 52.0,
    })
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:50:00+08:00",
        "ref_temperature": 24.0, "ref_humidity": 58.0,
    })

    r = client.get("/api/v1/cases/CASE-CAL/calibrated-timeline")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "CASE-CAL"
    assert body["sample_count"] == 6
    assert body["aligned_anchor_count"] == 2
    assert body["start"] is None and body["end"] is None

    pts = body["points"]
    assert [p["ts"] for p in pts] == sorted(p["ts"] for p in pts)
    # 温度偏移：首节点 +2.0，末节点 +1.0，中间按时间线性插值（步长 -0.2）
    temp_offsets = [p["temperature_offset"] for p in pts]
    assert temp_offsets == pytest.approx([2.0, 1.8, 1.6, 1.4, 1.2, 1.0])
    # 湿度偏移：首节点 +2.0，末节点 +5.0（步长 +0.6）
    hum_offsets = [p["humidity_offset"] for p in pts]
    assert hum_offsets == pytest.approx([2.0, 2.6, 3.2, 3.8, 4.4, 5.0])
    # 校正值 = 原值 + 偏移
    for p in pts:
        assert p["calibrated_temperature"] == pytest.approx(
            p["temperature"] + p["temperature_offset"]
        )
        assert p["calibrated_humidity"] == pytest.approx(
            p["humidity"] + p["humidity_offset"]
        )
    # 首末读数的校正值恰等于参考值
    assert pts[0]["calibrated_temperature"] == pytest.approx(22.0)
    assert pts[0]["calibrated_humidity"] == pytest.approx(52.0)
    assert pts[-1]["calibrated_temperature"] == pytest.approx(24.0)
    assert pts[-1]["calibrated_humidity"] == pytest.approx(58.0)


def test_calibrated_timeline_extrapolates_beyond_anchors(client):
    _post_readings(client)
    # 只对中间两个读数（08:20、08:30）登记锚点
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:20:00+08:00",
        "ref_temperature": 23.0, "ref_humidity": 53.0,
    })
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:30:00+08:00",
        "ref_temperature": 24.0, "ref_humidity": 55.0,
    })
    r = client.get("/api/v1/cases/CASE-CAL/calibrated-timeline")
    pts = r.json()["points"]
    # 08:20 读数 21.2/51.2 → 偏移 +1.8/+1.8；08:30 读数 21.8/51.8 → +2.2/+3.2
    # 首区间沿用首节点偏移，末区间沿用末节点偏移
    temp_offsets = [p["temperature_offset"] for p in pts]
    assert temp_offsets == pytest.approx([1.8, 1.8, 1.8, 2.2, 2.2, 2.2])
    hum_offsets = [p["humidity_offset"] for p in pts]
    assert hum_offsets == pytest.approx([1.8, 1.8, 1.8, 3.2, 3.2, 3.2])


def test_calibrated_timeline_curve_built_before_filter(client):
    """先基于整条时间线建曲线再过滤：窗口外锚点仍影响窗口内读数。"""
    _post_readings(client)
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:00:00+08:00",
        "ref_temperature": 22.0, "ref_humidity": 52.0,
    })
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:50:00+08:00",
        "ref_temperature": 24.0, "ref_humidity": 58.0,
    })
    # 只取中间三个读数（08:20、08:30、08:40，UTC 00:20–00:40）
    r = client.get(
        "/api/v1/cases/CASE-CAL/calibrated-timeline",
        params={
            "start": "2026-09-01T08:20:00+08:00",
            "end": "2026-09-01T08:40:00+08:00",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_count"] == 3
    assert body["start"].startswith("2026-09-01T00:20:00")
    assert body["end"].startswith("2026-09-01T00:40:00")
    # 偏移与全量查询时对应点完全一致（曲线不受过滤窗口影响）
    temp_offsets = [p["temperature_offset"] for p in body["points"]]
    assert temp_offsets == pytest.approx([1.6, 1.4, 1.2])
    hum_offsets = [p["humidity_offset"] for p in body["points"]]
    assert hum_offsets == pytest.approx([3.2, 3.8, 4.4])


def test_calibrated_timeline_dimension_independent(client):
    _post_readings(client)
    # 温度锚点 + 只带湿度参考值的锚点：两维各自独立
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:00:00+08:00",
        "ref_temperature": 21.0,
    })
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:30:00+08:00",
        "ref_humidity": 60.0,
    })
    r = client.get("/api/v1/cases/CASE-CAL/calibrated-timeline")
    pts = r.json()["points"]
    # 温度维：唯一节点 +1.0，全程常数
    assert [p["temperature_offset"] for p in pts] == pytest.approx([1.0] * 6)
    # 湿度维：唯一节点 60.0 − 51.8 = +8.2，全程常数
    assert [p["humidity_offset"] for p in pts] == pytest.approx([8.2] * 6)


def test_calibrated_timeline_multiple_anchors_same_reading(client):
    _post_readings(client)
    # 两个锚点对齐同一读数（08:00）：按锚点时间升序，后者生效
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:00:00+08:00",
        "ref_temperature": 21.0, "ref_humidity": 51.0,
    })
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:01:00+08:00",
        "ref_temperature": 25.0, "ref_humidity": 55.0,
    })
    anchors = client.get("/api/v1/cases/CASE-CAL/anchors").json()
    assert all(a["aligned"] and a["aligned_ts"].endswith("00:00:00Z") for a in anchors)

    r = client.get("/api/v1/cases/CASE-CAL/calibrated-timeline")
    pts = r.json()["points"]
    # 唯一节点取后者：温度 25 − 20 = +5，湿度 55 − 50 = +5，全程常数
    assert [p["temperature_offset"] for p in pts] == pytest.approx([5.0] * 6)
    assert [p["humidity_offset"] for p in pts] == pytest.approx([5.0] * 6)


def test_calibrated_timeline_no_anchors_returns_zero_offsets(client):
    _post_readings(client)
    r = client.get("/api/v1/cases/CASE-CAL/calibrated-timeline")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["aligned_anchor_count"] == 0
    for p in body["points"]:
        assert p["temperature_offset"] == 0.0
        assert p["humidity_offset"] == 0.0
        assert p["calibrated_temperature"] == p["temperature"]
        assert p["calibrated_humidity"] == p["humidity"]


def test_calibrated_timeline_unaligned_anchor_ignored(client):
    _post_readings(client)
    # 锚点远离任何读数（容差 300 秒之外）：未对齐，不参与曲线
    r = client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T12:00:00+08:00",
        "ref_temperature": 30.0, "ref_humidity": 90.0,
    })
    assert r.json()["aligned"] is False
    body = client.get("/api/v1/cases/CASE-CAL/calibrated-timeline").json()
    assert body["aligned_anchor_count"] == 0
    assert all(p["temperature_offset"] == 0.0 for p in body["points"])
    assert all(p["humidity_offset"] == 0.0 for p in body["points"])


def test_calibrated_timeline_empty_range_returns_empty(client):
    _post_readings(client)
    r = client.get(
        "/api/v1/cases/CASE-CAL/calibrated-timeline",
        params={"start": "2030-01-01T00:00:00Z", "end": "2030-02-01T00:00:00Z"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["points"] == []
    assert body["sample_count"] == 0


def test_calibrated_timeline_unknown_case_404(client):
    r = client.get("/api/v1/cases/NOPE/calibrated-timeline")
    assert r.status_code == 404


def test_calibrated_timeline_start_after_end_422(client):
    _post_readings(client)
    r = client.get(
        "/api/v1/cases/CASE-CAL/calibrated-timeline",
        params={"start": "2030-01-01T00:00:00Z", "end": "2029-01-01T00:00:00Z"},
    )
    assert r.status_code == 422


def test_calibrated_timeline_is_read_only(client, db):
    _post_readings(client)
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:00:00+08:00",
        "ref_temperature": 22.0, "ref_humidity": 52.0,
    })
    version_before = db.get(CaseState, "CASE-CAL").version
    count_before = db.query(Reading).filter(Reading.device_sn == "CASE-CAL").count()
    snapshots_before = [
        (row.version, row.segment_index, row.start_ts, row.end_ts)
        for row in db.query(SegmentSnapshot)
        .filter(SegmentSnapshot.device_sn == "CASE-CAL")
        .order_by(SegmentSnapshot.segment_index)
    ]

    r = client.get("/api/v1/cases/CASE-CAL/calibrated-timeline")
    assert r.status_code == 200, r.text

    # 读数、分段快照与版本号均未被修改
    db.expire_all()
    assert db.query(Reading).filter(Reading.device_sn == "CASE-CAL").count() == count_before
    assert db.get(CaseState, "CASE-CAL").version == version_before
    snapshots_after = [
        (row.version, row.segment_index, row.start_ts, row.end_ts)
        for row in db.query(SegmentSnapshot)
        .filter(SegmentSnapshot.device_sn == "CASE-CAL")
        .order_by(SegmentSnapshot.segment_index)
    ]
    assert snapshots_after == snapshots_before


def test_calibrated_timeline_deterministic(client):
    _post_readings(client)
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-CAL", "ts": "2026-09-01T08:10:00+08:00",
        "ref_temperature": 21.5, "ref_humidity": 53.5,
    })
    first = client.get("/api/v1/cases/CASE-CAL/calibrated-timeline").json()
    second = client.get("/api/v1/cases/CASE-CAL/calibrated-timeline").json()
    assert first == second
