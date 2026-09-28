"""集成测试：使用内存 SQLite，互不影响。"""

from __future__ import annotations

import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import (
    Base,
    CaseState,
    Reading,
    SegmentSnapshot,
    SessionLocal,
    engine,
    init_db,
)
from app.main import app, get_db


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


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


# ---------- 基础 ----------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_openapi_available(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["title"]


# ---------- 变点分段 ----------


def test_changepoint_detected_with_confidence(client):
    payload = []
    # 前 6 点 ~20.0/55，后 6 点 ~24.0/70（大幅、稳定的跳变）
    for i in range(6):
        h = 8 + i
        payload.append(_reading("CASE-1", f"2026-09-01T08:{h:02d}:00+08:00", 20.0, 55.0))
    for i in range(6):
        h = 20 + i
        payload.append(_reading("CASE-1", f"2026-09-01T09:{h:02d}:00+08:00", 24.0, 70.0))

    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["inserted"] == 12
    replay = body["summaries"][0]["replay"]
    confs = [s["confidence"] for s in replay["segments"]]
    assert confs[0] == "baseline"
    assert "high" in confs
    # 跳变段方向与幅度
    jump = next(s for s in replay["segments"] if s["confidence"] == "high")
    assert jump["delta_temperature"] == pytest.approx(4.0, abs=1e-6)
    assert jump["delta_humidity"] == pytest.approx(15.0, abs=1e-6)
    assert jump["magnitude"] > 0
    assert jump["start_ts"].startswith("2026-09-01T01:20:00")  # UTC


def test_single_segment_when_stable(client):
    payload = [
        _reading("CASE-2", f"2026-09-01T08:{i:02d}:00+08:00", 20.0 + 0.01 * (i % 2), 55.0)
        for i in range(8)
    ]
    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    segs = r.json()["summaries"][0]["replay"]["segments"]
    assert len(segs) == 1
    assert segs[0]["confidence"] == "baseline"


def test_deterministic(client):
    payload = []
    for i in range(5):
        payload.append(_reading("CASE-3", f"2026-09-01T08:{i:02d}:00+08:00", 20.0, 55.0))
    for i in range(5):
        payload.append(_reading("CASE-3", f"2026-09-01T09:{i:02d}:00+08:00", 22.5, 60.0))
    # 乱序上报
    shuffled = payload[::2] + payload[1::2]
    r1 = client.post("/api/v1/readings/batch", json={"readings": shuffled})
    s1 = r1.json()["summaries"][0]["replay"]

    # 第二次同样数据重放（全部命中合并）分段结构应完全一致
    r2 = client.post("/api/v1/readings/batch", json={"readings": payload})
    s2 = r2.json()["summaries"][0]["replay"]
    key = lambda s: [(x["start_ts"], x["end_ts"], x["sample_count"]) for x in s["segments"]]
    assert key(s1) == key(s2)


def test_penalty_monotonicity(client):
    payload = [
        _reading("CASE-4", f"2026-09-01T08:{i:02d}:00+08:00",
                 20.0 + (1.5 if i >= 4 else 0.0), 55.0)
        for i in range(8)
    ]
    low = client.post("/api/v1/readings/batch",
                      json={"readings": payload, "penalty": 0.5}).json()
    n_low = len(low["summaries"][0]["replay"]["segments"])

    high = client.post("/api/v1/readings/batch",
                       json={"readings": payload, "penalty": 100.0}).json()
    n_high = len(high["summaries"][0]["replay"]["segments"])
    assert n_low >= n_high
    assert n_high == 1


# ---------- 重复合并 ----------


def test_duplicate_reports_merged(client, db):
    base = [_reading("CASE-5", "2026-09-01T08:00:00+08:00", 20.0, 55.0)]
    client.post("/api/v1/readings/batch", json={"readings": base})
    again = [
        _reading("CASE-5", "2026-09-01T08:00:00+08:00", 22.0, 60.0),
        _reading("CASE-5", "2026-09-01T08:10:00+08:00", 21.0, 56.0),
    ]
    r = client.post("/api/v1/readings/batch", json={"readings": again})
    body = r.json()
    assert body["inserted"] == 1
    assert body["merged"] == 1

    db.expire_all()
    row = db.query(Reading).filter(Reading.device_sn == "CASE-5").order_by(Reading.ts).first()
    assert row.report_count == 2
    assert row.temperature == pytest.approx(21.0)
    assert row.humidity == pytest.approx(57.5)


# ---------- 迟到读数重放 ----------


def test_late_reading_replays_timeline(client):
    # 先上两段
    payload = [
        _reading("CASE-6", "2026-09-01T08:00:00+08:00", 20.0, 55.0),
        _reading("CASE-6", "2026-09-01T08:10:00+08:00", 20.0, 55.0),
        _reading("CASE-6", "2026-09-01T08:20:00+08:00", 20.0, 55.0),
        _reading("CASE-6", "2026-09-01T09:00:00+08:00", 20.0, 55.0),
        _reading("CASE-6", "2026-09-01T09:10:00+08:00", 20.0, 55.0),
        _reading("CASE-6", "2026-09-01T09:20:00+08:00", 20.0, 55.0),
    ]
    r1 = client.post("/api/v1/readings/batch", json={"readings": payload})
    v1 = r1.json()["summaries"][0]["after_version"]
    n1 = len(r1.json()["summaries"][0]["replay"]["segments"])

    # 迟到一个中间时刻的高读数
    late = [_reading("CASE-6", "2026-09-01T08:40:00+08:00", 26.0, 75.0)]
    r2 = client.post("/api/v1/readings/batch", json={"readings": late})
    summary = r2.json()["summaries"][0]
    assert summary["after_version"] == v1 + 1
    n2 = len(summary["replay"]["segments"])
    assert n2 >= n1


def test_multi_device_isolation(client):
    payload = [
        _reading("A", "2026-09-01T08:00:00+08:00", 20.0, 55.0),
        _reading("B", "2026-09-01T08:00:00+08:00", 20.0, 55.0),
    ]
    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    sns = {s["device_sn"] for s in r.json()["summaries"]}
    assert sns == {"A", "B"}
    assert set(r.json()["replayed_devices"]) == {"A", "B"}


# ---------- 锚点 ----------


def test_anchor_forces_boundary(client):
    payload = [
        _reading("CASE-7", f"2026-09-01T08:{i * 5:02d}:00+08:00", 20.0, 55.0)
        for i in range(10)
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    # 完全平稳的数据：无锚点时只有一段
    segs = client.get("/api/v1/cases/CASE-7/segments").json()["segments"]
    assert len(segs) == 1

    r = client.post("/api/v1/anchors", json={
        "device_sn": "CASE-7",
        "ts": "2026-09-01T08:25:00+08:00",
        "label": "开柜检查",
    })
    assert r.status_code == 200, r.text
    anchor = r.json()
    assert anchor["aligned"] is True
    assert anchor["aligned_ts"] == "2026-09-01T00:25:00Z"  # 统一 UTC 输出

    segs = client.get("/api/v1/cases/CASE-7/segments").json()["segments"]
    assert len(segs) == 2
    assert segs[1]["start_anchored"] is True
    assert segs[1]["confidence"] == "high"


def test_anchor_delete_replays(client):
    payload = [
        _reading("CASE-8", f"2026-09-01T08:{i * 5:02d}:00+08:00", 20.0, 55.0)
        for i in range(10)
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-8", "ts": "2026-09-01T08:30:00+08:00"})
    assert len(client.get("/api/v1/cases/CASE-8/segments").json()["segments"]) == 2

    r = client.delete("/api/v1/cases/CASE-8/anchors/2026-09-01T08:30:00+08:00")
    assert r.status_code == 200
    assert len(r.json()["segments"]) == 1


# ---------- 查询 ----------


def test_segment_time_filter(client):
    payload = [
        _reading("CASE-9", "2026-09-01T08:00:00+08:00", 20.0, 55.0),
        _reading("CASE-9", "2026-09-01T08:10:00+08:00", 20.0, 55.0),
        _reading("CASE-9", "2026-09-01T08:20:00+08:00", 20.0, 55.0),
        _reading("CASE-9", "2026-09-01T09:00:00+08:00", 24.0, 70.0),
        _reading("CASE-9", "2026-09-01T09:10:00+08:00", 24.0, 70.0),
        _reading("CASE-9", "2026-09-01T09:20:00+08:00", 24.0, 70.0),
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    r = client.get("/api/v1/cases/CASE-9/segments", params={
        "start": "2026-09-01T09:00:00+08:00",
        "end": "2026-09-01T10:00:00+08:00",
    })
    segs = r.json()["segments"]
    assert len(segs) == 1
    assert segs[0]["mean_temperature"] == pytest.approx(24.0)


def test_unknown_case_404(client):
    r = client.get("/api/v1/cases/NOPE/segments")
    assert r.status_code == 404


# ---------- 重放预览：不得写库 ----------


def test_preview_does_not_write(client, db):
    payload = [
        _reading("CASE-P", "2026-09-01T08:00:00+08:00", 20.0, 55.0),
        _reading("CASE-P", "2026-09-01T08:10:00+08:00", 20.0, 55.0),
        _reading("CASE-P", "2026-09-01T08:20:00+08:00", 20.0, 55.0),
        _reading("CASE-P", "2026-09-01T08:30:00+08:00", 20.0, 55.0),
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})

    version_before = db.get(CaseState, "CASE-P").version
    count_before = db.query(Reading).filter(Reading.device_sn == "CASE-P").count()

    r = client.post("/api/v1/replay/preview", params={"device_sn": "CASE-P"}, json={
        "readings": [
            _reading("CASE-P", "2026-09-01T09:00:00+08:00", 26.0, 75.0),
            _reading("CASE-P", "2026-09-01T09:10:00+08:00", 26.0, 75.0),
            _reading("CASE-P", "2026-09-01T09:20:00+08:00", 26.0, 75.0),
        ]
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] is True
    assert body["would_bump_version"] is True
    assert body["current_version"] == version_before
    assert body["preview"]["version"] == version_before + 1
    assert body["preview"]["replayed"] is False
    assert body["affected_from_ts"] is not None
    # 库内读数数量与版本号不变
    db.expire_all()
    assert db.query(Reading).filter(Reading.device_sn == "CASE-P").count() == count_before
    assert db.get(CaseState, "CASE-P").version == version_before

    # 预览中出现的变点在实际分段里尚不存在
    stored = client.get("/api/v1/cases/CASE-P/segments").json()
    assert len(stored["segments"]) == len(body["current"]["segments"])


def test_preview_with_anchor_does_not_write(client, db):
    payload = [
        _reading("CASE-Q", f"2026-09-01T08:{i * 5:02d}:00+08:00", 20.0, 55.0)
        for i in range(8)
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    r = client.post("/api/v1/replay/preview", params={"device_sn": "CASE-Q"}, json={
        "anchors": [{"device_sn": "CASE-Q", "ts": "2026-09-01T08:20:00+08:00"}]
    })
    body = r.json()
    assert body["added_anchors"] == 1
    assert len(body["preview"]["segments"]) == 2
    # 库内仍然只有一段、没有锚点
    stored = client.get("/api/v1/cases/CASE-Q/segments").json()
    assert len(stored["segments"]) == 1
    assert client.get("/api/v1/cases/CASE-Q/anchors").json() == []


def test_preview_identical_when_no_change(client):
    payload = [
        _reading("CASE-R", f"2026-09-01T08:{i * 5:02d}:00+08:00",
                 20.0 + (0.02 if i % 2 else -0.02), 55.0)
        for i in range(8)
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    r = client.post("/api/v1/replay/preview", params={"device_sn": "CASE-R"}, json={})
    body = r.json()
    assert body["changed"] is False
    assert body["would_bump_version"] is False
    assert body["affected_from_ts"] is None


# ---------- 预览合并重复读数（回归：不可变 Sample 不得被改写） ----------


def _snapshot_rows(db, sn):
    rows = (
        db.query(SegmentSnapshot)
        .filter(SegmentSnapshot.device_sn == sn)
        .order_by(SegmentSnapshot.segment_index)
        .all()
    )
    return [
        (r.version, r.segment_index, r.start_ts, r.end_ts, r.sample_count,
         r.mean_temperature, r.mean_humidity)
        for r in rows
    ]


def test_preview_merges_duplicate_of_stored_reading(client, db):
    payload = [
        _reading("CASE-PD", f"2026-09-01T08:{i:02d}:00+08:00", 20.0, 55.0)
        for i in range(0, 40, 10)
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    version_before = db.get(CaseState, "CASE-PD").version
    snapshots_before = _snapshot_rows(db, "CASE-PD")

    # 待预览读数与库内同设备、同时间戳重复：在内存副本上累计均值，返回 200
    body = {"readings": [_reading("CASE-PD", "2026-09-01T08:00:00+08:00", 22.0, 60.0)]}
    r = client.post("/api/v1/replay/preview", params={"device_sn": "CASE-PD"}, json=body)
    assert r.status_code == 200, r.text
    preview = r.json()
    assert preview["changed"] is True
    # 08:00 样本合并为 (20+22)/2=21.0 / (55+60)/2=57.5；合并不新增样本
    assert preview["preview"]["sample_count"] == 4
    seg = preview["preview"]["segments"][0]
    assert seg["mean_temperature"] == pytest.approx((21.0 + 20.0 * 3) / 4)
    assert seg["mean_humidity"] == pytest.approx((57.5 + 55.0 * 3) / 4)
    # 当前已存储分段仍保持原均值，差异对比稳定
    assert preview["current"]["segments"][0]["mean_temperature"] == pytest.approx(20.0)

    # 同样的预览重复执行，分段差异完全一致
    again = client.post("/api/v1/replay/preview", params={"device_sn": "CASE-PD"}, json=body)
    assert again.status_code == 200, again.text
    assert again.json()["preview"]["segments"] == preview["preview"]["segments"]
    assert again.json()["changed"] is True

    # 库内读数、快照与版本号均未被改写
    db.expire_all()
    row = (
        db.query(Reading)
        .filter(Reading.device_sn == "CASE-PD")
        .order_by(Reading.ts)
        .first()
    )
    assert row.report_count == 1
    assert row.temperature == pytest.approx(20.0)
    assert row.humidity == pytest.approx(55.0)
    assert db.get(CaseState, "CASE-PD").version == version_before
    assert _snapshot_rows(db, "CASE-PD") == snapshots_before


def test_preview_merges_duplicates_within_request(client, db):
    payload = [
        _reading("CASE-PR", f"2026-09-01T08:{i:02d}:00+08:00", 20.0, 55.0)
        for i in range(0, 30, 10)
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    version_before = db.get(CaseState, "CASE-PR").version

    # 请求内三条读数时间戳相同：逐条累计为一条 26.0 / 80.0
    r = client.post("/api/v1/replay/preview", params={"device_sn": "CASE-PR"}, json={
        "readings": [
            _reading("CASE-PR", "2026-09-01T09:00:00+08:00", 24.0, 70.0),
            _reading("CASE-PR", "2026-09-01T09:00:00+08:00", 26.0, 80.0),
            _reading("CASE-PR", "2026-09-01T09:00:00+08:00", 28.0, 90.0),
        ]
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added_readings"] == 3
    # 三条同 ts 只合并出一个新样本：3 + 1
    assert body["preview"]["sample_count"] == 4
    seg = body["preview"]["segments"][0]
    assert seg["mean_temperature"] == pytest.approx((20.0 * 3 + 26.0) / 4)
    assert seg["mean_humidity"] == pytest.approx((55.0 * 3 + 80.0) / 4)

    # 不落库：读数数量与版本号不变
    db.expire_all()
    assert db.query(Reading).filter(Reading.device_sn == "CASE-PR").count() == 3
    assert db.get(CaseState, "CASE-PR").version == version_before


def test_preview_duplicate_merge_respects_report_count(client, db):
    # 同一时刻两次上报，库内已合并为 report_count=2 的均值 21.0 / 52.0
    client.post("/api/v1/readings/batch", json={"readings": [
        _reading("CASE-PC", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-PC", "2026-09-01T08:10:00+08:00", 20.0, 55.0),
        _reading("CASE-PC", "2026-09-01T08:20:00+08:00", 20.0, 55.0),
    ]})
    client.post("/api/v1/readings/batch", json={"readings": [
        _reading("CASE-PC", "2026-09-01T08:00:00+08:00", 22.0, 54.0),
    ]})

    # 预览再补两条同 ts 读数：基于 report_count=2 逐条累计
    # 温度 (21*2+24)/3=22 → (22*3+26)/4=23；湿度 (52*2+55)/3=53 → (53*3+61)/4=55
    r = client.post("/api/v1/replay/preview", params={"device_sn": "CASE-PC"}, json={
        "readings": [
            _reading("CASE-PC", "2026-09-01T08:00:00+08:00", 24.0, 55.0),
            _reading("CASE-PC", "2026-09-01T08:00:00+08:00", 26.0, 61.0),
        ]
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] is True
    assert body["preview"]["sample_count"] == 3
    seg = body["preview"]["segments"][0]
    assert seg["mean_temperature"] == pytest.approx((23.0 + 20.0 + 20.0) / 3)
    assert seg["mean_humidity"] == pytest.approx(55.0)

    # 库内合并结果保持 report_count=2，不被预览推进
    db.expire_all()
    row = (
        db.query(Reading)
        .filter(Reading.device_sn == "CASE-PC")
        .order_by(Reading.ts)
        .first()
    )
    assert row.report_count == 2
    assert row.temperature == pytest.approx(21.0)
    assert row.humidity == pytest.approx(52.0)


# ---------- 时间线降采样 ----------


def _timeline_payload(sn, n=12):
    """带明显跳变、独立温湿度极值的 12 点序列（每 5 分钟一点）。"""
    return [
        _reading(
            sn,
            f"2026-09-01T08:{i * 5:02d}:00+08:00",
            20.0
            + (4.0 if 6 <= i <= 8 else 0.0)
            + (0.3 if i == 3 else 0.0)
            - (0.5 if i == n - 1 else 0.0),
            55.0 + (20.0 if i == 9 else 0.0) - (10.0 if i == 2 else 0.0),
        )
        for i in range(n)
    ]


def test_timeline_returns_all_when_budget_allows(client):
    client.post(
        "/api/v1/readings/batch", json={"readings": _timeline_payload("CASE-T0")}
    )
    r = client.get("/api/v1/cases/CASE-T0/timeline", params={"max_points": 50})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["original_count"] == 12
    assert body["returned_count"] == 12
    assert body["max_points"] == 50
    ts = [p["ts"] for p in body["points"]]
    assert ts == sorted(ts)
    assert body["start"] is None and body["end"] is None


def test_timeline_forced_points_and_lttb(client):
    client.post(
        "/api/v1/readings/batch", json={"readings": _timeline_payload("CASE-T1")}
    )
    client.post(
        "/api/v1/anchors",
        json={"device_sn": "CASE-T1", "ts": "2026-09-01T08:20:00+08:00"},
    )
    r = client.get("/api/v1/cases/CASE-T1/timeline", params={"max_points": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["original_count"] == 12
    assert body["returned_count"] == 10

    by_ts = {p["ts"]: p for p in body["points"]}
    first = body["points"][0]
    last = body["points"][-1]
    assert "range_start" in first["reasons"]
    assert "range_end" in last["reasons"]
    # 已对齐锚点（00:20 UTC = 08:20+08）
    assert "aligned_anchor" in by_ts["2026-09-01T00:20:00Z"]["reasons"]
    # 温湿度极值
    reasons_all = {x for p in body["points"] for x in p["reasons"]}
    for reason in (
        "temp_min",
        "temp_max",
        "humidity_min",
        "humidity_max",
        "segment_boundary",
        "lttb",
    ):
        assert reason in reasons_all
    # 每点原因按固定顺序、去重
    for p in body["points"]:
        assert p["reasons"] == sorted(set(p["reasons"]), key=p["reasons"].index)


def test_timeline_422_when_forced_points_exceed_budget(client):
    payload = [
        _reading("CASE-T2", f"2026-09-01T08:{i * 5:02d}:00+08:00", 20.0, 55.0)
        for i in range(8)
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    # 在每个读数上登记对齐锚点 → 8 个去重强制点
    for i in range(8):
        client.post(
            "/api/v1/anchors",
            json={
                "device_sn": "CASE-T2",
                "ts": f"2026-09-01T08:{i * 5:02d}:00+08:00",
            },
        )

    r = client.get("/api/v1/cases/CASE-T2/timeline", params={"max_points": 5})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["minimum_required"] == 8
    assert detail["max_points"] == 5

    # 达到下限时必须全量返回，不得静默丢点
    r = client.get("/api/v1/cases/CASE-T2/timeline", params={"max_points": 8})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["returned_count"] == 8
    assert body["original_count"] == 8


def test_timeline_start_end_filter_scopes_extremes_and_budget(client):
    client.post(
        "/api/v1/readings/batch", json={"readings": _timeline_payload("CASE-T3")}
    )
    client.post(
        "/api/v1/anchors",
        json={"device_sn": "CASE-T3", "ts": "2026-09-01T08:20:00+08:00"},
    )
    r = client.get(
        "/api/v1/cases/CASE-T3/timeline",
        params={
            "max_points": 50,
            "start": "2026-09-01T08:00:00+08:00",
            "end": "2026-09-01T08:15:00+08:00",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["original_count"] == 4  # 00,05,10,15 UTC
    assert body["returned_count"] == 4
    assert body["start"].startswith("2026-09-01T00:00:00")
    assert body["end"].startswith("2026-09-01T00:15:00")
    ts_list = [p["ts"] for p in body["points"]]
    assert ts_list[0] == "2026-09-01T00:00:00Z"
    assert ts_list[-1] == "2026-09-01T00:15:00Z"
    # 窗口外的锚点/分段边界与极值不参与强制点
    reasons_all = {x for p in body["points"] for x in p["reasons"]}
    assert "aligned_anchor" not in reasons_all
    # 窗口内首点湿度 55、次点 45 才是窗口湿度极小（全局湿度极小在窗口外）
    hum_min_point = next(
        p for p in body["points"] if "humidity_min" in p["reasons"]
    )
    assert hum_min_point["humidity"] == 45.0


def test_timeline_empty_range_returns_empty_array(client):
    client.post(
        "/api/v1/readings/batch", json={"readings": _timeline_payload("CASE-T4")}
    )
    r = client.get(
        "/api/v1/cases/CASE-T4/timeline",
        params={
            "max_points": 10,
            "start": "2030-01-01T00:00:00Z",
            "end": "2030-02-01T00:00:00Z",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["points"] == []
    assert body["original_count"] == 0
    assert body["returned_count"] == 0


def test_timeline_unknown_case_404(client):
    r = client.get("/api/v1/cases/NOPE/timeline", params={"max_points": 10})
    assert r.status_code == 404


def test_timeline_validation_errors(client):
    client.post(
        "/api/v1/readings/batch", json={"readings": _timeline_payload("CASE-T5")}
    )
    # max_points 必填、且 >= 1
    assert client.get("/api/v1/cases/CASE-T5/timeline").status_code == 422
    assert (
        client.get("/api/v1/cases/CASE-T5/timeline", params={"max_points": 0}).status_code
        == 422
    )
    # start 晚于 end
    assert (
        client.get(
            "/api/v1/cases/CASE-T5/timeline",
            params={
                "max_points": 10,
                "start": "2030-01-01T00:00:00Z",
                "end": "2029-01-01T00:00:00Z",
            },
        ).status_code
        == 422
    )


def test_timeline_deterministic(client):
    client.post(
        "/api/v1/readings/batch", json={"readings": _timeline_payload("CASE-T6")}
    )
    client.post(
        "/api/v1/anchors",
        json={"device_sn": "CASE-T6", "ts": "2026-09-01T08:25:00+08:00"},
    )
    params = {"max_points": 9}
    first = client.get("/api/v1/cases/CASE-T6/timeline", params=params).json()
    second = client.get("/api/v1/cases/CASE-T6/timeline", params=params).json()
    assert first == second
    # 独立乱序重放后结果仍一致
    shuffled = _timeline_payload("CASE-T6")[::-1]
    client.post("/api/v1/readings/batch", json={"readings": shuffled})
    third = client.get("/api/v1/cases/CASE-T6/timeline", params=params).json()
    assert [(p["ts"], p["reasons"]) for p in third["points"]] == [
        (p["ts"], p["reasons"]) for p in first["points"]
    ]


def test_timeline_without_prior_replay_uses_on_demand_segmentation(client):
    # 关闭自动重放：落库但没有分段快照，timeline 仍应即时计算分段边界
    r = client.post(
        "/api/v1/readings/batch",
        json={"readings": _timeline_payload("CASE-T7"), "replay": False},
    )
    assert r.status_code == 200
    r = client.get("/api/v1/cases/CASE-T7/timeline", params={"max_points": 10})
    assert r.status_code == 200, r.text
    reasons_all = {x for p in r.json()["points"] for x in p["reasons"]}
    assert "segment_boundary" in reasons_all
