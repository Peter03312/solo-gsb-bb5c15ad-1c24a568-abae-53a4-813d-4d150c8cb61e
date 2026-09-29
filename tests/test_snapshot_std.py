"""两维段内总体标准差持久化与旧快照恢复回归。

覆盖本轮缺陷：重放后重复查询、服务重启后的分段查询必须返回与该次
重放相同的 temperature_std / humidity_std；旧版快照（无这两列）按
读数核实补写；读数已无法对应时明确 409，不把缺失值当零、不改分段版本。
"""

from __future__ import annotations

import math
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

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


def _popstd(values: list[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def _varied_payload(sn="CASE-STD"):
    # 两段：前段温度小幅抖动（非恒定），后段整段抬升
    temps = [20.0, 20.4, 19.8, 20.2, 24.2, 23.8, 24.0, 24.4]
    hums = [55.0, 55.4, 54.6, 55.0, 70.2, 69.8, 70.0, 70.4]
    return [
        _reading(sn, f"2026-09-01T08:{i * 5:02d}:00+08:00", temps[i], hums[i])
        for i in range(8)
    ], temps, hums


# ---------- 即时重放与后续查询一致 ----------


def test_replay_std_matches_followup_segment_queries(client):
    payload, temps, hums = _varied_payload()
    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    assert r.status_code == 200, r.text
    replay_segs = r.json()["summaries"][0]["replay"]["segments"]

    # 即时重放结果中存在非零 std（非恒定读数）
    assert any(s["temperature_std"] > 0 or s["humidity_std"] > 0 for s in replay_segs)

    # 重放后立即重复查询：两维 std 与即时重放完全一致
    q1 = client.get("/api/v1/cases/CASE-STD/segments").json()["segments"]
    assert [
        (s["temperature_std"], s["humidity_std"]) for s in q1
    ] == [
        (s["temperature_std"], s["humidity_std"]) for s in replay_segs
    ]
    # 再来一次同样查询，结果稳定
    q2 = client.get("/api/v1/cases/CASE-STD/segments").json()["segments"]
    assert q2 == q1

    # 数值可由读数核实（总体标准差，除以 n 而非 n-1）
    by_count = {s["sample_count"]: s for s in replay_segs}
    n0 = replay_segs[0]["sample_count"]
    assert replay_segs[0]["temperature_std"] == pytest.approx(_popstd(temps[:n0]), rel=1e-9)
    assert replay_segs[0]["humidity_std"] == pytest.approx(_popstd(hums[:n0]), rel=1e-9)
    assert by_count is not None


def test_std_persisted_in_snapshot_rows(client, db):
    payload, temps, hums = _varied_payload("CASE-STD-DB")
    client.post("/api/v1/readings/batch", json={"readings": payload})
    db.expire_all()
    rows = (
        db.query(SegmentSnapshot)
        .filter(SegmentSnapshot.device_sn == "CASE-STD-DB")
        .order_by(SegmentSnapshot.segment_index)
        .all()
    )
    assert rows
    for row in rows:
        assert row.temperature_std is not None
        assert row.humidity_std is not None
        assert row.temperature_std >= 0.0 and row.humidity_std >= 0.0
    # 首段非恒定：两维 std 严格为正
    assert rows[0].temperature_std > 0
    assert rows[0].humidity_std > 0
    n0 = rows[0].sample_count
    assert rows[0].temperature_std == pytest.approx(_popstd(temps[:n0]), rel=1e-9)
    assert rows[0].humidity_std == pytest.approx(_popstd(hums[:n0]), rel=1e-9)


def test_constant_segment_std_is_zero_not_missing(client, db):
    payload = [
        _reading("CASE-CONST", f"2026-09-01T08:{i * 5:02d}:00+08:00", 21.5, 60.0)
        for i in range(6)
    ]
    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    seg = r.json()["summaries"][0]["replay"]["segments"][0]
    assert seg["temperature_std"] == 0.0
    assert seg["humidity_std"] == 0.0
    stored = client.get("/api/v1/cases/CASE-CONST/segments").json()["segments"][0]
    assert stored["temperature_std"] == 0.0
    assert stored["humidity_std"] == 0.0
    db.expire_all()
    row = db.query(SegmentSnapshot).filter(SegmentSnapshot.device_sn == "CASE-CONST").one()
    # 真实零值显式持久化为 0.0，而非 NULL（缺失）
    assert row.temperature_std == 0.0
    assert row.humidity_std == 0.0


# ---------- 服务重启：文件库重新初始化 ----------


def _build_legacy_db(path, sn, readings, snapshots, state_version=1, penalty=4.0):
    """在无 std 列的旧表结构上写入读数/快照，模拟升级前的旧库。"""
    eng = create_engine(f"sqlite:///{path}")
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE readings ("
                "id INTEGER PRIMARY KEY, device_sn VARCHAR(64) NOT NULL, "
                "ts DATETIME NOT NULL, temperature FLOAT NOT NULL, "
                "humidity FLOAT NOT NULL, report_count INTEGER NOT NULL, "
                "created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, "
                "UNIQUE(device_sn, ts))"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE segment_snapshots ("
                "id INTEGER PRIMARY KEY, device_sn VARCHAR(64) NOT NULL, "
                "version INTEGER NOT NULL, segment_index INTEGER NOT NULL, "
                "start_ts DATETIME NOT NULL, end_ts DATETIME NOT NULL, "
                "sample_count INTEGER NOT NULL, mean_temperature FLOAT NOT NULL, "
                "mean_humidity FLOAT NOT NULL, delta_temperature FLOAT NOT NULL, "
                "delta_humidity FLOAT NOT NULL, magnitude FLOAT NOT NULL, "
                "confidence VARCHAR(16) NOT NULL, start_anchored BOOLEAN NOT NULL, "
                "end_anchored BOOLEAN NOT NULL, cost FLOAT NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE case_states (device_sn VARCHAR(64) PRIMARY KEY, "
                "version INTEGER NOT NULL, penalty FLOAT NOT NULL, "
                "reading_count INTEGER NOT NULL, replayed_at DATETIME, extra JSON)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE anchors ("
                "id INTEGER PRIMARY KEY, device_sn VARCHAR(64) NOT NULL, "
                "ts DATETIME NOT NULL, reading_id INTEGER, label VARCHAR(128), "
                "ref_temperature FLOAT, ref_humidity FLOAT, created_at DATETIME, "
                "UNIQUE(device_sn, ts))"
            )
        )
        for i, (ts, t, h) in enumerate(readings):
            conn.execute(
                text(
                    "INSERT INTO readings (id, device_sn, ts, temperature, humidity, "
                    "report_count, created_at, updated_at) VALUES "
                    f"({i}, '{sn}', '{ts}', {t}, {h}, 1, '2026-09-26 00:00:00', "
                    "'2026-09-26 00:00:00')"
                )
            )
        for idx, seg_ in enumerate(snapshots):
            conn.execute(
                text(
                    "INSERT INTO segment_snapshots (device_sn, version, segment_index, "
                    "start_ts, end_ts, sample_count, mean_temperature, mean_humidity, "
                    "delta_temperature, delta_humidity, magnitude, confidence, "
                    "start_anchored, end_anchored, cost) VALUES "
                    f"('{sn}', {state_version}, {idx}, '{seg_['start']}', "
                    f"'{seg_['end']}', {seg_['n']}, {seg_['mt']}, {seg_['mh']}, "
                    f"{seg_['dt']}, {seg_['dh']}, {seg_['mag']}, '{seg_['conf']}', "
                    f"{int(seg_['sa'])}, {int(seg_['ea'])}, {seg_['cost']})"
                )
            )
        conn.execute(
            text(
                "INSERT INTO case_states (device_sn, version, penalty, reading_count, "
                f"replayed_at) VALUES ('{sn}', {state_version}, {penalty}, "
                f"{len(readings)}, '2026-09-26 00:00:00')"
            )
        )
    eng.dispose()


def test_service_restart_recovers_legacy_std(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    sn = "CASE-LEGACY"
    readings = [
        ("2026-09-01 00:00:00", 20.0, 55.0),
        ("2026-09-01 00:05:00", 20.4, 55.4),
        ("2026-09-01 00:10:00", 19.8, 54.6),
        ("2026-09-01 00:15:00", 24.2, 70.2),
        ("2026-09-01 00:20:00", 23.8, 69.8),
        ("2026-09-01 00:25:00", 24.0, 70.0),
    ]
    t0 = [r[1] for r in readings[:3]]
    h0 = [r[2] for r in readings[:3]]
    t1 = [r[1] for r in readings[3:]]
    h1 = [r[2] for r in readings[3:]]
    snapshots = [
        {
            "start": readings[0][0], "end": readings[2][0], "n": 3,
            "mt": sum(t0) / 3, "mh": sum(h0) / 3, "dt": 0.0, "dh": 0.0,
            "mag": 0.0, "conf": "baseline", "sa": 0, "ea": 1,
            "cost": sum((v - sum(t0) / 3) ** 2 for v in t0)
            + sum((v - sum(h0) / 3) ** 2 for v in h0),
        },
        {
            "start": readings[3][0], "end": readings[5][0], "n": 3,
            "mt": sum(t1) / 3, "mh": sum(h1) / 3,
            "dt": sum(t1) / 3 - sum(t0) / 3, "dh": sum(h1) / 3 - sum(h0) / 3,
            "mag": 1.0, "conf": "high", "sa": 1, "ea": 0,
            "cost": sum((v - sum(t1) / 3) ** 2 for v in t1)
            + sum((v - sum(h1) / 3) ** 2 for v in h1),
        },
    ]
    _build_legacy_db(db_path, sn, readings, snapshots)

    # 以该文件库启动服务（等价于容器重启跑 init_db）：列补齐 + 统计核实回补
    monkeypatch.setenv("CASE_DB_URL", f"sqlite:///{db_path}")
    from app import config, database

    monkeypatch.setattr(config.settings, "DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(database, "engine", create_engine(f"sqlite:///{db_path}"))
    monkeypatch.setattr(
        database,
        "SessionLocal",
        sessionmaker(bind=database.engine, autoflush=False, expire_on_commit=False),
    )
    database.init_db()

    # 旧快照两列已按读数补写，且分段与版本未被改动
    with Session(database.engine) as session:
        rows = (
            session.query(SegmentSnapshot)
            .filter(SegmentSnapshot.device_sn == sn)
            .order_by(SegmentSnapshot.segment_index)
            .all()
        )
        assert [r.version for r in rows] == [1, 1]
        assert [r.segment_index for r in rows] == [0, 1]
        assert rows[0].temperature_std == pytest.approx(_popstd(t0), rel=1e-9)
        assert rows[0].humidity_std == pytest.approx(_popstd(h0), rel=1e-9)
        assert rows[1].temperature_std == pytest.approx(_popstd(t1), rel=1e-9)
        assert rows[1].humidity_std == pytest.approx(_popstd(h1), rel=1e-9)

    # HTTP 查询（绕过 app 的默认内存引擎依赖）
    def _override_db():
        s = database.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides = {}
    app.dependency_overrides[get_db] = _override_db
    try:
        r = TestClient(app).get(f"/api/v1/cases/{sn}/segments")
        assert r.status_code == 200, r.text
        segs = r.json()["segments"]
        assert segs[0]["temperature_std"] == pytest.approx(_popstd(t0), rel=1e-9)
        assert segs[0]["humidity_std"] == pytest.approx(_popstd(h0), rel=1e-9)
        assert segs[1]["temperature_std"] == pytest.approx(_popstd(t1), rel=1e-9)
        assert segs[1]["humidity_std"] == pytest.approx(_popstd(h1), rel=1e-9)
        assert r.json()["version"] == 1
    finally:
        app.dependency_overrides = {}

    # 再“重启”一次：幂等，无重复迁移、无版本变化
    database.init_db()
    with Session(database.engine) as session:
        assert session.get(CaseState, sn).version == 1
    database.engine.dispose()


def test_legacy_snapshot_with_unmappable_readings_returns_409(tmp_path, monkeypatch):
    db_path = tmp_path / "broken.db"
    sn = "CASE-BROKEN"
    readings = [
        ("2026-09-01 00:00:00", 20.0, 55.0),
        ("2026-09-01 00:05:00", 20.4, 55.4),
        ("2026-09-01 00:10:00", 19.8, 54.6),
    ]
    # 快照保存的均值与现有读数不符（例如读数被改动/快照来自另一批数据）
    snapshots = [
        {
            "start": readings[0][0], "end": readings[2][0], "n": 3,
            "mt": 99.0, "mh": 10.0, "dt": 0.0, "dh": 0.0, "mag": 0.0,
            "conf": "baseline", "sa": 0, "ea": 0, "cost": 12345.0,
        }
    ]
    _build_legacy_db(db_path, sn, readings, snapshots)

    monkeypatch.setenv("CASE_DB_URL", f"sqlite:///{db_path}")
    from app import config, database

    monkeypatch.setattr(config.settings, "DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(database, "engine", create_engine(f"sqlite:///{db_path}"))
    monkeypatch.setattr(
        database,
        "SessionLocal",
        sessionmaker(bind=database.engine, autoflush=False, expire_on_commit=False),
    )
    database.init_db()  # 启动恢复失败不应阻断，列保持 NULL

    with Session(database.engine) as session:
        row = session.query(SegmentSnapshot).filter(SegmentSnapshot.device_sn == sn).one()
        assert row.temperature_std is None
        assert row.humidity_std is None

    def _override_db():
        s = database.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    from app.main import get_db

    app.dependency_overrides[get_db] = _override_db
    try:
        client_ = TestClient(app, raise_server_exceptions=False)
        r = client_.get(f"/api/v1/cases/{sn}/segments")
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert detail["device_sn"] == sn
        assert "replay" in detail["remediation"]

        # 显式重放以当前读数重建快照后恢复正常；版本递增（分段状态变化）
        r2 = client_.post(f"/api/v1/cases/{sn}/replay")
        assert r2.status_code == 200, r2.text
        rebuilt = r2.json()
        assert rebuilt["version"] == 2
        seg = rebuilt["segments"][0]
        temps = [r[1] for r in readings]
        hums = [r[2] for r in readings]
        assert seg["mean_temperature"] == pytest.approx(sum(temps) / 3, rel=1e-9)
        assert seg["temperature_std"] == pytest.approx(_popstd(temps), rel=1e-9)
        assert seg["humidity_std"] == pytest.approx(_popstd(hums), rel=1e-9)

        r3 = client_.get(f"/api/v1/cases/{sn}/segments")
        assert r3.status_code == 200
        assert r3.json()["segments"] == rebuilt["segments"]
    finally:
        app.dependency_overrides = {}
    database.engine.dispose()


# ---------- 显式重放端点 ----------


def test_explicit_replay_repairs_and_stabilizes(client, db):
    payload, temps, hums = _varied_payload("CASE-FIX")
    client.post("/api/v1/readings/batch", json={"readings": payload})
    first = client.get("/api/v1/cases/CASE-FIX/segments").json()
    version = first["version"]

    # 数据未变的显式重放：不推进版本，两维 std 与既有查询一致
    r = client.post("/api/v1/cases/CASE-FIX/replay")
    assert r.status_code == 200, r.text
    again = r.json()
    assert again["version"] == version
    assert [
        (s["temperature_std"], s["humidity_std"]) for s in again["segments"]
    ] == [(s["temperature_std"], s["humidity_std"]) for s in first["segments"]]
    assert again["replayed"] is True
    assert again["replayed_at"] is not None

    # 改变惩罚项显式重放：版本递增（分段可能变化），std 仍可由读数核实
    r2 = client.post("/api/v1/cases/CASE-FIX/replay", params={"penalty": 100.0})
    assert r2.status_code == 200
    assert r2.json()["version"] == version + 1
    assert all(s["temperature_std"] >= 0.0 for s in r2.json()["segments"])


def test_explicit_replay_unknown_case_404(client):
    r = client.post("/api/v1/cases/NOPE/replay")
    assert r.status_code == 404
