"""回归：分段快照的两维总体标准差必须随重放保存、随查询原样返回。

历史缺陷：标准差只在即时重放结果中计算，未落库；读取快照时被硬编码
为 0，导致重放后的重复查询与服务重启后的查询把非恒定段误报为零波动。
本文件覆盖：

- 重放响应、重复查询、快照落库三者标准差一致（总体标准差，非样本标准差）；
- 旧库（缺标准差列/值）启动时由读数核实恢复，恒定段为真实零、非恒定
  段为非零值，且不改动分段与版本号；
- 快照无法由当前读数核实（读数缺失、均值不符）时显式 500，绝不把
  缺失值当零值回显；
- 服务“重启”（全新引擎/会话重新打开同一文件库）后查询仍与重放一致。
"""

from __future__ import annotations

import datetime as dt
import math
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.database import (
    Base,
    CaseState,
    Reading,
    SegmentSnapshot,
    SessionLocal,
    create_engine,
    engine as default_engine,
    init_db,
    sessionmaker,
)
from app.main import app, get_db
from app.snapshot_heal import migrate_and_heal


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=default_engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=default_engine)


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _pstd(values):
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def _varying_readings(sn):
    # 前 4 点、后 4 点各自温湿度均非恒定（保证段内总体标准差 > 0），
    # 两段均值明显不同以稳定切出两段
    temps = [20.0, 20.2, 19.8, 20.1, 24.0, 24.3, 23.7, 24.1]
    hums = [50.0, 51.0, 49.0, 50.2, 60.0, 61.1, 59.0, 60.3]
    payload = [
        _reading(sn, f"2026-09-01T08:{i:02d}:00+08:00", t, h)
        for i, (t, h) in enumerate(zip(temps, hums))
    ]
    return payload, temps, hums


# ---------- 即时重放、重复查询、落库一致 ----------


def test_replayed_std_matches_repeated_query_and_snapshot(client):
    payload, temps, hums = _varying_readings("CASE-STD")
    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    assert r.status_code == 200, r.text
    replay_segs = r.json()["summaries"][0]["replay"]["segments"]
    assert len(replay_segs) == 2

    # 非恒定段标准差必须非零
    for seg in replay_segs:
        assert seg["temperature_std"] > 0
        assert seg["humidity_std"] > 0

    # 与按读数手算的总体标准差（除以 N，非 N−1）逐段一致
    expected = [
        (_pstd(temps[:4]), _pstd(hums[:4])),
        (_pstd(temps[4:]), _pstd(hums[4:])),
    ]
    for seg, (et, eh) in zip(replay_segs, expected):
        assert seg["temperature_std"] == pytest.approx(et)
        assert seg["humidity_std"] == pytest.approx(eh)

    # 重放后再次查询（走快照）：分段必须与重放完全一致
    queried = client.get("/api/v1/cases/CASE-STD/segments")
    assert queried.status_code == 200
    assert queried.json()["segments"] == replay_segs

    # 时间过滤后的分段查询同样不得回零
    filtered = client.get(
        "/api/v1/cases/CASE-STD/segments",
        params={"start": "2026-09-01T08:04:00+08:00"},
    ).json()["segments"]
    assert len(filtered) == 1
    assert filtered[0]["temperature_std"] == pytest.approx(expected[1][0])
    assert filtered[0]["humidity_std"] == pytest.approx(expected[1][1])

    # 快照表确实保存了两维标准差
    db = SessionLocal()
    try:
        db.expire_all()
        rows = (
            db.query(SegmentSnapshot)
            .filter(SegmentSnapshot.device_sn == "CASE-STD")
            .order_by(SegmentSnapshot.segment_index)
            .all()
        )
        for row, (et, eh) in zip(rows, expected):
            assert row.temperature_std == pytest.approx(et)
            assert row.humidity_std == pytest.approx(eh)
    finally:
        db.close()


def test_constant_segment_std_is_true_zero_after_replay(client):
    payload = [
        _reading("CASE-CST", f"2026-09-01T08:{i:02d}:00+08:00", 20.0, 55.0)
        for i in range(6)
    ]
    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    seg = r.json()["summaries"][0]["replay"]["segments"][0]
    assert seg["temperature_std"] == 0.0
    assert seg["humidity_std"] == 0.0
    again = client.get("/api/v1/cases/CASE-CST/segments").json()["segments"][0]
    assert again["temperature_std"] == 0.0
    assert again["humidity_std"] == 0.0


def test_std_unchanged_by_merge_and_re_replay(client):
    # 重复上报合并后再次重放，重复查询仍与最近一次重放一致
    payload, _, _ = _varying_readings("CASE-MRG")
    client.post("/api/v1/readings/batch", json={"readings": payload})
    again = [_reading("CASE-MRG", "2026-09-01T08:00:00+08:00", 20.5, 50.5)]
    r = client.post("/api/v1/readings/batch", json={"readings": again})
    replay_segs = r.json()["summaries"][0]["replay"]["segments"]
    queried = client.get("/api/v1/cases/CASE-MRG/segments").json()["segments"]
    assert queried == replay_segs


# ---------- 服务“重启”：全新引擎重新打开同一文件库 ----------


def _file_db_stack(db_path):
    """对同一 SQLite 文件建立一套全新的 engine/session（模拟重启）。"""
    url = f"sqlite:///{db_path.as_posix()}"
    engine = create_engine(url)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def _get_db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    return engine, Session


def test_std_survives_restart_with_file_db(tmp_path):
    db_path = tmp_path / "restart.db"
    payload, temps, hums = _varying_readings("CASE-RST")

    # 阶段 1：全新文件库，经 API 重放落库
    engine_a, SessionA = _file_db_stack(db_path)
    Base.metadata.drop_all(bind=engine_a)
    Base.metadata.create_all(bind=engine_a)
    migrate_and_heal(engine_a)
    try:
        with TestClient(app) as c:
            r = c.post("/api/v1/readings/batch", json={"readings": payload})
            replay_segs = r.json()["summaries"][0]["replay"]["segments"]
        # 显式释放文件句柄（SQLite 单连接池）
        engine_a.dispose()
    finally:
        app.dependency_overrides.pop(get_db, None)

    expected = [
        (_pstd(temps[:4]), _pstd(hums[:4])),
        (_pstd(temps[4:]), _pstd(hums[4:])),
    ]

    # 阶段 2：“重启”——全新引擎打开同一文件，init_db 不得改动数据，
    # 不做任何重放直接查询，标准差必须与阶段 1 的重放完全一致
    engine_b, _SessionB = _file_db_stack(db_path)
    try:
        init_db_via_engine(engine_b)
        with TestClient(app) as c:
            r = c.get("/api/v1/cases/CASE-RST/segments")
            assert r.status_code == 200, r.text
            segs = r.json()["segments"]
            assert len(segs) == 2
            for seg, (et, eh) in zip(segs, expected):
                assert seg["temperature_std"] == pytest.approx(et)
                assert seg["humidity_std"] == pytest.approx(eh)
            # 版本号、分段边界与重放时相同
            assert r.json()["version"] == 1
            assert [(s["start_ts"], s["end_ts"], s["sample_count"]) for s in segs] == [
                (s["start_ts"], s["end_ts"], s["sample_count"]) for s in replay_segs
            ]
    finally:
        engine_b.dispose()
        app.dependency_overrides.pop(get_db, None)


def init_db_via_engine(engine):
    """在给定文件库上执行幂等迁移（init_db 默认绑定全局 engine）。"""
    Base.metadata.create_all(bind=engine)
    migrate_and_heal(engine)


# ---------- 旧库快照（缺标准差列/值）的核实恢复 ----------


def _seed_legacy_snapshot(Session, sn, temps, hums, *, version=1, n=8):
    """按旧版本行为写入读数与快照：快照没有标准差。"""
    engine = Session.kw["bind"]
    Base.metadata.create_all(bind=engine)
    db = Session()
    base = dt.datetime(2026, 9, 1, 0, 0, 0, tzinfo=dt.timezone.utc).replace(tzinfo=None)
    for i in range(n):
        db.add(
            Reading(
                device_sn=sn,
                ts=base + dt.timedelta(minutes=10 * i),
                temperature=temps[i],
                humidity=hums[i],
                report_count=1,
            )
        )
    half = n // 2

    def mean(xs):
        return sum(xs) / len(xs)

    db.add(
        CaseState(
            device_sn=sn, version=version, penalty=4.0, reading_count=n,
            replayed_at=base,
        )
    )
    db.add(SegmentSnapshot(
        device_sn=sn, version=version, segment_index=0,
        start_ts=base, end_ts=base + dt.timedelta(minutes=10 * (half - 1)),
        sample_count=half,
        mean_temperature=mean(temps[:half]), mean_humidity=mean(hums[:half]),
        delta_temperature=0.0, delta_humidity=0.0, magnitude=0.0,
        confidence="baseline", start_anchored=False, end_anchored=False, cost=0.0,
    ))
    db.add(SegmentSnapshot(
        device_sn=sn, version=version, segment_index=1,
        start_ts=base + dt.timedelta(minutes=10 * half),
        end_ts=base + dt.timedelta(minutes=10 * (n - 1)),
        sample_count=n - half,
        mean_temperature=mean(temps[half:]), mean_humidity=mean(hums[half:]),
        delta_temperature=mean(temps[half:]) - mean(temps[:half]),
        delta_humidity=mean(hums[half:]) - mean(hums[:half]),
        magnitude=1.5, confidence="high",
        start_anchored=False, end_anchored=False, cost=0.0,
    ))
    db.commit()
    db.close()


def _seed_legacy_constant(engine, Session, sn, temps, hums):
    """恒为 7 点单段的真实旧库形态（与 data/cases.db 一致）。"""
    Base.metadata.create_all(bind=engine)
    db = Session()
    base = dt.datetime(2026, 9, 1, 0, 10, 0)
    for i in range(7):
        db.add(Reading(device_sn=sn, ts=base + dt.timedelta(minutes=10 * i),
                       temperature=temps[i], humidity=hums[i], report_count=1))
    db.add(CaseState(device_sn=sn, version=1, penalty=4.0, reading_count=7,
                     replayed_at=base))
    db.add(SegmentSnapshot(
        device_sn=sn, version=1, segment_index=0,
        start_ts=base, end_ts=base + dt.timedelta(minutes=60),
        sample_count=7, mean_temperature=20.0, mean_humidity=50.0,
        delta_temperature=0.0, delta_humidity=0.0, magnitude=0.0,
        confidence="baseline", start_anchored=False, end_anchored=False, cost=0.0,
    ))
    db.commit()
    db.close()
    # 回到旧 schema：丢掉新列（SQLite 需重建表）
    con = sqlite3.connect(db_path_of(engine))
    con.executescript(
        """
        CREATE TABLE segment_snapshots_old AS
            SELECT id, device_sn, version, segment_index, start_ts, end_ts,
                   sample_count, mean_temperature, mean_humidity,
                   delta_temperature, delta_humidity, magnitude, confidence,
                   start_anchored, end_anchored, cost
            FROM segment_snapshots;
        DROP TABLE segment_snapshots;
        CREATE TABLE segment_snapshots (
            id INTEGER PRIMARY KEY, device_sn VARCHAR(64) NOT NULL,
            version INTEGER NOT NULL, segment_index INTEGER NOT NULL,
            start_ts DATETIME NOT NULL, end_ts DATETIME NOT NULL,
            sample_count INTEGER NOT NULL,
            mean_temperature FLOAT NOT NULL, mean_humidity FLOAT NOT NULL,
            delta_temperature FLOAT NOT NULL, delta_humidity FLOAT NOT NULL,
            magnitude FLOAT NOT NULL, confidence VARCHAR(16) NOT NULL,
            start_anchored BOOLEAN NOT NULL, end_anchored BOOLEAN NOT NULL,
            cost FLOAT NOT NULL);
        INSERT INTO segment_snapshots SELECT * FROM segment_snapshots_old;
        DROP TABLE segment_snapshots_old;
        """
    )
    con.commit()
    con.close()


def db_path_of(engine):
    return engine.url.database


def test_legacy_nonconstant_snapshot_recovered_via_api(tmp_path):
    db_path = tmp_path / "legacy_nonconst.db"
    url = f"sqlite:///{db_path.as_posix()}"
    engine = create_engine(url)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    temps = [20.0, 20.2, 19.8, 20.1, 24.0, 24.3, 23.7, 24.1]
    hums = [50.0, 51.0, 49.0, 50.2, 60.0, 61.1, 59.0, 60.3]
    _seed_legacy_snapshot(Session, "CASE-LNC", temps, hums)

    def _get_db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    try:
        # 旧行 std 为 NULL（ORM 新建表已含列但行值为 None）
        migrate_and_heal(engine)
        with TestClient(app) as c:
            r = c.get("/api/v1/cases/CASE-LNC/segments")
            assert r.status_code == 200, r.text
            segs = r.json()["segments"]
            assert segs[0]["temperature_std"] == pytest.approx(_pstd(temps[:4]))
            assert segs[0]["humidity_std"] == pytest.approx(_pstd(hums[:4]))
            assert segs[1]["temperature_std"] == pytest.approx(_pstd(temps[4:]))
            assert segs[1]["humidity_std"] == pytest.approx(_pstd(hums[4:]))
            assert segs[0]["temperature_std"] > 0
            assert r.json()["version"] == 1  # 恢复不改版本
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def test_legacy_missing_column_added_and_constant_healed(tmp_path):
    """真实形态：旧库表完全没有 std 列（ALTER TABLE 补列 + 恢复）。"""
    db_path = tmp_path / "legacy_alter.db"
    url = f"sqlite:///{db_path.as_posix()}"
    engine = create_engine(url)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    _seed_legacy_constant(engine, Session, "CASE-E2E", [20.0] * 7, [50.0] * 7)
    engine.dispose()

    engine = create_engine(url)
    migrate_and_heal(engine)
    con = sqlite3.connect(db_path)
    cols = {r[1] for r in con.execute("PRAGMA table_info(segment_snapshots)")}
    assert {"temperature_std", "humidity_std"} <= cols
    val = con.execute("select temperature_std, humidity_std from segment_snapshots").fetchone()
    con.close()
    assert val == (0.0, 0.0)
    engine.dispose()


# ---------- 无法核实的快照：显式报错，不用零值冒充 ----------


def test_unverifiable_snapshot_missing_reading_returns_500(tmp_path):
    db_path = tmp_path / "broken.db"
    url = f"sqlite:///{db_path.as_posix()}"
    engine = create_engine(url)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    temps = [20.0, 20.2, 19.8, 20.1, 24.0, 24.3, 23.7, 24.1]
    hums = [50.0, 51.0, 49.0, 50.2, 60.0, 61.1, 59.0, 60.3]
    _seed_legacy_snapshot(Session, "CASE-BRK", temps, hums)

    # 删掉首条读数并把 std 置空：快照无法由剩余读数核实
    con = sqlite3.connect(db_path)
    con.execute(
        "update segment_snapshots set temperature_std=NULL, humidity_std=NULL"
    )
    con.execute("delete from readings where ts=(select min(ts) from readings)")
    con.commit()
    con.close()

    migrate_and_heal(engine)  # 启动恢复失败不得拖垮服务启动

    def _get_db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    try:
        with TestClient(app) as c:
            r = c.get("/api/v1/cases/CASE-BRK/segments")
            assert r.status_code == 500
            detail = r.json()["detail"]
            assert detail["device_sn"] == "CASE-BRK"
            assert detail["version"] == 1
            assert detail["segment_index"] == 0
            assert detail["reason"] in {
                "sample_count_mismatch",
                "boundary_reading_missing",
            }
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def test_unverifiable_snapshot_mean_mismatch_returns_500(client):
    # 内存库：写一条 std 为 NULL、但均值与读数对不上的快照
    payload, _, _ = _varying_readings("CASE-MM")
    client.post("/api/v1/readings/batch", json={"readings": payload})
    db = SessionLocal()
    try:
        for row in db.query(SegmentSnapshot).filter(SegmentSnapshot.device_sn == "CASE-MM"):
            row.temperature_std = None
            row.humidity_std = None
            row.mean_temperature = -99.0  # 与读数均值不符
        db.commit()
    finally:
        db.close()

    r = client.get("/api/v1/cases/CASE-MM/segments")
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert detail["reason"] == "mean_mismatch"
    assert detail["device_sn"] == "CASE-MM"
