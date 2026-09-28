"""采集质量审计：邻点中位数/MAD 孤立异常值（接口集成测试）。"""

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


def _flat(sn, n, t=20.0, h=50.0, minute_step=10, start_day=1):
    out = []
    for i in range(n):
        total = i * minute_step
        hh, mm = divmod(total, 60)
        out.append(
            _reading(sn, f"2026-09-{start_day:02d}T{hh:02d}:{mm:02d}:00Z", t, h)
        )
    return out


PARAMS = {
    "window_size": 5,
    "k": 3.0,
    "min_temperature_delta": 0.5,
    "min_humidity_delta": 1.0,
    "max_gap_seconds": 600,
}


# ---------- 基本命中 ----------


def test_spike_flagged_with_median_mad_and_sorted(client):
    payload = _flat("CASE-O1", 13)
    payload[6]["temperature"] = 25.0   # +5 的温度尖峰
    payload[9]["humidity"] = 65.0      # +15 的湿度尖峰
    client.post("/api/v1/readings/batch", json={"readings": payload})

    r = client.get("/api/v1/cases/CASE-O1/reading-outliers", params=PARAMS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_count"] == 13
    assert body["segment_count"] == 1
    # 段首、段尾各 2 点因一侧不足 2 邻点不判
    assert body["evaluated_count"] == 9
    assert body["outlier_count"] == 2

    assert [o["ts"] for o in body["outliers"]] == sorted(o["ts"] for o in body["outliers"])

    first = body["outliers"][0]
    assert first["ts"] == "2026-09-01T01:00:00Z"
    assert first["temperature"] == 25.0
    assert first["humidity"] == 50.0
    dim = first["dimensions"][0]
    assert dim["dimension"] == "temperature"
    assert dim["median"] == pytest.approx(20.0)
    assert dim["deviation"] == pytest.approx(5.0)

    second = body["outliers"][1]
    assert second["ts"] == "2026-09-01T01:30:00Z"
    assert second["dimensions"][0]["dimension"] == "humidity"
    assert second["dimensions"][0]["median"] == pytest.approx(50.0)
    assert second["dimensions"][0]["deviation"] == pytest.approx(15.0)

def test_both_dimensions_flagged_together_and_ordered(client):
    payload = _flat("CASE-O2", 9)
    payload[4]["temperature"] = 30.0
    payload[4]["humidity"] = 80.0
    client.post("/api/v1/readings/batch", json={"readings": payload})

    body = client.get(
        "/api/v1/cases/CASE-O2/reading-outliers", params=PARAMS
    ).json()
    assert body["outlier_count"] == 1
    out = body["outliers"][0]
    assert [d["dimension"] for d in out["dimensions"]] == [
        "temperature",
        "humidity",
    ]


def test_normal_variation_not_flagged(client):
    # ±0.1 的抖动：邻点 MAD≈0，尺度由 min_delta 兜底，偏差 0.1 不超 3×0.5
    payload = _flat("CASE-O3", 11)
    for i, p in enumerate(payload):
        p["temperature"] = 20.0 + (0.1 if i % 2 else -0.1)
        p["humidity"] = 50.0 + (0.2 if i % 2 else -0.2)
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = client.get(
        "/api/v1/cases/CASE-O3/reading-outliers", params=PARAMS
    ).json()
    assert body["outliers"] == []
    assert body["evaluated_count"] == 7


# ---------- 窗口大小 ----------


def test_window_limits_neighbors(client):
    # 21 点序列：目标点 i=10 为 20；其左右最近 2 个邻点（i=8,9,11,12）
    # 也是 20；其余 16 点为 30。
    payload = _flat("CASE-O4", 21)
    for idx in (*range(0, 8), *range(13, 21)):
        payload[idx]["temperature"] = 30.0
    client.post("/api/v1/readings/batch", json={"readings": payload})

    # window_size=2：目标点的 4 个邻点全是 20，不命中
    small = client.get(
        "/api/v1/cases/CASE-O4/reading-outliers",
        params={**PARAMS, "window_size": 2},
    ).json()
    assert not any(o["ts"] == "2026-09-01T01:40:00Z" for o in small["outliers"])

    # window_size=10：邻点含 16 个 30（MAD=0），中位数翻为 30，目标 20 命中
    large = client.get(
        "/api/v1/cases/CASE-O4/reading-outliers",
        params={**PARAMS, "window_size": 10, "min_temperature_delta": 0.1},
    ).json()
    hit = [o for o in large["outliers"] if o["ts"] == "2026-09-01T01:40:00Z"]
    assert len(hit) == 1
    assert hit[0]["dimensions"][0]["median"] == pytest.approx(30.0)
    assert hit[0]["dimensions"][0]["deviation"] == pytest.approx(10.0)


def test_window_expansion_pulls_in_far_spike(client):
    # 13 点：目标 i=6 为 20；左右最近 2 邻点（i=4,5,7,8）为 20；
    # 更外侧 i=1,2,3 与 i=9,10,11 共 6 点为 30。
    payload = _flat("CASE-O4B", 13)
    for idx in (1, 2, 3, 9, 10, 11):
        payload[idx]["temperature"] = 30.0
    client.post("/api/v1/readings/batch", json={"readings": payload})

    # window=2：4 个邻点全是 20，目标不命中
    small = client.get(
        "/api/v1/cases/CASE-O4B/reading-outliers",
        params={**PARAMS, "window_size": 2, "min_temperature_delta": 0.1},
    ).json()
    assert not any(o["ts"] == "2026-09-01T01:00:00Z" for o in small["outliers"])

    # window=5：10 个邻点中 6 个 30 → 中位数 30、MAD 0，目标命中
    large = client.get(
        "/api/v1/cases/CASE-O4B/reading-outliers",
        params={**PARAMS, "window_size": 5, "min_temperature_delta": 0.1},
    ).json()
    hit = [o for o in large["outliers"] if o["ts"] == "2026-09-01T01:00:00Z"]
    assert len(hit) == 1
    assert hit[0]["dimensions"][0]["median"] == pytest.approx(30.0)
    assert hit[0]["dimensions"][0]["deviation"] == pytest.approx(10.0)


# ---------- 断档切段 ----------


def test_gap_severs_sequence(client):
    # 段 A：7 点（09-01 00:00…01:00，10 分钟间隔），尖峰在中段
    payload = _flat("CASE-O5", 7)
    payload[3]["temperature"] = 30.0
    # 间隔约 23 小时后的段 B：3 点（不足 5 点，无点可判）
    payload += _flat("CASE-O5", 3, start_day=2)
    client.post("/api/v1/readings/batch", json={"readings": payload})

    body = client.get(
        "/api/v1/cases/CASE-O5/reading-outliers", params=PARAMS
    ).json()
    assert body["segment_count"] == 2
    assert body["sample_count"] == 10
    # 段 A 7 点：可判的为 i=2..4 共 3 个；段 B 3 点：0 个
    assert body["evaluated_count"] == 3
    assert body["outlier_count"] == 1
    assert body["outliers"][0]["ts"] == "2026-09-01T00:30:00Z"


def test_gap_equal_is_continuous(client):
    # 间隔恰好 600 秒不断开
    payload = _flat("CASE-O6", 5, minute_step=10)
    payload[2]["temperature"] = 30.0
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = client.get(
        "/api/v1/cases/CASE-O6/reading-outliers",
        params={**PARAMS, "max_gap_seconds": 600},
    ).json()
    assert body["segment_count"] == 1
    assert body["evaluated_count"] == 1  # 仅中点两侧各 ≥2
    assert body["outlier_count"] == 1


def test_segment_edge_points_skipped(client):
    # 每段只有 4 点：i=0,1,3 任一侧不足 2，仅 i=2（左2右1）也不判 → 0
    payload = _flat("CASE-O7", 4)
    payload[0]["temperature"] = 50.0
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = client.get(
        "/api/v1/cases/CASE-O7/reading-outliers", params=PARAMS
    ).json()
    assert body["evaluated_count"] == 0
    assert body["outliers"] == []


# ---------- 严格大于与 min_delta 兜底 ----------


def test_strict_greater_than_boundary(client):
    # 邻点全为 20：MAD=0。min_temp_delta=2、k=3 → 阈值 6；
    # 26.0 恰好等于阈值不命中（严格大于），26.001 命中；
    # min_temp_delta=1 时阈值 3：22.999 不命中、23.001 命中。
    cases = (
        ("CASE-O8A", 26.0, 2.0, False),
        ("CASE-O8B", 26.001, 2.0, True),
        ("CASE-O8C", 22.999, 1.0, False),
        ("CASE-O8D", 23.001, 1.0, True),
    )
    for sn, value, min_delta, expect in cases:
        payload = _flat(sn, 5)
        payload[2]["temperature"] = value
        client.post("/api/v1/readings/batch", json={"readings": payload})
        body = client.get(
            f"/api/v1/cases/{sn}/reading-outliers",
            params={**PARAMS, "min_temperature_delta": min_delta},
        ).json()
        assert (body["outlier_count"] == 1) is expect, (sn, body)


def test_min_delta_floors_scale_when_mad_zero(client):
    # 邻点全 20，尖峰 22.5：偏差 2.5。
    # min_temp_delta=1 → 阈值 3，不命中；min_temp_delta=0.5 → 阈值 1.5，命中
    for sn, min_delta, expect in (
        ("CASE-O9A", 1.0, False),
        ("CASE-O9B", 0.5, True),
    ):
        payload = _flat(sn, 5)
        payload[2]["temperature"] = 22.5
        client.post("/api/v1/readings/batch", json={"readings": payload})
        body = client.get(
            f"/api/v1/cases/{sn}/reading-outliers",
            params={**PARAMS, "min_temperature_delta": min_delta},
        ).json()
        assert (body["outlier_count"] == 1) is expect


# ---------- 404 / 422 ----------


def test_unknown_case_404(client):
    r = client.get(
        "/api/v1/cases/NOPE/reading-outliers",
        params={
            "window_size": 5,
            "k": 3.0,
            "min_temperature_delta": 0.5,
            "min_humidity_delta": 1.0,
        },
    )
    assert r.status_code == 404


def test_validation_422(client):
    client.post("/api/v1/readings/batch", json={"readings": _flat("CASE-O10", 5)})
    base = (
        "/api/v1/cases/CASE-O10/reading-outliers"
        "?k=3&min_temperature_delta=0.5&min_humidity_delta=1"
    )
    assert client.get(base).status_code == 422  # 缺 window_size
    assert client.get(base + "&window_size=1").status_code == 422
    assert client.get(base + "&window_size=51").status_code == 422
    assert client.get(base + "&window_size=5&k=0").status_code == 422
    assert client.get(base + "&window_size=5&k=-1").status_code == 422
    assert client.get(
        base + "&window_size=5&min_temperature_delta=0"
    ).status_code == 422
    assert client.get(
        base + "&window_size=5&min_humidity_delta=-1"
    ).status_code == 422
    assert client.get(
        base + "&window_size=5&max_gap_seconds=0"
    ).status_code == 422
    assert client.get(
        base + "&window_size=5&max_gap_seconds=-600"
    ).status_code == 422


# ---------- 只读 & 确定性 ----------


def test_read_only_does_not_mutate(client, db):
    payload = _flat("CASE-O11", 7)
    payload[3]["temperature"] = 30.0
    client.post("/api/v1/readings/batch", json={"readings": payload})

    state = db.get(CaseState, "CASE-O11")
    version_before = state.version
    readings_before = [
        (r.ts, r.temperature, r.humidity, r.report_count)
        for r in db.query(Reading).filter(Reading.device_sn == "CASE-O11").order_by(Reading.ts)
    ]
    snaps_before = db.query(SegmentSnapshot).filter(
        SegmentSnapshot.device_sn == "CASE-O11"
    ).count()

    params = {
        "window_size": 3,
        "k": 3.0,
        "min_temperature_delta": 0.5,
        "min_humidity_delta": 1.0,
    }
    r1 = client.get("/api/v1/cases/CASE-O11/reading-outliers", params=params).json()
    r2 = client.get("/api/v1/cases/CASE-O11/reading-outliers", params=params).json()
    assert r1 == r2

    db.expire_all()
    assert db.get(CaseState, "CASE-O11").version == version_before
    readings_after = [
        (r.ts, r.temperature, r.humidity, r.report_count)
        for r in db.query(Reading).filter(Reading.device_sn == "CASE-O11").order_by(Reading.ts)
    ]
    assert readings_after == readings_before
    assert (
        db.query(SegmentSnapshot).filter(
            SegmentSnapshot.device_sn == "CASE-O11"
        ).count()
        == snaps_before
    )


def test_merged_readings_are_used(client, db):
    # 同 ts 重复上报先被合并，审计基于合并后的均值：21.0 不是尖峰
    client.post("/api/v1/readings/batch", json={"readings": _flat("CASE-O12", 5)})
    client.post("/api/v1/readings/batch", json={"readings": [
        _reading("CASE-O12", "2026-09-01T00:20:00Z", 22.0, 50.0)
    ]})
    db.expire_all()
    row = (
        db.query(Reading)
        .filter(Reading.device_sn == "CASE-O12", Reading.ts.like("%00:20:%"))
        .one()
    )
    assert row.temperature == pytest.approx(21.0)

    body = client.get(
        "/api/v1/cases/CASE-O12/reading-outliers", params=PARAMS
    ).json()
    assert body["sample_count"] == 5
    assert body["outliers"] == []
