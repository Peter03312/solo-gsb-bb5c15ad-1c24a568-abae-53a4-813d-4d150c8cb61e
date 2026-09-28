"""跨展柜共模突变：候选窗枚举/全局选择单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine, init_db
from app.incidents import (
    Boundary,
    detect_incidents,
    enumerate_incidents,
    select_incidents,
    boundary_direction,
)
from app.main import app

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def _b(device: str, minutes: float, magnitude: float, ts: dt.datetime | None = None) -> Boundary:
    return Boundary(
        device_sn=device,
        ts=ts if ts is not None else BASE + dt.timedelta(minutes=minutes),
        magnitude=magnitude,
        delta_temperature=magnitude * 2.0,
        delta_humidity=0.0,
    )


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _step_payload(sn, jump_minute: int, n_before: int = 6, n_after: int = 6,
                  t0: float = 20.0, t1: float = 24.0, h0: float = 55.0, h1: float = 70.0):
    """08:00 起每分钟一点，在 jump_minute 处发生温湿双跳变。"""
    payload = []
    for i in range(n_before):
        payload.append(_reading(sn, f"2026-09-01T08:{i:02d}:00+08:00", t0, h0))
    for i in range(n_after):
        m = jump_minute + i
        payload.append(_reading(sn, f"2026-09-01T08:{m:02d}:00+08:00", t1, h1))
    return payload


# ---------- 纯算法：候选窗枚举 ----------


def test_enumerate_basic_window():
    boundaries = [_b("A", 0, 1.0), _b("B", 0.5, 1.0), _b("C", 1.0, 1.0)]
    events = enumerate_incidents(boundaries, tolerance_seconds=60.0, min_devices=2)
    # 连续窗 [A,B]、[B,C]、[A,B,C]（A、C 不相邻，无 [A,C]）；跨度恰好 60s 合法
    assert {frozenset(b.device_sn for b in e.members) for e in events} == {
        frozenset({"A", "B"}),
        frozenset({"B", "C"}),
        frozenset({"A", "B", "C"}),
    }
    full = next(e for e in events if e.device_count == 3)
    assert full.start_ts == BASE
    assert full.duration_seconds == pytest.approx(60.0)
    assert full.total_magnitude == pytest.approx(3.0)


def test_tolerance_boundary_is_inclusive():
    boundaries = [_b("A", 0, 1.0), _b("B", 1, 1.0)]
    assert enumerate_incidents(boundaries, 60.0, 2)  # 恰好等于容差
    assert not enumerate_incidents(boundaries, 59.999, 2)


def test_same_device_keeps_largest_magnitude_then_earliest():
    # 同一展柜窗内两条边界：保留幅度更大者
    boundaries = [
        _b("A", 0, 1.0),
        _b("A", 0.5, 3.0),
        _b("B", 0.8, 1.0),
    ]
    events = enumerate_incidents(boundaries, 60.0, 2)
    full = next(e for e in events if e.device_count == 2)
    assert {b.device_sn for b in full.members} == {"A", "B"}
    a_member = next(b for b in full.members if b.device_sn == "A")
    assert a_member.magnitude == 3.0

    # 幅度并列：取更早者
    boundaries_tie = [
        _b("A", 0, 2.0),
        _b("A", 0.5, 2.0),
        _b("B", 0.8, 1.0),
    ]
    events_tie = enumerate_incidents(boundaries_tie, 60.0, 2)
    full_tie = next(e for e in events_tie if e.device_count == 2)
    a_tie = next(b for b in full_tie.members if b.device_sn == "A")
    assert a_tie.ts == BASE  # 更早的一条


def test_min_devices_rejects_single_device_windows():
    boundaries = [_b("A", 0, 1.0), _b("A", 0.5, 2.0), _b("B", 10, 1.0)]
    assert not enumerate_incidents(boundaries, 60.0, 2)


def test_members_sorted_by_ts_then_device():
    boundaries = [_b("B", 0.5, 1.0), _b("A", 0.5, 1.0), _b("C", 0, 1.0)]
    events = enumerate_incidents(boundaries, 60.0, 3)
    full = next(e for e in events if e.device_count == 3)
    keys = [(b.ts, b.device_sn) for b in full.members]
    assert keys == sorted(keys)


# ---------- 纯算法：全局选择（杜绝贪心链式归并） ----------


def test_select_disjoint_both_chosen():
    left = next(
        e
        for e in enumerate_incidents(
            [_b("A", 0, 1.0), _b("B", 0.5, 1.0)], 60.0, 2
        )
        if e.device_count == 2
    )
    right = next(
        e
        for e in enumerate_incidents(
            [_b("C", 5, 1.0), _b("D", 5.5, 1.0)], 60.0, 2
        )
        if e.device_count == 2
    )
    picked = select_incidents([left, right])
    assert picked == sorted(picked, key=lambda e: e.start_ts)
    assert [e.device_count for e in picked] == [2, 2]


def test_select_overlapping_prefers_more_devices():
    # 窄窗 [A,B]（2 设备）完全落在宽窗 [A,B,C]（3 设备）内
    events = enumerate_incidents(
        [_b("A", 0, 1.0), _b("B", 0.2, 1.0), _b("C", 0.5, 1.0)],
        60.0,
        2,
    )
    picked = select_incidents(events)
    assert len(picked) == 1
    assert picked[0].device_count == 3


def test_select_global_optimum_not_greedy_chain():
    # 边界：5 个展柜挤在中部（C0..C4），两侧各 2 个展柜（A/B 与 E/F）。
    # 中部任意连续窗的 distinct device 始终为 5；容差 300s。
    # - 贪心链式归并从中部连续出发：任何 5 设备窗口要么与 A/B 簇重叠、
    #   要么与 E/F 簇重叠，最终 A/B、E/F 都会被并进同一个大事件
    #   （或因跨度截断而丢弃），最多得到 1~2 个事件；
    # - 全局最优把三簇分开，所有 9 个展柜边界各入一个互不重叠事件，
    #   设备数总和为 9。
    boundaries = (
        [_b("A", 0, 1.0), _b("B", 0.5, 1.0)]
        + [_b(f"C{i}", 2 + i * 0.1, 1.0) for i in range(5)]
        + [_b("E", 7, 1.0), _b("F", 7.5, 1.0)]
    )
    candidates = enumerate_incidents(boundaries, 300.0, 2)
    picked = select_incidents(candidates)
    assert sum(e.device_count for e in picked) == 9
    # 首尾两簇必须独立成事件，未被链式归并吞没
    first, last = picked[0], picked[-1]
    assert {b.device_sn for b in first.members} == {"A", "B"}
    assert {b.device_sn for b in last.members} == {"E", "F"}
    # 中部 5 个展柜的边界也都入选（可拆成多个更早的互不重叠事件；
    # 起始序列并列时更多事件优先，仍不归并）
    middle_devices = {
        b.device_sn
        for e in picked[1:-1]
        for b in e.members
    }
    assert middle_devices == {f"C{i}" for i in range(5)}
    # 严格互不重叠
    for prev, nxt in zip(picked, picked[1:]):
        assert nxt.start_ts > prev.end_ts


def test_select_prefers_total_magnitude_then_start_sequence():
    # 两个时间完全分离、设备数相同但幅度不同的候选：幅度总和大者胜
    hi = next(
        e
        for e in enumerate_incidents(
            [_b("A", 0, 3.0), _b("B", 0.2, 3.0)], 60.0, 2
        )
        if e.device_count == 2
    )
    lo = next(
        e
        for e in enumerate_incidents(
            [_b("C", 5, 1.0), _b("D", 5.2, 1.0)], 60.0, 2
        )
        if e.device_count == 2
    )
    picked = select_incidents([hi, lo])
    assert len(picked) == 2  # 互不重叠 → 两个都选，幅度总和 8
    assert sum(e.total_magnitude for e in picked) == pytest.approx(8.0)

    # 同区间互相重叠的两个候选，设备数相同：幅度更大者胜
    big = next(
        e
        for e in enumerate_incidents(
            [_b("A", 0, 3.0), _b("B", 0.2, 3.0), _b("C", 0.4, 1.0)],
            60.0,
            2,
        )
        if {b.device_sn for b in e.members} == {"A", "B"}
        and e.device_count == 2
    )
    small = next(
        e
        for e in enumerate_incidents(
            [_b("A", 0, 3.0), _b("B", 0.2, 3.0), _b("C", 0.4, 1.0)],
            60.0,
            2,
        )
        if {b.device_sn for b in e.members} == {"B", "C"}
        and e.device_count == 2
    )
    picked = select_incidents([big, small])
    assert picked == [big]


def test_same_timestamp_cannot_split_into_two_events():
    # A、B、C 三条边界同一时刻：只能是一个事件，不能拆成 [A,B] 与 [C,...]
    t = BASE
    boundaries = [
        _b("A", 0, 1.0, ts=t),
        _b("B", 0, 1.0, ts=t),
        _b("C", 0, 1.0, ts=t + dt.timedelta(minutes=10)),
        _b("D", 0, 1.0, ts=t + dt.timedelta(minutes=10)),
    ]
    candidates = enumerate_incidents(boundaries, 59.0, 2)
    picked = select_incidents(candidates)
    # 容差 59s 下同刻的两簇各为 2 设备，但 start_ts==end_ts 相同 → 视为重叠
    # 两簇间隔 10 分钟 > 59s，其实是两个独立同刻簇，互不冲突，都入选
    assert [e.device_count for e in picked] == [2, 2]

    candidates_zero = enumerate_incidents(
        [
            _b("A", 0, 1.0, ts=t),
            _b("B", 0, 1.0, ts=t),
            _b("C", 0, 1.0, ts=t),
        ],
        10.0,
        2,
    )
    picked_zero = select_incidents(candidates_zero)
    assert len(picked_zero) == 1
    assert picked_zero[0].device_count == 3


def test_detect_empty():
    assert detect_incidents([], 60.0, 2) == []


def test_detect_deterministic_regardless_of_input_order():
    boundaries = [
        _b("F", 7.5, 1.0),
        _b("C4", 2.4, 1.0),
        _b("A", 0, 1.0),
        _b("C0", 2.0, 1.0),
        _b("E", 7, 1.0),
        _b("B", 0.5, 1.0),
        _b("C2", 2.2, 1.0),
        _b("C1", 2.1, 1.0),
        _b("C3", 2.3, 1.0),
    ]
    first = detect_incidents(boundaries, 300.0, 2)
    second = detect_incidents(list(reversed(boundaries)), 300.0, 2)
    keys = [(e.start_ts, e.end_ts, e.event_key) for e in first]
    assert keys == [(e.start_ts, e.end_ts, e.event_key) for e in second]
    # 结果按开始时间升序
    assert [e.start_ts for e in first] == sorted(e.start_ts for e in first)


def test_boundary_direction():
    assert boundary_direction(2.0, 0.0) == "up"
    assert boundary_direction(-2.0, 0.0) == "down"
    assert boundary_direction(0.0, 5.0) == "up"
    assert boundary_direction(0.0, -5.0) == "down"
    assert boundary_direction(0.0, 0.0) == "flat"
    # 温湿并列取温度
    assert boundary_direction(2.0, 5.0) == "up"  # temp (2/2)^2==1 == hum (5/5)^2==1


# ---------- 接口集成 ----------


def test_incidents_api_common_mode_event(client):
    for sn, jump in (("CASE-A", 6), ("CASE-B", 7), ("CASE-C", 8)):
        r = client.post(
            "/api/v1/readings/batch",
            json={"readings": _step_payload(sn, jump)},
        )
        assert r.status_code == 200, r.text

    r = client.get(
        "/api/v1/incidents",
        params={
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-09-01T02:00:00Z",
            "tolerance_seconds": 600,
            "min_devices": 2,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tolerance_seconds"] == 600.0
    assert body["min_devices"] == 2
    assert body["boundary_count"] == 3
    assert body["candidate_count"] >= 1
    assert body["incident_count"] == len(body["incidents"]) == 1

    event = body["incidents"][0]
    assert event["device_count"] == 3
    assert event["start_ts"] == "2026-09-01T00:06:00Z"
    assert event["end_ts"] == "2026-09-01T00:08:00Z"
    assert event["duration_seconds"] == 120.0
    assert len(event["boundaries"]) == 3
    assert [b["device_sn"] for b in event["boundaries"]] == ["CASE-A", "CASE-B", "CASE-C"]
    for b in event["boundaries"]:
        assert b["direction"] == "up"
        assert b["magnitude"] > 0
    assert event["total_magnitude"] == pytest.approx(
        sum(b["magnitude"] for b in event["boundaries"])
    )


def test_incidents_api_two_separate_events(client):
    # 四个展柜组成两簇共模突变（00:06/00:07 与 00:40/00:41），
    # 两簇相距 33 分钟、超出 600s 容差：全局选择应得到两个互不重叠事件，
    # 而不是链式归并成一个大候选。
    client.post("/api/v1/readings/batch", json={"readings": _step_payload("CASE-A", 6)})
    client.post("/api/v1/readings/batch", json={"readings": _step_payload("CASE-B", 7)})
    client.post("/api/v1/readings/batch", json={"readings": _step_payload("CASE-C", 40)})
    client.post("/api/v1/readings/batch", json={"readings": _step_payload("CASE-D", 41)})

    r = client.get(
        "/api/v1/incidents",
        params={
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-09-01T02:00:00Z",
            "tolerance_seconds": 600,
            "min_devices": 2,
        },
    )
    assert r.status_code == 200, r.text
    incidents = r.json()["incidents"]
    starts = [e["start_ts"] for e in incidents]
    assert starts == sorted(starts)
    assert len(incidents) == 2
    assert [{b["device_sn"] for b in e["boundaries"]} for e in incidents] == [
        {"CASE-A", "CASE-B"},
        {"CASE-C", "CASE-D"},
    ]
    # 严格互不重叠
    assert incidents[1]["start_ts"] > incidents[0]["end_ts"]


def test_incidents_api_min_devices_filters(client):
    client.post("/api/v1/readings/batch", json={"readings": _step_payload("CASE-A", 6)})
    client.post("/api/v1/readings/batch", json={"readings": _step_payload("CASE-B", 20)})
    r = client.get(
        "/api/v1/incidents",
        params={
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-09-01T02:00:00Z",
            "tolerance_seconds": 300,
            "min_devices": 2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    # 两次跳变相距 14 分钟，超过 300s 容差，无法同候选 → 无事件
    assert body["incidents"] == []
    assert body["incident_count"] == 0
    assert body["boundary_count"] == 2


def test_incidents_api_empty_db(client):
    r = client.get(
        "/api/v1/incidents",
        params={
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-09-01T02:00:00Z",
            "tolerance_seconds": 600,
            "min_devices": 2,
        },
    )
    assert r.status_code == 200
    assert r.json()["incidents"] == []


def test_incidents_api_time_range_filters_boundaries(client):
    client.post("/api/v1/readings/batch", json={"readings": _step_payload("CASE-A", 6)})
    client.post("/api/v1/readings/batch", json={"readings": _step_payload("CASE-B", 7)})
    r = client.get(
        "/api/v1/incidents",
        params={
            "start": "2026-09-01T01:00:00Z",
            "end": "2026-09-01T02:00:00Z",
            "tolerance_seconds": 600,
            "min_devices": 2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["boundary_count"] == 0
    assert body["incidents"] == []


def test_incidents_api_validation_422(client):
    base = {
        "start": "2026-09-01T00:00:00Z",
        "end": "2026-09-01T02:00:00Z",
        "tolerance_seconds": 600,
        "min_devices": 2,
    }

    def get(params):
        return client.get("/api/v1/incidents", params=params)

    # start 晚于 end
    bad = dict(base, start="2026-09-02T00:00:00Z")
    assert get(bad).status_code == 422
    # 非正容差（0 与负数）
    assert get(dict(base, tolerance_seconds=0)).status_code == 422
    assert get(dict(base, tolerance_seconds=-5)).status_code == 422
    # min_devices < 2
    assert get(dict(base, min_devices=1)).status_code == 422
    assert get(dict(base, min_devices=0)).status_code == 422
    # 缺少必填参数
    assert get({k: v for k, v in base.items() if k != "start"}).status_code == 422
    assert get({k: v for k, v in base.items() if k != "tolerance_seconds"}).status_code == 422
    assert get({k: v for k, v in base.items() if k != "min_devices"}).status_code == 422
    # 非法时间格式
    assert get(dict(base, start="not-a-time")).status_code == 422


def test_incidents_api_down_direction(client):
    # 同步降温降湿：边界方向应为 down
    for sn in ("CASE-D1", "CASE-D2"):
        client.post(
            "/api/v1/readings/batch",
            json={"readings": _step_payload(sn, 6, t0=24.0, t1=20.0, h0=70.0, h1=55.0)},
        )
    r = client.get(
        "/api/v1/incidents",
        params={
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-09-01T02:00:00Z",
            "tolerance_seconds": 600,
            "min_devices": 2,
        },
    )
    assert r.status_code == 200, r.text
    event = r.json()["incidents"][0]
    assert all(b["direction"] == "down" for b in event["boundaries"])
    assert all(b["delta_temperature"] < 0 for b in event["boundaries"])


def test_incidents_api_start_equals_end_ok(client):
    client.post("/api/v1/readings/batch", json={"readings": _step_payload("CASE-A", 6)})
    client.post("/api/v1/readings/batch", json={"readings": _step_payload("CASE-B", 6)})
    r = client.get(
        "/api/v1/incidents",
        params={
            "start": "2026-09-01T00:06:00Z",
            "end": "2026-09-01T00:06:00Z",
            "tolerance_seconds": 1,
            "min_devices": 2,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["boundary_count"] == 2
    assert len(body["incidents"]) == 1
    assert body["incidents"][0]["device_count"] == 2
