"""展柜同伴基线偏离查询：纯算法单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine, init_db
from app.deviation import compute_peer_baseline_deviation
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


def _sample(minute: float, t: float, h: float = 50.0) -> Sample:
    return Sample(BASE + dt.timedelta(seconds=60 * minute), t, h)


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _ts(minute: float) -> str:
    return (BASE + dt.timedelta(seconds=60 * minute)).isoformat().replace("+00:00", "Z")


# 对照柜恒定 20 / 22：基线中位数恒为 21.0
PEER1 = [_sample(i, 20.0) for i in range(11)]
PEER2 = [_sample(i, 22.0) for i in range(11)]


def _compute(target, peers=None, dimension="temperature", *, start=BASE,
             end=None, step=60.0, max_gap=60.0, threshold=2.0, min_duration=0.0):
    return compute_peer_baseline_deviation(
        target,
        [PEER1, PEER2] if peers is None else peers,
        dimension,
        start=start,
        end=end if end is not None else BASE + dt.timedelta(seconds=600),
        step_seconds=step,
        max_gap_seconds=max_gap,
        deviation_threshold=threshold,
        min_duration_seconds=min_duration,
    )


# 目标在 3..5 分钟升至 25（偏离 +4），其余时刻贴合基线 21
TARGET_UP = [_sample(i, 25.0 if 3 <= i <= 5 else 21.0) for i in range(11)]


# ---------- 纯算法：标记、方向、峰值 ----------


def test_basic_deviation_period_up():
    res = _compute(TARGET_UP)
    assert res.grid_point_count == 11
    assert res.evaluated_count == 11
    assert res.marked_count == 3
    assert len(res.periods) == 1
    p = res.periods[0]
    assert p.start_ts == BASE + dt.timedelta(seconds=180)
    assert p.end_ts == BASE + dt.timedelta(seconds=300)
    assert p.duration_seconds == 120.0
    assert p.marked_count == 3
    assert p.direction == "up"
    assert p.max_deviation == pytest.approx(4.0)
    assert p.peak_ts == BASE + dt.timedelta(seconds=180)  # 并列取最早


def test_direction_down():
    target = [_sample(i, 17.0 if 3 <= i <= 5 else 21.0) for i in range(11)]
    res = _compute(target)
    assert len(res.periods) == 1
    p = res.periods[0]
    assert p.direction == "down"
    assert p.max_deviation == pytest.approx(4.0)


def test_threshold_is_inclusive():
    res = _compute(TARGET_UP, threshold=4.0)  # 偏离恰好 4.0 仍标记
    assert res.marked_count == 3
    assert len(res.periods) == 1
    res = _compute(TARGET_UP, threshold=4.0001)
    assert res.marked_count == 0
    assert res.periods == ()


def test_median_of_even_peer_count_is_mean_of_middle_two():
    peers = [[_sample(i, 20.0) for i in range(11)], [_sample(i, 24.0) for i in range(11)]]
    target = [_sample(i, 25.0 if i == 4 else 22.0) for i in range(11)]
    res = _compute(target, peers=peers, threshold=2.0)
    # 基线 = (20+24)/2 = 22；仅第 4 分钟偏离 +3 达标
    assert res.marked_count == 1
    assert res.periods[0].max_deviation == pytest.approx(3.0)


def test_median_uses_all_valid_peers():
    peers = [
        [_sample(i, 20.0) for i in range(11)],
        [_sample(i, 22.0) for i in range(11)],
        [_sample(i, 30.0) for i in range(11)],
    ]
    target = [_sample(i, 25.0 if i == 4 else 22.0) for i in range(11)]
    res = _compute(target, peers=peers, threshold=2.0)
    # 三个对照中位数 = 22（奇数取中间值，不受 30 拉动）
    assert res.periods[0].max_deviation == pytest.approx(3.0)


def test_peak_tie_takes_earliest():
    target = [_sample(i, t) for i, t in enumerate(
        [21.0, 25.0, 26.0, 26.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0]
    )]
    res = _compute(target)
    p = res.periods[0]
    assert p.max_deviation == pytest.approx(5.0)
    assert p.peak_ts == BASE + dt.timedelta(seconds=120)  # 2、3 分钟并列取最早


def test_humidity_dimension():
    peers = [[_sample(i, 20.0, 40.0) for i in range(11)],
             [_sample(i, 22.0, 44.0) for i in range(11)]]
    target = [_sample(i, 21.0, 50.0 if 2 <= i <= 3 else 42.0) for i in range(11)]
    res = _compute(target, peers=peers, dimension="humidity", threshold=5.0)
    # 湿度基线 42，目标 2..3 分钟 50 → 偏离 +8
    assert res.marked_count == 2
    p = res.periods[0]
    assert p.direction == "up"
    assert p.max_deviation == pytest.approx(8.0)
    assert p.start_ts == BASE + dt.timedelta(seconds=120)


# ---------- 纯算法：断档、对照不足与合并切断 ----------


def test_fewer_than_two_valid_peers_interrupts_evaluation():
    # P2 只有 0..2 与 8..10 分钟读数；max_gap=60 时 3..7 分钟无效
    p2 = [_sample(i, 22.0) for i in (0, 1, 2, 8, 9, 10)]
    target = [_sample(i, 25.0 if 2 <= i <= 4 else 21.0) for i in range(11)]
    res = _compute(target, peers=[PEER1, p2])
    # 仅 0,1,2,8,9,10 六个时刻两柜对照同时有效；3、4 分钟仅 1 个有效
    # 对照 → 中断判定，只有第 2 分钟被标记
    assert res.evaluated_count == 6
    assert res.marked_count == 1
    assert len(res.periods) == 1
    p = res.periods[0]
    assert p.start_ts == p.end_ts == BASE + dt.timedelta(seconds=120)
    assert p.duration_seconds == 0.0
    assert p.marked_count == 1


def test_target_gap_interrupts_and_splits_periods():
    # 目标缺第 5 分钟读数，间隔 120s > max_gap=60 → 第 5 分钟无效并切断合并
    target = [_sample(i, 25.0 if 3 <= i <= 7 else 21.0) for i in range(11) if i != 5]
    res = _compute(target)
    assert res.evaluated_count == 10
    assert res.marked_count == 4
    assert len(res.periods) == 2
    first, second = res.periods
    assert (first.start_ts, first.end_ts) == (
        BASE + dt.timedelta(seconds=180),
        BASE + dt.timedelta(seconds=240),
    )
    assert (second.start_ts, second.end_ts) == (
        BASE + dt.timedelta(seconds=360),
        BASE + dt.timedelta(seconds=420),
    )


def test_unmarked_ticks_split_periods():
    target = [_sample(i, 25.0 if i in (2, 3, 7, 8) else 21.0) for i in range(11)]
    res = _compute(target)
    assert [ (p.start_ts, p.end_ts) for p in res.periods ] == [
        (BASE + dt.timedelta(seconds=120), BASE + dt.timedelta(seconds=180)),
        (BASE + dt.timedelta(seconds=420), BASE + dt.timedelta(seconds=480)),
    ]


def test_gap_equal_to_max_gap_is_still_continuous():
    target = [_sample(0, 25.0), _sample(1, 25.0)]  # 间隔 60s == max_gap
    res = _compute(target, start=BASE, end=BASE + dt.timedelta(seconds=60),
                   step=30.0, max_gap=60.0, threshold=2.0)
    # 0、30、60 秒三点全部有效（30 秒为断档边界内的线性插值）
    assert res.evaluated_count == 3
    assert res.marked_count == 3
    assert res.periods[0].duration_seconds == 60.0


def test_gap_larger_than_max_gap_invalidates_interior_point():
    target = [_sample(0, 25.0), _sample(2, 25.0)]  # 间隔 120s > max_gap=60
    res = _compute(target, start=BASE, end=BASE + dt.timedelta(seconds=120),
                   step=60.0, max_gap=60.0, threshold=2.0)
    # 中间 60 秒处无效：0、120 两个孤立标记不能合并
    assert res.evaluated_count == 2
    assert res.marked_count == 2
    assert len(res.periods) == 2
    assert all(p.duration_seconds == 0.0 for p in res.periods)


def test_outside_coverage_is_not_evaluated():
    target = [_sample(0, 25.0), _sample(1, 25.0)]
    res = _compute(target, start=BASE, end=BASE + dt.timedelta(seconds=600))
    # 网格 11 点，目标仅在 0、60 秒覆盖内有效
    assert res.grid_point_count == 11
    assert res.evaluated_count == 2


# ---------- 纯算法：最短持续、网格、确定性 ----------


def test_min_duration_filters_and_is_inclusive():
    target = [_sample(i, 25.0 if i in (2, 3, 4, 7) else 21.0) for i in range(11)]
    res = _compute(target, min_duration=90.0)
    # 时段 [2,4] 持续 120s 保留；[7,7] 持续 0s 被过滤
    assert len(res.periods) == 1
    assert res.periods[0].duration_seconds == 120.0
    # 含等于：min_duration 恰为 120 仍保留
    res = _compute(target, min_duration=120.0)
    assert len(res.periods) == 1
    res = _compute(target, min_duration=121.0)
    assert res.periods == ()


def test_grid_origin_at_start_and_tail_not_padded():
    res = _compute(TARGET_UP, start=BASE + dt.timedelta(seconds=30),
                   end=BASE + dt.timedelta(seconds=630))
    # 原点 30 秒，步长 60：30,90,...,630 共 11 点（690 超出终点不补）
    assert res.grid_point_count == 11
    p = res.periods[0]
    # 150s 处目标插值 23（偏离 2.0 达标）→ 时段起点；330s 同理为终点
    assert p.start_ts == BASE + dt.timedelta(seconds=150)
    assert p.end_ts == BASE + dt.timedelta(seconds=330)
    assert p.duration_seconds == 180.0
    assert p.marked_count == 4
    assert p.max_deviation == pytest.approx(4.0)
    assert p.peak_ts == BASE + dt.timedelta(seconds=210)  # 偏离 4.0 并列取最早


def test_no_valid_peers_gives_empty_result():
    res = _compute(TARGET_UP, peers=[[], []])
    assert res.evaluated_count == 0
    assert res.marked_count == 0
    assert res.periods == ()


def test_unsorted_input_deterministic():
    r1 = _compute(TARGET_UP)
    r2 = _compute(list(reversed(TARGET_UP)),
                  peers=[list(reversed(PEER1)), list(reversed(PEER2))])
    assert r1 == r2


# ---------- 接口集成 ----------


def _ingest(client, sn, rows):
    r = client.post("/api/v1/readings/batch", json={"readings": rows})
    assert r.status_code == 200, r.text


def _seed_cases(client, suffix=""):
    _ingest(client, f"CASE-T{suffix}", [
        _reading(f"CASE-T{suffix}", _ts(i), 25.0 if 3 <= i <= 5 else 21.0, 50.0)
        for i in range(11)
    ])
    _ingest(client, f"CASE-P1{suffix}", [
        _reading(f"CASE-P1{suffix}", _ts(i), 20.0, 50.0) for i in range(11)
    ])
    _ingest(client, f"CASE-P2{suffix}", [
        _reading(f"CASE-P2{suffix}", _ts(i), 22.0, 50.0) for i in range(11)
    ])


def _params(**over):
    p = {
        "target_sn": "CASE-T",
        "peer_sn": ["CASE-P1", "CASE-P2"],
        "dimension": "temperature",
        "start": "2026-09-01T00:00:00Z",
        "end": "2026-09-01T00:10:00Z",
        "step_seconds": 60,
        "deviation_threshold": 2.0,
        "min_duration_seconds": 0,
        "max_gap_seconds": 60,
    }
    p.update(over)
    return p


def test_endpoint_happy_path(client):
    _seed_cases(client)
    r = client.get("/api/v1/peer-baseline-deviation", params=_params())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_sn"] == "CASE-T"
    assert body["peer_sns"] == ["CASE-P1", "CASE-P2"]
    assert body["dimension"] == "temperature"
    assert body["grid_point_count"] == 11
    assert body["evaluated_count"] == 11
    assert body["marked_count"] == 3
    assert body["period_count"] == 1
    (p,) = body["periods"]
    assert p["start_ts"] == "2026-09-01T00:03:00Z"
    assert p["end_ts"] == "2026-09-01T00:05:00Z"
    assert p["duration_seconds"] == 120.0
    assert p["marked_count"] == 3
    assert p["direction"] == "up"
    assert p["max_deviation"] == pytest.approx(4.0)
    assert p["peak_ts"] == "2026-09-01T00:03:00Z"


def test_endpoint_unknown_target_404(client):
    _seed_cases(client)
    r = client.get("/api/v1/peer-baseline-deviation",
                   params=_params(target_sn="CASE-GHOST"))
    assert r.status_code == 404


def test_endpoint_unknown_peers_give_empty_result(client):
    _seed_cases(client)
    r = client.get(
        "/api/v1/peer-baseline-deviation",
        params=_params(peer_sn=["CASE-G1", "CASE-G2"]),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["evaluated_count"] == 0
    assert body["marked_count"] == 0
    assert body["period_count"] == 0
    assert body["periods"] == []


def test_endpoint_peer_validation(client):
    _seed_cases(client)
    # 对照数量不足 2 个
    assert client.get("/api/v1/peer-baseline-deviation",
                      params=_params(peer_sn=["CASE-P1"])).status_code == 422
    # 对照数量超过 10 个
    assert client.get(
        "/api/v1/peer-baseline-deviation",
        params=_params(peer_sn=[f"CASE-X{i}" for i in range(11)]),
    ).status_code == 422
    # 对照编号重复
    assert client.get(
        "/api/v1/peer-baseline-deviation",
        params=_params(peer_sn=["CASE-P1", "CASE-P1"]),
    ).status_code == 422
    # 对照包含目标
    assert client.get(
        "/api/v1/peer-baseline-deviation",
        params=_params(peer_sn=["CASE-P1", "CASE-T"]),
    ).status_code == 422
    # 对照编号为空串
    assert client.get(
        "/api/v1/peer-baseline-deviation",
        params=_params(peer_sn=["CASE-P1", ""]),
    ).status_code == 422


def test_endpoint_parameter_validation(client):
    _seed_cases(client)
    # start 晚于 end
    assert client.get(
        "/api/v1/peer-baseline-deviation",
        params=_params(start="2026-09-02T00:00:00Z", end="2026-09-01T00:00:00Z"),
    ).status_code == 422
    # 非法维度
    assert client.get("/api/v1/peer-baseline-deviation",
                      params=_params(dimension="pressure")).status_code == 422
    # 步长非正
    assert client.get("/api/v1/peer-baseline-deviation",
                      params=_params(step_seconds=0)).status_code == 422
    # 阈值非正
    assert client.get("/api/v1/peer-baseline-deviation",
                      params=_params(deviation_threshold=0)).status_code == 422
    # max_gap 非正
    assert client.get("/api/v1/peer-baseline-deviation",
                      params=_params(max_gap_seconds=-1)).status_code == 422
    # 最短持续为负
    assert client.get("/api/v1/peer-baseline-deviation",
                      params=_params(min_duration_seconds=-1)).status_code == 422
    # 缺少必填参数
    assert client.get("/api/v1/peer-baseline-deviation",
                      params={"target_sn": "CASE-T"}).status_code == 422


def test_endpoint_min_duration_filters_periods(client):
    _seed_cases(client)
    r = client.get("/api/v1/peer-baseline-deviation",
                   params=_params(min_duration_seconds=121))
    body = r.json()
    assert body["marked_count"] == 3  # 标记不受影响
    assert body["period_count"] == 0  # 但持续 120s < 121s 不返回


def test_endpoint_is_read_only(client):
    _seed_cases(client)
    before = {c["device_sn"]: c for c in client.get("/api/v1/cases").json()}
    r = client.get("/api/v1/peer-baseline-deviation", params=_params())
    assert r.status_code == 200
    after = {c["device_sn"]: c for c in client.get("/api/v1/cases").json()}
    assert before == after  # 版本号、读数数量均不变


def test_endpoint_deterministic(client):
    _seed_cases(client)
    first = client.get("/api/v1/peer-baseline-deviation", params=_params()).json()
    second = client.get("/api/v1/peer-baseline-deviation", params=_params()).json()
    assert first == second
