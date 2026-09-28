"""采样覆盖率审计：周期时隙 + 容差内一对一匹配（接口集成测试）。"""

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
from app.main import app


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


def _reading(sn, ts, t=20.0, h=50.0):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _ts(minutes, seconds=0):
    total = minutes * 60 + seconds
    hh, rem = divmod(total, 3600)
    mm, ss = divmod(rem, 60)
    return f"2026-09-01T{hh:02d}:{mm:02d}:{ss:02d}Z"


def _post(client, payload):
    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    assert r.status_code == 200, r.text


def _coverage(client, sn, **params):
    return client.get(f"/api/v1/cases/{sn}/sampling-coverage", params=params)


BASE = {
    "start": "2026-09-01T00:00:00Z",
    "end": "2026-09-01T01:00:00Z",
    "period_seconds": 600,
    "tolerance_seconds": 60,
}


# ---------- 时隙生成与基本覆盖 ----------


def test_full_coverage_no_gaps(client):
    _post(client, [_reading("CASE-S1", _ts(m)) for m in range(0, 61, 10)])
    r = _coverage(client, "CASE-S1", **BASE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "CASE-S1"
    assert body["start"] == "2026-09-01T00:00:00Z"
    assert body["end"] == "2026-09-01T01:00:00Z"
    assert body["period_seconds"] == 600.0
    assert body["tolerance_seconds"] == 60.0
    assert body["expected_count"] == 7
    assert body["covered_count"] == 7
    assert body["coverage"] == pytest.approx(1.0)
    assert body["gap_count"] == 0
    assert body["gaps"] == []


def test_slots_from_start_not_beyond_end(client):
    _post(client, [_reading("CASE-S2", _ts(m)) for m in range(0, 11)])
    params = {"start": "2026-09-01T00:00:00Z", "period_seconds": 300,
              "tolerance_seconds": 0}
    # end 恰好落在时隙上：00:00 / 00:05 / 00:10 共 3 个
    body = _coverage(client, "CASE-S2", **params,
                     end="2026-09-01T00:10:00Z").json()
    assert body["expected_count"] == 3
    assert body["covered_count"] == 3
    # end 差 1 秒不到下一个时隙：只有 00:00 / 00:05
    body = _coverage(client, "CASE-S2", **params,
                     end="2026-09-01T00:09:59Z").json()
    assert body["expected_count"] == 2
    # start == end：仅起点一个时隙
    body = _coverage(client, "CASE-S2", **params,
                     end="2026-09-01T00:00:00Z").json()
    assert body["expected_count"] == 1
    assert body["covered_count"] == 1


# ---------- 容差窗口（含等于） ----------


def test_tolerance_window_inclusive(client):
    # 单个时隙 00:00，容差 60 秒：恰好 ±60s 可覆盖，±61s 不可
    _post(client, [
        _reading("CASE-S3A", "2026-08-31T23:59:00Z"),
        _reading("CASE-S3B", "2026-08-31T23:58:59Z"),
        _reading("CASE-S3C", "2026-09-01T00:01:00Z"),
        _reading("CASE-S3D", "2026-09-01T00:01:01Z"),
    ])
    params = {"start": "2026-09-01T00:00:00Z", "end": "2026-09-01T00:00:00Z",
              "period_seconds": 600, "tolerance_seconds": 60}
    assert _coverage(client, "CASE-S3A", **params).json()["covered_count"] == 1
    assert _coverage(client, "CASE-S3B", **params).json()["covered_count"] == 0
    assert _coverage(client, "CASE-S3C", **params).json()["covered_count"] == 1
    assert _coverage(client, "CASE-S3D", **params).json()["covered_count"] == 0


def test_tolerance_extended_range_reaches_outside_window(client):
    # 首时隙之前、末时隙之后容差内的读数仍参与匹配
    _post(client, [
        _reading("CASE-S4", "2026-08-31T23:59:30Z"),
        _reading("CASE-S4", "2026-09-01T00:10:30Z"),
    ])
    body = _coverage(
        client, "CASE-S4", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:10:00Z", period_seconds=600, tolerance_seconds=60,
    ).json()
    assert body["covered_count"] == 2
    assert body["gaps"] == []


def test_readings_outside_extended_range_ignored(client):
    _post(client, [
        _reading("CASE-S5", "2026-08-31T20:00:00Z"),
        _reading("CASE-S5", "2026-09-01T05:00:00Z"),
    ])
    body = _coverage(client, "CASE-S5", **BASE).json()
    assert body["covered_count"] == 0
    assert body["coverage"] == pytest.approx(0.0)
    assert body["gaps"] == [
        {"start_ts": "2026-09-01T00:00:00Z",
         "end_ts": "2026-09-01T01:00:00Z", "missing_slots": 7}
    ]


# ---------- 一对一匹配 ----------


def test_reading_prefers_earliest_coverable_slot(client):
    # 时隙 00:00 / 00:10，容差 400s：两窗口在 00:03:20…00:06:40 重叠。
    # 00:05 的读数必须先覆盖 00:00（最早可覆盖），00:07 的读数再覆盖 00:10；
    # 若 00:05 错配 00:10，00:07 将无法覆盖 00:00（窗口止于 00:06:40）。
    _post(client, [
        _reading("CASE-S6", "2026-09-01T00:05:00Z"),
        _reading("CASE-S6", "2026-09-01T00:07:00Z"),
    ])
    body = _coverage(
        client, "CASE-S6", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:10:00Z", period_seconds=600, tolerance_seconds=400,
    ).json()
    assert body["covered_count"] == 2
    assert body["gaps"] == []


def test_one_reading_covers_at_most_one_slot(client):
    _post(client, [_reading("CASE-S7", "2026-09-01T00:00:00Z")])
    body = _coverage(client, "CASE-S7", **{**BASE, "tolerance_seconds": 0}).json()
    assert body["expected_count"] == 7
    assert body["covered_count"] == 1
    assert body["coverage"] == pytest.approx(1 / 7)
    # 缺测时隙 00:10…01:00 连续合并为一个缺口
    assert body["gap_count"] == 1
    assert body["gaps"] == [
        {"start_ts": "2026-09-01T00:10:00Z",
         "end_ts": "2026-09-01T01:00:00Z", "missing_slots": 6}
    ]


def test_slot_accepts_at_most_one_reading(client):
    _post(client, [
        _reading("CASE-S8", "2026-08-31T23:59:30Z"),
        _reading("CASE-S8", "2026-09-01T00:00:30Z"),
    ])
    body = _coverage(
        client, "CASE-S8", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:00:00Z", period_seconds=600, tolerance_seconds=60,
    ).json()
    assert body["expected_count"] == 1
    assert body["covered_count"] == 1
    assert body["coverage"] == pytest.approx(1.0)


def test_merged_readings_participate(client):
    # 同时间戳重复上报合并为一条读数；合并读数只能覆盖一个时隙
    _post(client, [_reading("CASE-S9", "2026-09-01T00:00:00Z")])
    _post(client, [_reading("CASE-S9", "2026-09-01T00:00:00Z", 21.0, 51.0)])
    body = _coverage(client, "CASE-S9", **{**BASE, "tolerance_seconds": 0}).json()
    assert body["covered_count"] == 1


# ---------- 缺口合并 ----------


def test_consecutive_missing_slots_merge_into_gaps(client):
    # 7 个时隙；覆盖 00:00 / 00:10 / 00:40 → 缺 00:20/00:30 与 00:50/01:00 两段
    _post(client, [
        _reading("CASE-S10", "2026-09-01T00:00:00Z"),
        _reading("CASE-S10", "2026-09-01T00:10:00Z"),
        _reading("CASE-S10", "2026-09-01T00:40:00Z"),
    ])
    body = _coverage(client, "CASE-S10", **{**BASE, "tolerance_seconds": 0}).json()
    assert body["expected_count"] == 7
    assert body["covered_count"] == 3
    assert body["coverage"] == pytest.approx(3 / 7)
    assert body["gap_count"] == 2
    assert body["gaps"] == [
        {"start_ts": "2026-09-01T00:20:00Z",
         "end_ts": "2026-09-01T00:30:00Z", "missing_slots": 2},
        {"start_ts": "2026-09-01T00:50:00Z",
         "end_ts": "2026-09-01T01:00:00Z", "missing_slots": 2},
    ]


def test_isolated_missing_slots_form_single_slot_gaps(client):
    # 覆盖 00:00 / 00:20 / 00:40 → 缺 00:10、00:30、00:50+01:00
    _post(client, [_reading("CASE-S11", _ts(m)) for m in (0, 20, 40)])
    body = _coverage(client, "CASE-S11", **{**BASE, "tolerance_seconds": 30}).json()
    assert body["covered_count"] == 3
    assert [g["missing_slots"] for g in body["gaps"]] == [1, 1, 2]
    assert [(g["start_ts"], g["end_ts"]) for g in body["gaps"]] == [
        ("2026-09-01T00:10:00Z", "2026-09-01T00:10:00Z"),
        ("2026-09-01T00:30:00Z", "2026-09-01T00:30:00Z"),
        ("2026-09-01T00:50:00Z", "2026-09-01T01:00:00Z"),
    ]


# ---------- 404 / 422 ----------


def test_unknown_case_404(client):
    r = _coverage(client, "NOPE", **BASE)
    assert r.status_code == 404


def test_422_start_after_end(client):
    _post(client, [_reading("CASE-S12", "2026-09-01T00:00:00Z")])
    r = _coverage(
        client, "CASE-S12", start="2026-09-01T01:00:00Z",
        end="2026-09-01T00:00:00Z", period_seconds=600, tolerance_seconds=60,
    )
    assert r.status_code == 422


def test_422_invalid_period_and_tolerance(client):
    _post(client, [_reading("CASE-S13", "2026-09-01T00:00:00Z")])
    base = {"start": "2026-09-01T00:00:00Z", "end": "2026-09-01T01:00:00Z"}
    # 周期必须为正
    assert _coverage(client, "CASE-S13", **base, period_seconds=0,
                     tolerance_seconds=0).status_code == 422
    assert _coverage(client, "CASE-S13", **base, period_seconds=-600,
                     tolerance_seconds=0).status_code == 422
    # 容差非负
    assert _coverage(client, "CASE-S13", **base, period_seconds=600,
                     tolerance_seconds=-1).status_code == 422
    # 容差必须严格小于周期：等于、大于均 422
    assert _coverage(client, "CASE-S13", **base, period_seconds=600,
                     tolerance_seconds=600).status_code == 422
    assert _coverage(client, "CASE-S13", **base, period_seconds=600,
                     tolerance_seconds=601).status_code == 422
    # 缺必填参数
    assert client.get("/api/v1/cases/CASE-S13/sampling-coverage").status_code == 422


# ---------- 周期边界：整数微秒归一化 ----------


def test_422_sub_microsecond_period(client):
    # 0.0000004 秒四舍五入到整数微秒为 0：旧实现会生成重复微秒时隙，
    # 虚增应采数与缺口；现在直接 422
    _post(client, [_reading("CASE-S20", "2026-09-01T00:00:00Z")])
    base = {"start": "2026-09-01T00:00:00Z", "end": "2026-09-01T01:00:00Z",
            "tolerance_seconds": 0}
    r = _coverage(client, "CASE-S20", **base, period_seconds=0.0000004)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["parameter"] == "period_seconds"
    assert "微秒" in detail["message"]
    # 远小于半微秒同样 422
    assert _coverage(client, "CASE-S20", **base,
                     period_seconds=1e-12).status_code == 422


def test_one_microsecond_period_slots_are_unique(client):
    # 1 微秒周期、99 微秒窗口 → 100 个互不重复的时隙，起点读数覆盖第 1 个
    _post(client, [_reading("CASE-S21", "2026-09-01T00:00:00Z")])
    r = _coverage(
        client, "CASE-S21", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:00:00.000099Z",
        period_seconds=0.000001, tolerance_seconds=0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["period_seconds"] == 1e-6
    assert body["expected_count"] == 100
    assert body["covered_count"] == 1
    assert body["gap_count"] == 1
    gap = body["gaps"][0]
    # 连续 99 个缺测微秒时隙合并为一个缺口，不出现重复时刻
    assert gap["missing_slots"] == 99
    assert gap["start_ts"] == "2026-09-01T00:00:00.000001Z"
    assert gap["end_ts"] == "2026-09-01T00:00:00.000099Z"


def test_period_normalized_to_effective_microseconds(client):
    # 非整数微秒的周期按四舍五入生效（0.6µs → 1µs），回显生效周期
    _post(client, [_reading("CASE-S22", "2026-09-01T00:00:00Z")])
    r = _coverage(
        client, "CASE-S22", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:00:00.000002Z",
        period_seconds=0.0000006, tolerance_seconds=0,
    )
    body = r.json()
    assert body["period_seconds"] == 1e-6
    assert body["expected_count"] == 3  # 0 / 1 / 2 微秒


def test_422_tolerance_equals_period_after_rounding(client):
    # 原值上容差 < 周期，但取整到微秒后两者相等 → 仍 422
    _post(client, [_reading("CASE-S23", "2026-09-01T00:00:00Z")])
    r = _coverage(
        client, "CASE-S23", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:10:00Z",
        period_seconds=1.0000004, tolerance_seconds=1.0000002,
    )
    assert r.status_code == 422
    assert r.json()["detail"]["parameter"] == "tolerance_seconds"


# ---------- 周期大于窗口 / 极大周期 ----------


def test_period_longer_than_window_emits_only_start_slot(client):
    # 周期 3601s > 3600s 窗口：只生成起点时隙，正常 200，不做越界运算
    _post(client, [_reading("CASE-S24", "2026-09-01T00:00:00Z")])
    r = _coverage(
        client, "CASE-S24", start="2026-09-01T00:00:00Z",
        end="2026-09-01T01:00:00Z", period_seconds=3601, tolerance_seconds=0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["expected_count"] == 1
    assert body["covered_count"] == 1
    assert body["gaps"] == []


def test_huge_positive_period_returns_200_not_500(client):
    # 旧实现试算 start + 1e100 秒时 timedelta 溢出 → 500；
    # 现在周期大于窗口只生成起点时隙，返回 200
    _post(client, [
        _reading("CASE-S25", "2026-09-01T00:00:30Z"),
        _reading("CASE-S25", "2026-09-01T00:02:01Z"),
    ])
    base = {"start": "2026-09-01T00:00:00Z", "end": "2026-09-01T01:00:00Z"}
    r = _coverage(client, "CASE-S25", **base,
                  period_seconds=1e100, tolerance_seconds=60)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["expected_count"] == 1
    # 00:00:30 在 60s 容差内覆盖起点；00:02:01 超容差，一对一匹配后仍只 1 个
    assert body["covered_count"] == 1
    assert body["gaps"] == []
    # 极大但仍在数值量级内的周期同样安全
    r2 = _coverage(client, "CASE-S25", **base,
                   period_seconds=1e30, tolerance_seconds=60)
    assert r2.status_code == 200, r2.text
    assert r2.json()["expected_count"] == 1


def test_huge_period_with_huge_tolerance_covers_far_reading(client):
    # 周期与容差都大到时间位移不可表示：起点时隙按饱和窗口覆盖全部读数
    _post(client, [_reading("CASE-S26", "2026-09-30T00:00:00Z")])
    r = _coverage(
        client, "CASE-S26", start="2026-09-01T00:00:00Z",
        end="2026-09-01T01:00:00Z", period_seconds=1e100, tolerance_seconds=1e90,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["expected_count"] == 1
    assert body["covered_count"] == 1


# ---------- 应采时隙数上限 100000 ----------


def test_expected_slots_at_limit_allowed(client):
    # 跨度 99999µs、周期 1µs → 恰好 100000 个时隙，放行
    _post(client, [_reading("CASE-S27", "2026-09-01T00:00:00Z")])
    r = _coverage(
        client, "CASE-S27", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:00:00.099999Z",
        period_seconds=0.000001, tolerance_seconds=0,
    )
    assert r.status_code == 200, r.text
    assert r.json()["expected_count"] == 100000


def test_422_expected_slots_over_limit(client):
    # 跨度 100000µs → 100001 个时隙，超过上限 100000 → 422（先算数，不构造）
    _post(client, [_reading("CASE-S28", "2026-09-01T00:00:00Z")])
    r = _coverage(
        client, "CASE-S28", start="2026-09-01T00:00:00Z",
        end="2026-09-01T00:00:00.100000Z",
        period_seconds=0.000001, tolerance_seconds=0,
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["parameter"] == "expected_count"
    assert detail["limit"] == 100000
    assert detail["actual"] == 100001


def test_422_over_limit_independent_of_case_existence(client):
    # 时隙数量按查询窗口预估，超限在查库之前即 422（不存在的设备也不返回 404）
    r = _coverage(
        client, "NOPE-LIMIT", start="2026-09-01T00:00:00Z",
        end="2026-09-01T01:00:00Z",
        period_seconds=0.000001, tolerance_seconds=0,
    )
    assert r.status_code == 422
    assert r.json()["detail"]["parameter"] == "expected_count"


def test_non_finite_period_and_tolerance_422(client):
    _post(client, [_reading("CASE-S29", "2026-09-01T00:00:00Z")])
    base = {"start": "2026-09-01T00:00:00Z", "end": "2026-09-01T01:00:00Z"}
    assert _coverage(client, "CASE-S29", **base,
                     period_seconds="inf", tolerance_seconds=0).status_code == 422
    assert _coverage(client, "CASE-S29", **base,
                     period_seconds=600, tolerance_seconds="inf").status_code == 422


# ---------- 纯计算：生效时隙序列的确定性与一致性 ----------


def test_unit_slots_are_integer_microseconds_and_shared():
    import datetime as dt

    from app.sampling import (
        compute_sampling_coverage,
        generate_slots,
        resolve_sampling_params,
    )

    start = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(milliseconds=1)
    resolved = resolve_sampling_params(start, end, 0.000001, 0.0)
    assert resolved.expected_count == 1001
    slots = generate_slots(start, resolved)
    # 旧实现 period_seconds=1e-6 逐次浮点累加会产生重复微秒时隙
    assert len(slots) == 1001
    assert len(set(slots)) == 1001
    assert all(s.microsecond == i for i, s in enumerate(slots[:1000]))

    class R:
        def __init__(self, ts):
            self.ts = ts

    # 匹配 / 覆盖率 / 缺口共用同一时隙序列：首时隙被覆盖，其余 1000 个成一缺口
    result = compute_sampling_coverage(
        [R(start)], start=start, end=end, resolved=resolved
    )
    assert result.expected_count == 1001
    assert result.covered_count == 1
    assert result.gaps[0].missing_slots == 1000
    assert result.gaps[0].start_ts == slots[1]
    assert result.gaps[0].end_ts == slots[-1]


def test_unit_sub_microsecond_period_rejected_before_slot_build():
    import datetime as dt

    import pytest as _pytest

    from app.sampling import SamplingParamsError, resolve_sampling_params

    start = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    with _pytest.raises(SamplingParamsError) as exc:
        resolve_sampling_params(start, start + dt.timedelta(hours=1), 4e-7, 0.0)
    assert exc.value.parameter == "period_seconds"


# ---------- 只读 & 确定性 ----------


def test_read_only_and_deterministic(client, db):
    _post(client, [_reading("CASE-S14", _ts(m)) for m in range(0, 61, 10)])

    state = db.get(CaseState, "CASE-S14")
    version_before = state.version
    readings_before = [
        (r.ts, r.temperature, r.humidity, r.report_count)
        for r in db.query(Reading)
        .filter(Reading.device_sn == "CASE-S14")
        .order_by(Reading.ts)
    ]
    snaps_before = db.query(SegmentSnapshot).filter(
        SegmentSnapshot.device_sn == "CASE-S14"
    ).count()

    r1 = _coverage(client, "CASE-S14", **BASE).json()
    r2 = _coverage(client, "CASE-S14", **BASE).json()
    assert r1 == r2

    db.expire_all()
    assert db.get(CaseState, "CASE-S14").version == version_before
    readings_after = [
        (r.ts, r.temperature, r.humidity, r.report_count)
        for r in db.query(Reading)
        .filter(Reading.device_sn == "CASE-S14")
        .order_by(Reading.ts)
    ]
    assert readings_after == readings_before
    assert (
        db.query(SegmentSnapshot)
        .filter(SegmentSnapshot.device_sn == "CASE-S14")
        .count()
        == snaps_before
    )
