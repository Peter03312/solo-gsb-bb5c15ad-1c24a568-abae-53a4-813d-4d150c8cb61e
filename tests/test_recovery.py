"""展柜扰动恢复分析：纯算法单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine, init_db
from app.main import app
from app.recovery import (
    DWELL_INCOMPLETE_AT_HORIZON,
    DWELL_INTERRUPTED,
    NEVER_SIMULTANEOUSLY_IN_BAND,
    NO_POST_EVENT_READINGS,
    InsufficientBaselineSamples,
    analyze_recovery,
)
from app.segmentation import Sample

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
EVENT = BASE + dt.timedelta(hours=1)  # 01:00Z


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def _at(minutes: float, t: float, h: float) -> Sample:
    """相对 BASE 的第 minutes 分钟读数。"""
    return Sample(BASE + dt.timedelta(minutes=minutes), t, h)


def _reading(sn, minutes, t, h):
    ts = (BASE + dt.timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


BASELINE = [_at(i, 20.0, 50.0) for i in (10, 20, 30, 40, 50)]
KW = dict(
    event_ts=EVENT,
    baseline_seconds=3600.0,
    horizon_seconds=7200.0,
    min_baseline_samples=3,
    temp_tolerance=1.0,
    humidity_tolerance=2.0,
    hold_seconds=300.0,
    max_gap_seconds=600.0,
)


# ---------- 基线 ----------


def test_baseline_median_and_band_edges():
    samples = [
        _at(10, 18.0, 46.0),
        _at(20, 22.0, 54.0),
        _at(30, 20.0, 50.0),
        _at(40, 19.0, 48.0),
        _at(50, 21.0, 52.0),
        _at(70, 30.0, 80.0),
    ]
    r = analyze_recovery(samples, **KW)
    assert r.baseline.sample_count == 5
    assert r.baseline.temperature == 20.0
    assert r.baseline.humidity == 50.0
    assert r.baseline.window_start == BASE
    assert r.baseline.window_end == EVENT
    assert (r.baseline.temperature_low, r.baseline.temperature_high) == (19.0, 21.0)
    assert (r.baseline.humidity_low, r.baseline.humidity_high) == (48.0, 52.0)


def test_baseline_even_count_averages_middle_pair():
    samples = [_at(10, 10.0, 40.0), _at(20, 20.0, 50.0),
               _at(30, 30.0, 60.0), _at(40, 40.0, 70.0),
               _at(70, 25.0, 55.0)]
    r = analyze_recovery(samples, **{**KW, "min_baseline_samples": 2})
    assert r.baseline.sample_count == 4
    assert r.baseline.temperature == 25.0  # (20 + 30) / 2
    assert r.baseline.humidity == 55.0


def test_reading_exactly_at_event_ts_counts_as_baseline():
    samples = BASELINE + [Sample(EVENT, 20.0, 50.0), _at(70, 30.0, 80.0)]
    r = analyze_recovery(samples, **{**KW, "min_baseline_samples": 6})
    assert r.baseline.sample_count == 6
    assert r.post_event_sample_count == 1


# ---------- 事件时刻读数兼作首段左端点 ----------


def test_first_segment_event_reading_in_band_reaches_hold():
    # 事件时刻读数（20, 50）仍计入基线，同时是首段左端点：首段
    # [01:00, 01:05] 两维恒在带内，恰好等于 hold=300s 即确认恢复。
    # 旧实现漏算首段：唯一事件后读数 01:05 没有任何区间，必判未恢复。
    samples = BASELINE + [
        Sample(EVENT, 20.0, 50.0),
        _at(65, 20.0, 50.0),
    ]
    ok = analyze_recovery(samples, **{**KW, "hold_seconds": 300.0})
    assert ok.recovered is True
    assert ok.band_enter_ts == EVENT
    assert ok.recovered_at == EVENT + dt.timedelta(seconds=300.0)
    assert ok.recovery_seconds == pytest.approx(300.0)
    # hold 再多 1 秒：首段恰好 300s，不足确认
    short = analyze_recovery(samples, **{**KW, "hold_seconds": 301.0})
    assert short.recovered is False
    assert short.reason == DWELL_INCOMPLETE_AT_HORIZON
    # 首段持续到观测末点但保持不足，仍不返回成功保持段起点
    assert short.band_enter_ts is None


def test_first_segment_interpolated_entry_within_first_segment():
    # 事件时刻读数冲出带外（24, 58）但仍计入中位数基线；首段
    # [01:00, 01:10] 线性回到带心，温度、湿度均在 75% 处（+7.5min，
    # 即 01:07:30）同时入带，首段内在带持续恰为 150s；hold=150
    # 恰在首段终点 01:10 确认。
    samples = BASELINE + [
        Sample(EVENT, 24.0, 58.0),
        _at(70, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **{**KW, "hold_seconds": 150.0})
    assert r.recovered is True
    assert r.band_enter_ts == BASE + dt.timedelta(minutes=67.5)
    assert r.recovered_at == BASE + dt.timedelta(minutes=70.0)
    assert r.recovery_seconds == pytest.approx(600.0)
    # 旧实现下唯一事件后读数没有相邻区间，不会恢复
    assert r.post_event_sample_count == 1
    # 首段在带仅 150s，hold 再多 1 秒即不足，且保持延伸到观测末点
    short = analyze_recovery(samples, **{**KW, "hold_seconds": 151.0})
    assert short.recovered is False
    assert short.reason == DWELL_INCOMPLETE_AT_HORIZON


def test_first_segment_gap_over_max_gap_is_dropped():
    # 事件时刻读数在带内，但与首条事件后读数间隔 900s > max_gap=600：
    # 首段断开，事件时刻的在带状态不得带入后续计时；之后 01:15 起
    # 连续在带，保持计时只能从 01:15 重新开始。
    samples = BASELINE + [
        Sample(EVENT, 20.0, 50.0),
        _at(75, 20.0, 50.0),
        _at(80, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **{**KW, "hold_seconds": 300.0})
    assert r.recovered is True
    assert r.band_enter_ts == BASE + dt.timedelta(minutes=75.0)
    assert r.recovered_at == BASE + dt.timedelta(minutes=80.0)

    # 若错误地把首段当作连续：会从事件时刻累计 600s 而提前恢复
    early = analyze_recovery(samples, **{**KW, "hold_seconds": 600.0})
    assert early.recovered is False
    assert early.reason == DWELL_INCOMPLETE_AT_HORIZON

    # max_gap 放宽到 900（恰好等于仍连续）：从事件时刻起累计 600s，
    # 在 01:10 确认
    wide = analyze_recovery(
        samples, **{**KW, "hold_seconds": 600.0, "max_gap_seconds": 900.0}
    )
    assert wide.recovered is True
    assert wide.band_enter_ts == EVENT
    assert wide.recovered_at == BASE + dt.timedelta(minutes=70.0)


def test_first_segment_gap_with_single_post_reading_never_starts():
    # 首段断开后只剩一条孤立的事件后读数（在带内）：没有任何正长度
    # 有效区间，不能凭“事件时刻 + 孤立读数都在带内”判恢复或中断
    samples = BASELINE + [
        Sample(EVENT, 20.0, 50.0),
        _at(75, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **{**KW, "hold_seconds": 300.0})
    assert r.recovered is False
    assert r.band_enter_ts is None
    assert r.reason == NEVER_SIMULTANEOUSLY_IN_BAND


def test_first_segment_gap_then_exit_interrupts_restarted_timer():
    # 首段断档后，01:15→01:20 在带（重新计时），随后 01:25 出带：
    # 保持计时被出带终结 → dwell_interrupted，而非观察窗末保持不足
    samples = BASELINE + [
        Sample(EVENT, 20.0, 50.0),
        _at(75, 20.0, 50.0),
        _at(80, 20.0, 50.0),
        _at(85, 20.0, 60.0),
    ]
    r = analyze_recovery(samples, **{**KW, "hold_seconds": 600.0})
    assert r.recovered is False
    assert r.reason == DWELL_INTERRUPTED


def test_first_segment_gap_out_of_band_never_simultaneously_in_band():
    # 首段既断档、事件后读数又在带外：归类为从未两维同时在带
    samples = BASELINE + [
        Sample(EVENT, 20.0, 50.0),
        _at(75, 30.0, 80.0),
    ]
    r = analyze_recovery(samples, **KW)
    assert r.recovered is False
    assert r.reason == NEVER_SIMULTANEOUSLY_IN_BAND


def test_dwell_continuity_across_first_segment_boundary():
    # 事件时刻读数在带内：首段 [01:00, 01:05] 在带，并与后续线段在
    # 01:05 读数时刻相接 → 连续计时；hold=600 恰在 01:10 确认。
    samples = BASELINE + [
        Sample(EVENT, 20.0, 50.0),
        _at(65, 20.0, 50.0),
        _at(70, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **{**KW, "hold_seconds": 600.0})
    assert r.recovered is True
    assert r.band_enter_ts == EVENT
    assert r.recovered_at == BASE + dt.timedelta(minutes=70.0)

    # 跨首段延续后在 01:05→01:10 段内出带：计时按精确交点中断
    exit_mid = samples[:-1] + [_at(70, 22.0, 58.0)]
    r2 = analyze_recovery(exit_mid, **{**KW, "hold_seconds": 600.0})
    assert r2.recovered is False
    # 温度在 01:07:30 越出 21、湿度在 01:07 越出 52 → 仅保持 420s
    assert r2.reason == DWELL_INTERRUPTED


def test_event_ts_reading_excluded_from_post_count_and_max_deviation():
    # 事件时刻读数是偏离最大的点，但它计入基线，绝不能计入事件后
    # 读数数目与最大偏差
    samples = BASELINE + [
        Sample(EVENT, 30.0, 80.0),
        _at(65, 22.0, 54.0),
        _at(70, 20.0, 50.0),
        _at(75, 20.0, 50.0),
        _at(80, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **KW)
    assert r.baseline.sample_count == 6
    assert r.post_event_sample_count == 4
    # 严格事件后读数的最大偏差来自 01:05 的 (22, 54)
    assert r.max_temperature_deviation == pytest.approx(2.0)
    assert r.max_temperature_deviation_ts == BASE + dt.timedelta(minutes=65)
    assert r.max_humidity_deviation == pytest.approx(4.0)
    assert r.max_humidity_deviation_ts == BASE + dt.timedelta(minutes=65)


def test_insufficient_baseline_samples_raises():
    with pytest.raises(InsufficientBaselineSamples) as exc:
        analyze_recovery(BASELINE[:2] + [_at(70, 30.0, 80.0)], **KW)
    assert exc.value.baseline_sample_count == 2
    assert exc.value.min_baseline_samples == 3
    assert exc.value.window_start == BASE
    assert exc.value.window_end == EVENT


# ---------- 恢复判定与精确交点 ----------


def test_recovers_at_exact_interpolated_entry_plus_hold():
    # 温度 24@61 → 22@65 → 20@70：在 65→70 段中点 67.5min 入带 21；
    # 湿度 58 → 54 → 50：同段中点 67.5min 入带 52；之后保持在带内。
    samples = BASELINE + [
        _at(61, 24.0, 58.0),
        _at(65, 22.0, 54.0),
        _at(70, 20.0, 50.0),
        _at(75, 20.0, 50.0),
        _at(80, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **KW)
    assert r.recovered is True
    assert r.band_enter_ts == BASE + dt.timedelta(minutes=67.5)
    assert r.recovered_at == BASE + dt.timedelta(minutes=72.5)
    assert r.recovery_seconds == pytest.approx(750.0)
    assert r.reason is None


def test_dimensions_enter_band_at_different_times_takes_later():
    # 温度 24 → 20 在 61→65 段中点 63min 入带；
    # 湿度 58 → 50 在该段 75% 处 64min 入带（(58+50)/2=54, 52 需 6/8）。
    samples = BASELINE + [
        _at(61, 24.0, 58.0),
        _at(65, 20.0, 50.0),
        _at(70, 20.0, 50.0),
        _at(75, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **{**KW, "hold_seconds": 600.0})
    assert r.recovered is True
    assert r.band_enter_ts == BASE + dt.timedelta(minutes=64.0)
    assert r.recovered_at == BASE + dt.timedelta(minutes=74.0)


def test_band_boundary_is_inclusive():
    # 事件后读数恰好落在温度 21 / 湿度 52 的容差边界上：仍算在带内
    samples = BASELINE + [
        _at(61, 30.0, 80.0),
        _at(70, 21.0, 52.0),
        _at(80, 21.0, 52.0),
    ]
    r = analyze_recovery(samples, **{**KW, "hold_seconds": 600.0})
    assert r.recovered is True
    assert r.band_enter_ts == BASE + dt.timedelta(minutes=70)


def test_hold_equal_duration_confirms_recovery():
    # 同时在带区间恰为 [67.5, 75)（湿度 75 时恰在边界，75 后出带），
    # 持续 450s，hold=450 恰好确认；hold=451 不确认。
    samples = BASELINE + [
        _at(61, 24.0, 58.0),
        _at(65, 22.0, 54.0),
        _at(70, 20.0, 50.0),
        _at(75, 20.0, 52.0),  # 湿度恰在上界，仍在带
        _at(80, 20.0, 55.0),  # 湿度出带
    ]
    ok = analyze_recovery(samples, **{**KW, "hold_seconds": 450.0})
    assert ok.recovered is True
    assert ok.recovered_at == BASE + dt.timedelta(minutes=75.0)
    short = analyze_recovery(samples, **{**KW, "hold_seconds": 451.0})
    assert short.recovered is False
    assert short.reason == DWELL_INTERRUPTED


def test_exit_resets_hold_timer():
    # 67.5min 起两维在带；湿度在 75→80 段内先出带（55 起，约 78.75min
    # 越过 52）又在 80 回到带内 → 保持计时中断并重新开始，重启后的段
    # 连续延伸到观测末点 85，但仅有 5 分钟（< hold=900s）
    samples = BASELINE + [
        _at(61, 24.0, 58.0),
        _at(65, 22.0, 54.0),
        _at(70, 20.0, 50.0),
        _at(75, 20.0, 55.0),
        _at(80, 20.0, 50.0),
        _at(85, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **{**KW, "hold_seconds": 900.0})
    assert r.recovered is False
    # 末点 85 在带内且重新计时的段连续到末点 → 保持时间不足而非中断
    assert r.reason == DWELL_INCOMPLETE_AT_HORIZON

    # 末点处于出带状态：计时确已被中断
    end_out = samples[:-1] + [_at(85, 20.0, 60.0)]
    r2 = analyze_recovery(end_out, **{**KW, "hold_seconds": 900.0})
    assert r2.recovered is False
    assert r2.reason == DWELL_INTERRUPTED


def test_gap_larger_than_max_gap_interrupts_and_restarts():
    # 75 → 90 间隔 900s > max_gap=600：曲线断开，之前的在带计时作废；
    # 90 的孤立在带读数之后无相邻区间，无法恢复。
    samples = BASELINE + [
        _at(61, 24.0, 58.0),
        _at(65, 22.0, 54.0),
        _at(70, 20.0, 50.0),
        _at(75, 20.0, 50.0),
        _at(90, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **{**KW, "hold_seconds": 600.0})
    assert r.recovered is False
    assert r.reason == DWELL_INTERRUPTED

    # 间隔恰好等于 max_gap：曲线连续，在带保持跨该区间累计
    equal = analyze_recovery(samples, **{**KW, "hold_seconds": 600.0,
                                         "max_gap_seconds": 900.0})
    assert equal.recovered is True
    assert equal.recovered_at == BASE + dt.timedelta(minutes=77.5)


def test_hold_incomplete_at_horizon():
    samples = BASELINE + [
        _at(61, 24.0, 58.0),
        _at(65, 22.0, 54.0),
        _at(70, 20.0, 50.0),
        _at(75, 20.0, 50.0),
        _at(80, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **{**KW, "hold_seconds": 1000.0})
    assert r.recovered is False
    assert r.reason == DWELL_INCOMPLETE_AT_HORIZON
    assert r.band_enter_ts is None  # 未满 hold，不返回成功保持段起点


def test_never_simultaneously_in_band():
    samples = BASELINE + [
        _at(61, 30.0, 80.0),
        _at(70, 30.0, 80.0),
        _at(80, 30.0, 80.0),
    ]
    r = analyze_recovery(samples, **KW)
    assert r.recovered is False
    assert r.reason == NEVER_SIMULTANEOUSLY_IN_BAND


def test_no_post_event_readings():
    r = analyze_recovery(BASELINE, **KW)
    assert r.recovered is False
    assert r.reason == NO_POST_EVENT_READINGS
    assert r.post_event_sample_count == 0
    assert r.max_temperature_deviation is None
    assert r.max_temperature_deviation_ts is None
    assert r.max_humidity_deviation is None
    assert r.max_humidity_deviation_ts is None


def test_post_event_window_exclusive_start_inclusive_end():
    # 恰在 event_ts 的读数不算事件后；恰在 event_ts + horizon 的读数算
    edge = EVENT + dt.timedelta(seconds=7200.0)
    samples = BASELINE + [
        Sample(EVENT, 20.0, 50.0),
        Sample(edge, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **KW)
    assert r.post_event_sample_count == 1
    assert r.horizon_end == edge


# ---------- 事件后最大偏差 ----------


def test_max_deviation_per_dimension_with_tie_earliest():
    samples = BASELINE + [
        _at(61, 24.0, 58.0),   # 温度 +4，湿度 +8
        _at(65, 16.0, 42.0),   # 温度 −4，湿度 −8（并列偏差取最早 61）
        _at(70, 20.0, 50.0),
    ]
    r = analyze_recovery(samples, **{**KW, "hold_seconds": 1.0})
    assert r.max_temperature_deviation == pytest.approx(4.0)
    assert r.max_temperature_deviation_ts == BASE + dt.timedelta(minutes=61)
    assert r.max_humidity_deviation == pytest.approx(8.0)
    assert r.max_humidity_deviation_ts == BASE + dt.timedelta(minutes=61)


# ---------- 确定性与排序 ----------


def test_unsorted_input_deterministic():
    samples = BASELINE + [
        _at(61, 24.0, 58.0),
        _at(65, 22.0, 54.0),
        _at(70, 20.0, 50.0),
        _at(75, 20.0, 50.0),
        _at(80, 20.0, 50.0),
    ]
    ordered = analyze_recovery(samples, **KW)
    shuffled = analyze_recovery(list(reversed(samples)), **KW)
    assert ordered == shuffled


# ---------- 接口集成 ----------


def _ingest(client, sn, rows):
    r = client.post("/api/v1/readings/batch", json={"readings": rows})
    assert r.status_code == 200, r.text


def _params(**over):
    p = {
        "event_ts": EVENT.isoformat().replace("+00:00", "Z"),
        "baseline_seconds": 3600,
        "horizon_seconds": 7200,
        "min_baseline_samples": 3,
        "temp_tolerance": 1.0,
        "humidity_tolerance": 2.0,
        "hold_seconds": 300,
        "max_gap": 600,
    }
    p.update(over)
    return p


def test_endpoint_recovers(client):
    rows = [_reading("RC-1", i, 20.0, 50.0) for i in (10, 20, 30, 40, 50)]
    rows += [
        _reading("RC-1", 61, 24.0, 58.0),
        _reading("RC-1", 65, 22.0, 54.0),
        _reading("RC-1", 70, 20.0, 50.0),
        _reading("RC-1", 75, 20.0, 50.0),
        _reading("RC-1", 80, 20.0, 50.0),
    ]
    _ingest(client, "RC-1", rows)
    r = client.get("/api/v1/cases/RC-1/recovery", params=_params())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "RC-1"
    assert body["recovered"] is True
    assert body["band_enter_ts"].startswith("2026-09-01T01:07:30")
    assert body["recovered_at"].startswith("2026-09-01T01:12:30")
    assert body["recovery_seconds"] == pytest.approx(750.0)
    assert body["reason"] is None
    assert body["baseline"]["temperature"] == 20.0
    assert body["baseline"]["humidity"] == 50.0
    assert body["baseline"]["sample_count"] == 5
    assert body["baseline_sample_count"] == 5
    assert body["post_event_sample_count"] == 5
    assert body["max_temperature_deviation"] == 4.0
    assert body["max_humidity_deviation"] == 8.0
    assert body["event_ts"].startswith("2026-09-01T01:00:00")
    assert body["horizon_end"].startswith("2026-09-01T03:00:00")
    assert body["hold_seconds"] == 300
    assert body["max_gap_seconds"] == 600


def test_endpoint_first_segment_recovers(client):
    # 事件时刻读数计入基线并兼作首段左端点：首段 [01:00, 01:05]
    # 两维恒在带内恰为 hold=300s，旧实现漏算首段会判未恢复。
    rows = [_reading("RC-1F", i, 20.0, 50.0) for i in (10, 20, 30, 40, 50)]
    rows.append(_reading("RC-1F", 60, 20.0, 50.0))  # 恰在事件时刻
    rows.append(_reading("RC-1F", 65, 20.0, 50.0))  # 首条事件后读数
    _ingest(client, "RC-1F", rows)
    body = client.get(
        "/api/v1/cases/RC-1F/recovery", params=_params()
    ).json()
    assert body["recovered"] is True
    assert body["band_enter_ts"].startswith("2026-09-01T01:00:00")
    assert body["recovered_at"].startswith("2026-09-01T01:05:00")
    assert body["recovery_seconds"] == pytest.approx(300.0)
    # 事件时刻读数计入基线，但不计入事件后读数数目
    assert body["baseline_sample_count"] == 6
    assert body["post_event_sample_count"] == 1


def test_endpoint_not_recovered_returns_reason(client):
    rows = [_reading("RC-2", i, 20.0, 50.0) for i in (10, 20, 30, 40, 50)]
    rows += [_reading("RC-2", i, 30.0, 80.0) for i in (61, 70, 80)]
    _ingest(client, "RC-2", rows)
    r = client.get("/api/v1/cases/RC-2/recovery", params=_params())
    assert r.status_code == 200
    body = r.json()
    assert body["recovered"] is False
    assert body["recovered_at"] is None
    assert body["recovery_seconds"] is None
    assert body["band_enter_ts"] is None
    assert body["reason"] == "never_simultaneously_in_band"
    assert body["max_temperature_deviation"] == 10.0
    assert body["max_humidity_deviation"] == 30.0


def test_endpoint_insufficient_baseline_samples_422(client):
    rows = [_reading("RC-3", i, 20.0, 50.0) for i in (30, 40, 50)]
    rows += [_reading("RC-3", 70, 20.0, 50.0)]
    _ingest(client, "RC-3", rows)
    r = client.get("/api/v1/cases/RC-3/recovery",
                   params=_params(min_baseline_samples=5))
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["baseline_sample_count"] == 3
    assert detail["min_baseline_samples"] == 5
    assert "基线样本不足" in detail["message"]
    assert detail["baseline_window_end"].startswith("2026-09-01T01:00:00")


def test_endpoint_unknown_case_404(client):
    r = client.get("/api/v1/cases/NOPE/recovery", params=_params())
    assert r.status_code == 404


def test_endpoint_validation_errors(client):
    rows = [_reading("RC-4", i, 20.0, 50.0) for i in (10, 20, 30, 40, 50)]
    rows += [_reading("RC-4", 70, 20.0, 50.0)]
    _ingest(client, "RC-4", rows)
    url = "/api/v1/cases/RC-4/recovery"

    # 缺少必填参数
    assert client.get(url, params={"event_ts": _params()["event_ts"]}).status_code == 422
    # 非正参数
    for key, value in [
        ("baseline_seconds", 0),
        ("horizon_seconds", -1),
        ("temp_tolerance", 0),
        ("humidity_tolerance", -2),
        ("hold_seconds", 0),
        ("max_gap", -1),
    ]:
        assert client.get(url, params=_params(**{key: value})).status_code == 422, (key, value)
    # min_baseline_samples 至少 1
    assert client.get(url, params=_params(min_baseline_samples=0)).status_code == 422
    # max_gap 上限
    assert client.get(url, params=_params(max_gap=100000)).status_code == 422
    # 时间戳非法
    assert client.get(url, params=_params(event_ts="not-a-timestamp")).status_code == 422


def test_endpoint_default_max_gap(client):
    rows = [_reading("RC-5", i, 20.0, 50.0) for i in (10, 20, 30, 40, 50)]
    rows += [
        _reading("RC-5", 61, 24.0, 58.0),
        _reading("RC-5", 65, 22.0, 54.0),
        _reading("RC-5", 70, 20.0, 50.0),
        _reading("RC-5", 75, 20.0, 50.0),
    ]
    _ingest(client, "RC-5", rows)
    p = _params()
    del p["max_gap"]
    body = client.get(url="/api/v1/cases/RC-5/recovery", params=p).json()
    assert body["max_gap_seconds"] == 600.0


def test_endpoint_is_read_only(client):
    """恢复查询不得改变读数、分段与版本号。"""
    from app.database import CaseState, Reading, SessionLocal

    rows = [_reading("RC-6", i, 20.0, 50.0) for i in (10, 20, 30, 40, 50)]
    rows += [_reading("RC-6", 70, 20.0, 50.0)]
    _ingest(client, "RC-6", rows)
    db = SessionLocal()
    try:
        version_before = db.get(CaseState, "RC-6").version
        count_before = db.query(Reading).filter(Reading.device_sn == "RC-6").count()
    finally:
        db.close()

    assert client.get("/api/v1/cases/RC-6/recovery", params=_params()).status_code == 200
    assert client.get("/api/v1/cases/RC-6/recovery", params=_params()).status_code == 200

    db = SessionLocal()
    try:
        db.expire_all()
        assert db.get(CaseState, "RC-6").version == version_before
        assert db.query(Reading).filter(Reading.device_sn == "RC-6").count() == count_before
    finally:
        db.close()


def test_endpoint_deterministic(client):
    rows = [_reading("RC-7", i, 20.0 + (i % 3), 50.0 + (i % 5))
            for i in range(10, 60, 5)]
    rows += [_reading("RC-7", i, 20.0, 50.0) for i in (61, 70, 80, 90)]
    _ingest(client, "RC-7", rows)
    p = _params(hold_seconds=120)
    first = client.get("/api/v1/cases/RC-7/recovery", params=p).json()
    second = client.get("/api/v1/cases/RC-7/recovery", params=p).json()
    assert first == second
