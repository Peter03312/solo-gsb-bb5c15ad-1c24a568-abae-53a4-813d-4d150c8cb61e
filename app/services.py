"""业务逻辑：重复上报合并、确定性重放、锚点校准与无写预览。"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from . import calibration as cal
from . import condensation as cond
from . import deviation as dev
from . import excursions as exc
from . import incidents as inc
from . import outliers as ol
from . import propagation as prop
from . import recovery as rec
from . import sampling as samp
from . import segmentation as seg
from . import timeline as tl
from .config import settings
from .database import Anchor, CaseState, Reading, SegmentSnapshot, to_utc, utcnow
from .schemas import (
    AnchorIn,
    CalibratedPointOut,
    CalibratedTimelineOut,
    CondensationRiskOut,
    CondensationWindowOut,
    DeviationPeriodOut,
    ExcursionEventOut,
    ExcursionsOut,
    ExposureBudgetDimensionOut,
    ExposureBudgetOut,
    IncidentBoundaryOut,
    IncidentOut,
    IncidentsOut,
    PeerBaselineDeviationOut,
    PropagationLagOut,
    PropagationOut,
    ReadingIn,
    ReadingOutlierDimensionOut,
    ReadingOutlierOut,
    ReadingOutliersOut,
    RecoveryBaselineOut,
    RecoveryOut,
    SamplingCoverageOut,
    SamplingGapOut,
    SegmentOut,
    SegmentationOut,
    TimelineOut,
    TimelinePointOut,
)


# ---------- 读取 / 序列化 ----------


def _load_samples(db: Session, device_sn: str) -> list[seg.Sample]:
    rows = db.scalars(
        select(Reading)
        .where(Reading.device_sn == device_sn)
        .order_by(Reading.ts.asc())
    ).all()
    return [
        seg.Sample(
            ts=to_utc(r.ts),
            temperature=r.temperature,
            humidity=r.humidity,
            report_count=r.report_count,
        )
        for r in rows
    ]


def _load_anchor_ts(db: Session, device_sn: str) -> list[dt.datetime]:
    rows = db.scalars(
        select(Anchor.ts).where(Anchor.device_sn == device_sn)
    ).all()
    return [to_utc(t) for t in rows]


def _snapshot_rows(db: Session, device_sn: str) -> list[SegmentSnapshot]:
    return list(
        db.scalars(
            select(SegmentSnapshot)
            .where(SegmentSnapshot.device_sn == device_sn)
            .order_by(SegmentSnapshot.segment_index.asc())
        ).all()
    )


def serialize_result(
    result: seg.SegmentationResult,
    version: int,
    replayed: bool,
    replayed_at: dt.datetime | None,
) -> SegmentationOut:
    return SegmentationOut(
        device_sn=result.device_sn,
        version=version,
        penalty=result.penalty,
        sample_count=result.sample_count,
        anchor_count=result.anchor_count,
        total_cost=result.total_cost,
        replayed=replayed,
        replayed_at=replayed_at,
        unmatched_anchors=list(result.unmatched_anchors),
        segments=[
            SegmentOut(
                index=s.index,
                start_ts=s.start_ts,
                end_ts=s.end_ts,
                sample_count=s.sample_count,
                mean_temperature=s.mean_temperature,
                mean_humidity=s.mean_humidity,
                temperature_std=s.temperature_std,
                humidity_std=s.humidity_std,
                delta_temperature=s.delta_temperature,
                delta_humidity=s.delta_humidity,
                magnitude=s.magnitude,
                confidence=s.confidence,
                start_anchored=s.start_anchored,
                end_anchored=s.end_anchored,
                cost=s.cost,
            )
            for s in result.segments
        ],
    )


def get_stored_segmentation(db: Session, device_sn: str) -> SegmentationOut | None:
    state = db.get(CaseState, device_sn)
    rows = _snapshot_rows(db, device_sn)
    if state is None or not rows:
        return None
    return SegmentationOut(
        device_sn=device_sn,
        version=state.version,
        penalty=state.penalty,
        sample_count=state.reading_count,
        anchor_count=_count_anchors(db, device_sn),
        total_cost=sum(r.cost for r in rows),
        replayed=False,
        replayed_at=to_utc(state.replayed_at) if state.replayed_at else None,
        unmatched_anchors=[],
        segments=[_snapshot_to_out(r, i) for i, r in enumerate(rows)],
    )


def _snapshot_to_out(r: SegmentSnapshot, i: int) -> SegmentOut:
    return SegmentOut(
        index=i,
        start_ts=to_utc(r.start_ts),
        end_ts=to_utc(r.end_ts),
        sample_count=r.sample_count,
        mean_temperature=r.mean_temperature,
        mean_humidity=r.mean_humidity,
        temperature_std=0.0,
        humidity_std=0.0,
        delta_temperature=r.delta_temperature,
        delta_humidity=r.delta_humidity,
        magnitude=r.magnitude,
        confidence=r.confidence,
        start_anchored=bool(r.start_anchored),
        end_anchored=bool(r.end_anchored),
        cost=r.cost,
    )


def _count_anchors(db: Session, device_sn: str) -> int:
    return int(
        db.scalar(select(func.count(Anchor.id)).where(Anchor.device_sn == device_sn)) or 0
    )


# ---------- 重复上报合并 ----------


def merge_readings(
    db: Session, readings: list[ReadingIn]
) -> dict[str, dict[str, int]]:
    """按 (device_sn, ts) 幂等合并。

    - 新时间戳：直接插入；
    - 已存在：以 report_count 为权把温湿度并入累计均值，
      report_count 加一（确定性的合并规则，重复重放结果稳定）。

    返回 {device_sn: {"inserted": n, "merged": n}}。
    """
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"inserted": 0, "merged": 0})

    existing: dict[tuple[str, dt.datetime], Reading] = {}
    device_sns = {r.device_sn for r in readings}
    for sn in device_sns:
        for row in db.scalars(select(Reading).where(Reading.device_sn == sn)).all():
            existing[(sn, to_utc(row.ts))] = row

    for item in readings:
        key = (item.device_sn, item.ts)
        row = existing.get(key)
        if row is None:
            row = Reading(
                device_sn=item.device_sn,
                ts=item.ts,
                temperature=item.temperature,
                humidity=item.humidity,
                report_count=1,
            )
            db.add(row)
            existing[key] = row
            stats[item.device_sn]["inserted"] += 1
        else:
            total = row.report_count + 1
            row.temperature = (row.temperature * row.report_count + item.temperature) / total
            row.humidity = (row.humidity * row.report_count + item.humidity) / total
            row.report_count = total
            stats[item.device_sn]["merged"] += 1

    db.flush()
    return stats


# ---------- 重放 ----------


def _compute(
    db: Session, device_sn: str, penalty: float | None = None
) -> seg.SegmentationResult:
    samples = _load_samples(db, device_sn)
    anchors = _load_anchor_ts(db, device_sn)
    beta = penalty if penalty is not None else settings.PENALTY
    state = db.get(CaseState, device_sn)
    if penalty is None and state is not None:
        beta = state.penalty
    return seg.segment_samples(device_sn, samples, anchors, penalty=beta)


def _persist(db: Session, result: seg.SegmentationResult, state: CaseState) -> None:
    db.execute(delete(SegmentSnapshot).where(SegmentSnapshot.device_sn == result.device_sn))
    for s in result.segments:
        db.add(
            SegmentSnapshot(
                device_sn=result.device_sn,
                version=state.version,
                segment_index=s.index,
                start_ts=s.start_ts,
                end_ts=s.end_ts,
                sample_count=s.sample_count,
                mean_temperature=s.mean_temperature,
                mean_humidity=s.mean_humidity,
                delta_temperature=s.delta_temperature,
                delta_humidity=s.delta_humidity,
                magnitude=s.magnitude,
                confidence=s.confidence,
                start_anchored=s.start_anchored,
                end_anchored=s.end_anchored,
                cost=s.cost,
            )
        )


def replay_device(
    db: Session,
    device_sn: str,
    penalty: float | None = None,
    changed: bool = True,
) -> SegmentationOut:
    """对单个展柜整条时间线执行确定性重放并落库。

    ``changed=False`` 时仅重新计算并刷新快照（用于首次生成等场景），
    不增加版本号；正常的数据/锚点变更调用应保持 ``changed=True``。
    """
    result = _compute(db, device_sn, penalty=penalty)

    state = db.get(CaseState, device_sn)
    if state is None:
        state = CaseState(device_sn=device_sn, version=0, penalty=result.penalty)
        db.add(state)
    if penalty is not None:
        state.penalty = result.penalty
    if changed:
        # 读数/锚点发生变化；版本号单调递增
        state.version += 1
    state.reading_count = result.sample_count
    state.replayed_at = utcnow()

    _persist(db, result, state)
    db.flush()
    return serialize_result(result, state.version, True, to_utc(state.replayed_at))


def recompute_if_needed(
    db: Session, device_sn: str, penalty: float | None = None
) -> SegmentationOut | None:
    """读数有新增/合并时重放；空展柜不产生版本。"""
    count = db.scalar(
        select(func.count(Reading.id)).where(Reading.device_sn == device_sn)
    ) or 0
    if count == 0:
        return None
    return replay_device(db, device_sn, penalty=penalty)


# ---------- 锚点 ----------


def upsert_anchor(db: Session, payload: AnchorIn) -> Anchor:
    existing = db.scalar(
        select(Anchor).where(
            Anchor.device_sn == payload.device_sn, Anchor.ts == payload.ts
        )
    )
    if existing is None:
        existing = Anchor(device_sn=payload.device_sn, ts=payload.ts)
        db.add(existing)
    existing.label = payload.label
    existing.ref_temperature = payload.ref_temperature
    existing.ref_humidity = payload.ref_humidity
    _align_anchor(db, existing)
    db.flush()
    return existing


def _align_anchor(db: Session, anchor: Anchor) -> None:
    nearest = db.scalars(
        select(Reading)
        .where(Reading.device_sn == anchor.device_sn)
        .order_by(func.abs(func.julianday(Reading.ts) - func.julianday(anchor.ts)))
    ).first()
    if (
        nearest is not None
        and abs((to_utc(nearest.ts) - to_utc(anchor.ts)).total_seconds())
        <= settings.ANCHOR_TOLERANCE
    ):
        anchor.reading_id = nearest.id
    else:
        anchor.reading_id = None


def list_anchors(db: Session, device_sn: str) -> list[dict]:
    rows = db.scalars(
        select(Anchor).where(Anchor.device_sn == device_sn).order_by(Anchor.ts.asc())
    ).all()
    out = []
    for a in rows:
        aligned_ts = to_utc(a.reading.ts) if a.reading is not None else None
        out.append(
            {
                "device_sn": a.device_sn,
                "ts": to_utc(a.ts),
                "label": a.label,
                "ref_temperature": a.ref_temperature,
                "ref_humidity": a.ref_humidity,
                "aligned": a.reading_id is not None,
                "aligned_ts": aligned_ts,
            }
        )
    return out


def delete_anchor(db: Session, device_sn: str, ts: dt.datetime) -> bool:
    anchor = db.scalar(
        select(Anchor).where(Anchor.device_sn == device_sn, Anchor.ts == ts)
    )
    if anchor is None:
        return False
    db.delete(anchor)
    db.flush()
    return True


def list_cases(db: Session) -> list[CaseState]:
    return list(db.scalars(select(CaseState).order_by(CaseState.device_sn.asc())).all())


# ---------- 时间线降采样 ----------


def get_effective_segmentation(db: Session, device_sn: str) -> SegmentationOut | None:
    """已落库的分段快照；展柜尚未重放时有读数则即时计算（只读，不落库）。"""
    stored = get_stored_segmentation(db, device_sn)
    if stored is not None:
        return stored
    samples = _load_samples(db, device_sn)
    if not samples:
        return None
    anchors = _load_anchor_ts(db, device_sn)
    result = seg.segment_samples(device_sn, samples, anchors)
    return serialize_result(result, version=0, replayed=False, replayed_at=None)


def build_case_timeline(
    db: Session,
    device_sn: str,
    start: dt.datetime | None,
    end: dt.datetime | None,
    max_points: int,
) -> TimelineOut | None:
    """过滤后对温湿度时间线执行“强制点 + 比例配额 + 确定性 LTTB”。

    设备没有任何读数时返回 ``None``（由上层判 404）；有读数但过滤后
    为空范围时返回空时间线。极值统计与名额预算均只基于过滤后范围；
    强制点去重后超过 ``max_points`` 时抛
    :class:`timeline.ForcedPointsOverflow`。
    """
    all_samples = _load_samples(db, device_sn)
    if not all_samples:
        return None
    samples = all_samples
    if start is not None:
        samples = [s for s in samples if s.ts >= start]
    if end is not None:
        samples = [s for s in samples if s.ts <= end]

    points = [tl.TimelinePoint(ts=s.ts, temperature=s.temperature, humidity=s.humidity)
              for s in samples]
    n = len(points)
    forced: dict[int, set[str]] = {}

    def mark(i: int, reason: str) -> None:
        forced.setdefault(i, set()).add(reason)

    if n:
        # 范围首尾（samples 已按 ts 升序）
        mark(0, tl.REASON_RANGE_START)
        mark(n - 1, tl.REASON_RANGE_END)

        # 温/湿度全局极值；并列时取最早出现（升序扫描 + 严格不等号）
        tmin_i = tmax_i = hmin_i = hmax_i = 0
        for i in range(1, n):
            if points[i].temperature < points[tmin_i].temperature:
                tmin_i = i
            if points[i].temperature > points[tmax_i].temperature:
                tmax_i = i
            if points[i].humidity < points[hmin_i].humidity:
                hmin_i = i
            if points[i].humidity > points[hmax_i].humidity:
                hmax_i = i
        mark(tmin_i, tl.REASON_TEMP_MIN)
        mark(tmax_i, tl.REASON_TEMP_MAX)
        mark(hmin_i, tl.REASON_HUMIDITY_MIN)
        mark(hmax_i, tl.REASON_HUMIDITY_MAX)

        segmentation = get_effective_segmentation(db, device_sn)
        assert segmentation is not None  # 有读数则必有分段结果

        ts_index = {p.ts: i for i, p in enumerate(points)}

        # 已对齐锚点所在读数（ts_index 已只含过滤后范围，天然受窗口约束）
        for a in list_anchors(db, device_sn):
            if a["aligned"] is True and a["aligned_ts"] in ts_index:
                mark(ts_index[a["aligned_ts"]], tl.REASON_ALIGNED_ANCHOR)

        # 过滤后仍与范围相交的分段，其两端读数作为分段边界
        for s in segmentation.segments:
            if start is not None and s.end_ts < start:
                continue
            if end is not None and s.start_ts > end:
                continue
            if s.start_ts in ts_index:
                mark(ts_index[s.start_ts], tl.REASON_SEGMENT_BOUNDARY)
            if s.end_ts in ts_index:
                mark(ts_index[s.end_ts], tl.REASON_SEGMENT_BOUNDARY)

    chosen = tl.downsample_timeline(points, forced, max_points)
    return TimelineOut(
        device_sn=device_sn,
        start=start,
        end=end,
        max_points=max_points,
        original_count=n,
        returned_count=len(chosen),
        points=[
            TimelinePointOut(
                ts=points[i].ts,
                temperature=points[i].temperature,
                humidity=points[i].humidity,
                reasons=reasons,
            )
            for i, reasons in chosen
        ],
    )


# ---------- 锚点参考值校正时间线 ----------


def _load_aligned_anchors(db: Session, device_sn: str) -> list[cal.AlignedAnchor]:
    """已对齐到读数的锚点（按锚点时间升序），附带对齐读数的当前原始值。"""
    rows = db.scalars(
        select(Anchor)
        .where(Anchor.device_sn == device_sn, Anchor.reading_id.is_not(None))
        .order_by(Anchor.ts.asc())
    ).all()
    out: list[cal.AlignedAnchor] = []
    for a in rows:
        reading = a.reading
        if reading is None:  # 对齐目标已不存在（防御；外键 SET NULL 时不会出现）
            continue
        out.append(
            cal.AlignedAnchor(
                anchor_ts=to_utc(a.ts),
                reading_ts=to_utc(reading.ts),
                reading_temperature=reading.temperature,
                reading_humidity=reading.humidity,
                ref_temperature=a.ref_temperature,
                ref_humidity=a.ref_humidity,
            )
        )
    return out


def build_case_calibrated_timeline(
    db: Session,
    device_sn: str,
    start: dt.datetime | None,
    end: dt.datetime | None,
) -> CalibratedTimelineOut | None:
    """基于已对齐锚点参考值的只读校正时间线（不落库、不改版本）。

    设备没有任何读数时返回 ``None``（由上层判 404）；有读数但过滤后
    为空范围时返回空序列。偏移曲线先基于整条时间线建立，再按
    start/end 过滤输出，窗口外的锚点仍参与窗口内读数的偏移。
    """
    samples = _load_samples(db, device_sn)
    if not samples:
        return None
    anchors = _load_aligned_anchors(db, device_sn)

    points = cal.calibrate_timeline(samples, anchors)
    if start is not None:
        points = [p for p in points if p.ts >= start]
    if end is not None:
        points = [p for p in points if p.ts <= end]

    return CalibratedTimelineOut(
        device_sn=device_sn,
        start=start,
        end=end,
        sample_count=len(points),
        aligned_anchor_count=len(anchors),
        points=[
            CalibratedPointOut(
                ts=p.ts,
                temperature=p.temperature,
                humidity=p.humidity,
                temperature_offset=p.temperature_offset,
                humidity_offset=p.humidity_offset,
                calibrated_temperature=p.calibrated_temperature,
                calibrated_humidity=p.calibrated_humidity,
            )
            for p in points
        ],
    )


# ---------- 越界剂量 ----------


def build_case_excursions(
    db: Session,
    device_sn: str,
    start: dt.datetime | None,
    end: dt.datetime | None,
    thresholds: exc.Thresholds,
    max_gap_seconds: float,
) -> ExcursionsOut | None:
    """越界剂量事件（只读计算，不落库）。

    设备没有任何读数时返回 ``None``（由上层判 404）；有读数但过滤后
    为空范围时返回空事件列表。剂量只基于过滤后范围内的分段线性曲线。
    """
    all_samples = _load_samples(db, device_sn)
    if not all_samples:
        return None
    samples = all_samples
    if start is not None:
        samples = [s for s in samples if s.ts >= start]
    if end is not None:
        samples = [s for s in samples if s.ts <= end]

    events = exc.compute_excursions(samples, thresholds, max_gap_seconds)
    return ExcursionsOut(
        device_sn=device_sn,
        start=start,
        end=end,
        max_gap_seconds=max_gap_seconds,
        temp_min=thresholds.temp_min,
        temp_max=thresholds.temp_max,
        humidity_min=thresholds.humidity_min,
        humidity_max=thresholds.humidity_max,
        sample_count=len(samples),
        event_count=len(events),
        events=[
            ExcursionEventOut(
                start_ts=e.start_ts,
                end_ts=e.end_ts,
                dimensions=list(e.dimensions),
                duration_seconds=e.duration_seconds,
                temp_dose=e.temp_dose,
                humidity_dose=e.humidity_dose,
                peak_temperature=e.peak_temperature,
                peak_temperature_ts=e.peak_temperature_ts,
                peak_humidity=e.peak_humidity,
                peak_humidity_ts=e.peak_humidity_ts,
            )
            for e in events
        ],
    )


# ---------- 累计剂量预算 ----------


def build_case_exposure_budget(
    db: Session,
    device_sn: str,
    start: dt.datetime | None,
    end: dt.datetime | None,
    thresholds: exc.Thresholds,
    budgets: dict[str, float],
    max_gap_seconds: float,
) -> ExposureBudgetOut | None:
    """窗末累计越界剂量、剩余额度与首次达到预算时刻（只读，不落库）。

    设备没有任何读数时返回 ``None``（由上层判 404）；有读数但过滤后
    为空范围时返回各维零累计、预算未耗尽。阈值交点与断档规则与
    :func:`build_case_excursions` 完全一致。
    """
    all_samples = _load_samples(db, device_sn)
    if not all_samples:
        return None
    samples = all_samples
    if start is not None:
        samples = [s for s in samples if s.ts >= start]
    if end is not None:
        samples = [s for s in samples if s.ts <= end]

    result = exc.compute_exposure_budget(samples, thresholds, budgets, max_gap_seconds)
    return ExposureBudgetOut(
        device_sn=device_sn,
        start=start,
        end=end,
        max_gap_seconds=max_gap_seconds,
        temp_min=thresholds.temp_min,
        temp_max=thresholds.temp_max,
        humidity_min=thresholds.humidity_min,
        humidity_max=thresholds.humidity_max,
        temp_budget=budgets.get(exc.TEMPERATURE),
        humidity_budget=budgets.get(exc.HUMIDITY),
        sample_count=len(samples),
        dimensions=[
            ExposureBudgetDimensionOut(
                dimension=d.dimension,
                budget=d.budget,
                cumulative_dose=d.cumulative_dose,
                remaining=d.remaining,
                budget_exhausted=d.exhausted_ts is not None,
                exhausted_ts=d.exhausted_ts,
            )
            for d in result.dimensions
        ],
    )


# ---------- 冷凝风险 ----------


def build_case_condensation_risk(
    db: Session,
    device_sn: str,
    start: dt.datetime | None,
    end: dt.datetime | None,
    surface_temperature_c: float,
    margin_threshold_c: float,
    max_gap_seconds: float,
    min_duration_seconds: float,
) -> CondensationRiskOut | None:
    """Magnus 露点 + 冷凝裕量风险窗口（只读计算，不落库）。

    设备没有任何读数时返回 ``None``（由上层判 404）；有读数但过滤后
    为空范围时返回空窗口列表。湿度为 0 的读数计入 ``sample_count``
    但不参与露点计算，并强制其两侧曲线断开。
    """
    all_samples = _load_samples(db, device_sn)
    if not all_samples:
        return None
    samples = all_samples
    if start is not None:
        samples = [s for s in samples if s.ts >= start]
    if end is not None:
        samples = [s for s in samples if s.ts <= end]

    windows = cond.compute_condensation_windows(
        samples,
        surface_temperature_c=surface_temperature_c,
        margin_threshold_c=margin_threshold_c,
        max_gap_seconds=max_gap_seconds,
        min_duration_seconds=min_duration_seconds,
    )
    return CondensationRiskOut(
        device_sn=device_sn,
        start=start,
        end=end,
        surface_temperature_c=surface_temperature_c,
        margin_threshold_c=margin_threshold_c,
        max_gap_seconds=max_gap_seconds,
        min_duration_seconds=min_duration_seconds,
        sample_count=len(samples),
        window_count=len(windows),
        windows=[
            CondensationWindowOut(
                start_ts=w.start_ts,
                end_ts=w.end_ts,
                duration_seconds=w.duration_seconds,
                min_margin=w.min_margin,
                min_margin_ts=w.min_margin_ts,
            )
            for w in windows
        ],
    )


# ---------- 展柜扰动恢复分析 ----------


def build_case_recovery(
    db: Session,
    device_sn: str,
    event_ts: dt.datetime,
    baseline_seconds: float,
    horizon_seconds: float,
    min_baseline_samples: int,
    temp_tolerance: float,
    humidity_tolerance: float,
    hold_seconds: float,
    max_gap_seconds: float,
) -> RecoveryOut:
    """扰动恢复分析（只读计算，不落库）。

    设备没有任何读数时由调用方通过 ``_case_has_readings`` 判 404；
    基线样本不足时透传 :class:`recovery.InsufficientBaselineSamples`
    由接口层判 422。
    """
    samples = _load_samples(db, device_sn)
    result = rec.analyze_recovery(
        samples,
        event_ts=event_ts,
        baseline_seconds=baseline_seconds,
        horizon_seconds=horizon_seconds,
        min_baseline_samples=min_baseline_samples,
        temp_tolerance=temp_tolerance,
        humidity_tolerance=humidity_tolerance,
        hold_seconds=hold_seconds,
        max_gap_seconds=max_gap_seconds,
    )
    return RecoveryOut(
        device_sn=device_sn,
        event_ts=result.event_ts,
        horizon_end=result.horizon_end,
        baseline_seconds=baseline_seconds,
        horizon_seconds=horizon_seconds,
        min_baseline_samples=min_baseline_samples,
        temp_tolerance=temp_tolerance,
        humidity_tolerance=humidity_tolerance,
        hold_seconds=hold_seconds,
        max_gap_seconds=max_gap_seconds,
        baseline_sample_count=result.baseline.sample_count,
        post_event_sample_count=result.post_event_sample_count,
        baseline=RecoveryBaselineOut(
            window_start=result.baseline.window_start,
            window_end=result.baseline.window_end,
            sample_count=result.baseline.sample_count,
            temperature=result.baseline.temperature,
            humidity=result.baseline.humidity,
            temperature_low=result.baseline.temperature_low,
            temperature_high=result.baseline.temperature_high,
            humidity_low=result.baseline.humidity_low,
            humidity_high=result.baseline.humidity_high,
        ),
        recovered=result.recovered,
        band_enter_ts=result.band_enter_ts,
        recovered_at=result.recovered_at,
        recovery_seconds=result.recovery_seconds,
        max_temperature_deviation=result.max_temperature_deviation,
        max_temperature_deviation_ts=result.max_temperature_deviation_ts,
        max_humidity_deviation=result.max_humidity_deviation,
        max_humidity_deviation_ts=result.max_humidity_deviation_ts,
        reason=result.reason,
    )


# ---------- 展柜采集质量审计：孤立异常值 ----------


def build_case_reading_outliers(
    db: Session,
    device_sn: str,
    window_size: int,
    k: float,
    min_temperature_delta: float,
    min_humidity_delta: float,
    max_gap_seconds: float,
) -> ReadingOutliersOut | None:
    """基于已合并读数的邻点中位数/MAD 孤立异常值定位（只读，不落库）。

    设备没有任何读数时返回 ``None``（由上层判 404）。断档切段、邻点窗、
    尺度与命中规则全部委托 :mod:`app.outliers`。
    """
    samples = _load_samples(db, device_sn)
    if not samples:
        return None

    result = ol.compute_reading_outliers(
        samples,
        window_size=window_size,
        k=k,
        min_temperature_delta=min_temperature_delta,
        min_humidity_delta=min_humidity_delta,
        max_gap_seconds=max_gap_seconds,
    )
    return ReadingOutliersOut(
        device_sn=device_sn,
        window_size=window_size,
        k=k,
        min_temperature_delta=min_temperature_delta,
        min_humidity_delta=min_humidity_delta,
        max_gap_seconds=max_gap_seconds,
        sample_count=result.sample_count,
        segment_count=result.segment_count,
        evaluated_count=result.evaluated_count,
        outlier_count=len(result.outliers),
        outliers=[
            ReadingOutlierOut(
                ts=o.ts,
                temperature=o.temperature,
                humidity=o.humidity,
                dimensions=[
                    ReadingOutlierDimensionOut(
                        dimension=d.dimension,
                        median=d.median,
                        deviation=d.deviation,
                    )
                    for d in o.dimensions
                ],
            )
            for o in result.outliers
        ],
    )


# ---------- 采样覆盖率审计 ----------


def build_case_sampling_coverage(
    db: Session,
    device_sn: str,
    start: dt.datetime,
    end: dt.datetime,
    resolved: samp.ResolvedSamplingParams,
) -> SamplingCoverageOut | None:
    """周期时隙采样覆盖率与连续缺测缺口（只读计算，不落库）。

    设备没有任何读数时返回 ``None``（由上层判 404）。入参校验、微秒
    归一化与时隙数量预估均已由 :func:`samp.resolve_sampling_params`
    完成；容差扩展范围过滤与一对一贪心匹配委托 :mod:`app.sampling`。
    """
    samples = _load_samples(db, device_sn)
    if not samples:
        return None

    result = samp.compute_sampling_coverage(
        samples,
        start=start,
        end=end,
        resolved=resolved,
    )
    return SamplingCoverageOut(
        device_sn=device_sn,
        start=start,
        end=end,
        period_seconds=resolved.period_effective_seconds,
        tolerance_seconds=resolved.tolerance_effective_seconds,
        expected_count=result.expected_count,
        covered_count=result.covered_count,
        coverage=result.coverage,
        gap_count=len(result.gaps),
        gaps=[
            SamplingGapOut(
                start_ts=g.start_ts,
                end_ts=g.end_ts,
                missing_slots=g.missing_slots,
            )
            for g in result.gaps
        ],
    )


# ---------- 跨展柜共模突变 ----------


def list_device_sns(db: Session) -> list[str]:
    """库内有读数的全部展柜序号（即便尚未重放、没有 CaseState 也包含）。"""
    rows = db.scalars(select(Reading.device_sn).distinct()).all()
    return sorted(rows)


def build_incidents(
    db: Session,
    start: dt.datetime,
    end: dt.datetime,
    tolerance_seconds: float,
    min_devices: int,
) -> IncidentsOut:
    """基于各展柜现有有效分段的非基线段首，检测跨展柜共模突变。

    段首取自已落库分段快照；展柜尚未重放时即时只读计算（不落库）。
    段首时间戳落在闭区间 ``[start, end]`` 内才参与检测。候选枚举、
    同展柜取最大幅度边界与全局最优选择全部委托 :mod:`app.incidents`。
    """
    boundaries: list[inc.Boundary] = []
    for device_sn in list_device_sns(db):
        segmentation = get_effective_segmentation(db, device_sn)
        if segmentation is None:
            continue
        for s in segmentation.segments:
            if s.confidence == "baseline":
                continue  # 只取非基线段首
            if start <= s.start_ts <= end:
                boundaries.append(
                    inc.Boundary(
                        device_sn=device_sn,
                        ts=s.start_ts,
                        magnitude=s.magnitude,
                        delta_temperature=s.delta_temperature,
                        delta_humidity=s.delta_humidity,
                    )
                )

    candidates = inc.enumerate_incidents(
        boundaries, tolerance_seconds, min_devices
    )
    selected = inc.select_incidents(candidates)

    return IncidentsOut(
        start=start,
        end=end,
        tolerance_seconds=tolerance_seconds,
        min_devices=min_devices,
        candidate_count=len(candidates),
        boundary_count=len(boundaries),
        incident_count=len(selected),
        incidents=[
            IncidentOut(
                start_ts=e.start_ts,
                end_ts=e.end_ts,
                duration_seconds=e.duration_seconds,
                device_count=e.device_count,
                total_magnitude=e.total_magnitude,
                boundaries=[
                    IncidentBoundaryOut(
                        device_sn=b.device_sn,
                        ts=b.ts,
                        direction=inc.boundary_direction(
                            b.delta_temperature, b.delta_humidity
                        ),
                        magnitude=b.magnitude,
                        delta_temperature=b.delta_temperature,
                        delta_humidity=b.delta_humidity,
                    )
                    for b in e.members
                ],
            )
            for e in selected
        ],
    )


# ---------- 跨展柜环境传播分析 ----------


def _case_has_readings(db: Session, device_sn: str) -> bool:
    return (
        db.scalar(
            select(func.count(Reading.id)).where(Reading.device_sn == device_sn)
        )
        or 0
    ) > 0


def build_propagation(
    db: Session,
    source_sn: str,
    target_sn: str,
    dimension: str,
    start: dt.datetime,
    end: dt.datetime,
    step_seconds: float,
    max_lag_seconds: float,
    max_gap_seconds: float,
) -> PropagationOut | None:
    """两展柜重叠时段的分维固定步长插值、时滞枚举与皮尔逊相关（只读）。

    任一展柜没有任何读数时返回 ``None``（由上层判 404）。两柜在查询窗内
    无重叠、或全部时滞样本不足/常量时，相关字段为 ``null``、方向为
    ``unavailable``，不产生 NaN。步长/时滞非法（非有限、正步长微秒取整
    后为 0）或预估规模超限时抛 :class:`prop.PropagationParamsError`
    （由上层转 422），且在分配网格 / 明细之前抛出。
    """
    source_samples = _load_samples(db, source_sn)
    target_samples = _load_samples(db, target_sn)
    if not source_samples or not target_samples:
        return None

    result = prop.compute_propagation(
        source_samples,
        target_samples,
        dimension=dimension,
        start=start,
        end=end,
        step_seconds=step_seconds,
        max_lag_seconds=max_lag_seconds,
        max_gap_seconds=max_gap_seconds,
    )
    return PropagationOut(
        source_sn=source_sn,
        target_sn=target_sn,
        dimension=result.dimension,
        start=result.start,
        end=result.end,
        overlap_start=result.overlap_start,
        overlap_end=result.overlap_end,
        step_seconds=result.step_seconds,
        max_lag_seconds=result.max_lag_seconds,
        max_gap_seconds=result.max_gap_seconds,
        grid_point_count=result.grid_point_count,
        source_valid_count=result.source_valid_count,
        target_valid_count=result.target_valid_count,
        lags=[
            PropagationLagOut(
                lag_seconds=item.lag_seconds,
                available=item.available,
                pair_count=item.pair_count,
                correlation=item.correlation,
                reason=item.reason,
            )
            for item in result.lags
        ],
        best_lag_seconds=result.best_lag_seconds,
        best_correlation=result.best_correlation,
        direction=result.direction,
    )


# ---------- 展柜同伴基线偏离 ----------


def build_peer_baseline_deviation(
    db: Session,
    target_sn: str,
    peer_sns: list[str],
    dimension: str,
    start: dt.datetime,
    end: dt.datetime,
    step_seconds: float,
    max_gap_seconds: float,
    deviation_threshold: float,
    min_duration_seconds: float,
) -> PeerBaselineDeviationOut | None:
    """目标展柜相对对照中位数基线的持续偏离时段（只读计算，不落库）。

    目标展柜没有任何读数时返回 ``None``（由上层判 404）；对照展柜
    没有读数不报错——该柜在全部网格时刻无效，有效对照值不足两个的
    时刻中断判定，自然得到空结果。
    """
    target_samples = _load_samples(db, target_sn)
    if not target_samples:
        return None
    peer_samples = [_load_samples(db, sn) for sn in peer_sns]

    result = dev.compute_peer_baseline_deviation(
        target_samples,
        peer_samples,
        dimension=dimension,
        start=start,
        end=end,
        step_seconds=step_seconds,
        max_gap_seconds=max_gap_seconds,
        deviation_threshold=deviation_threshold,
        min_duration_seconds=min_duration_seconds,
    )
    return PeerBaselineDeviationOut(
        target_sn=target_sn,
        peer_sns=list(peer_sns),
        dimension=result.dimension,
        start=result.start,
        end=result.end,
        step_seconds=result.step_seconds,
        max_gap_seconds=result.max_gap_seconds,
        deviation_threshold=result.deviation_threshold,
        min_duration_seconds=result.min_duration_seconds,
        grid_point_count=result.grid_point_count,
        evaluated_count=result.evaluated_count,
        marked_count=result.marked_count,
        period_count=len(result.periods),
        periods=[
            DeviationPeriodOut(
                start_ts=p.start_ts,
                end_ts=p.end_ts,
                duration_seconds=p.duration_seconds,
                marked_count=p.marked_count,
                direction=p.direction,
                max_deviation=p.max_deviation,
                peak_ts=p.peak_ts,
            )
            for p in result.periods
        ],
    )


def affected_window(
    current: SegmentationOut | None, preview: SegmentationOut
) -> tuple[dt.datetime | None, dt.datetime | None, bool]:
    """比较已存储分段与预览分段，给出受影响时间范围。

    迟到读数补入后，从第一个发生变化的段起点起，到时间线末尾都视为
    受影响（后续均值会被重新估计）。
    """
    if current is None or not current.segments:
        if preview.segments:
            return (
                preview.segments[0].start_ts,
                preview.segments[-1].end_ts,
                True,
            )
        return None, None, False

    changed = _segmentation_differs(current, preview)
    if not changed:
        return None, None, False

    first_diff = preview.segments[0].start_ts
    for cur, new in zip(current.segments, preview.segments):
        if (
            cur.start_ts != new.start_ts
            or cur.end_ts != new.end_ts
            or cur.sample_count != new.sample_count
            or abs(cur.mean_temperature - new.mean_temperature) > 1e-9
            or abs(cur.mean_humidity - new.mean_humidity) > 1e-9
        ):
            first_diff = new.start_ts
            break
    else:  # 长度相同但存在其他差异（如惩罚项）
        first_diff = preview.segments[-1].start_ts

    return first_diff, preview.segments[-1].end_ts, True


def _segmentation_differs(a: SegmentationOut, b: SegmentationOut) -> bool:
    if a.penalty != b.penalty:
        return True
    sa, sb = a.segments, b.segments
    if len(sa) != len(sb):
        return True
    for x, y in zip(sa, sb):
        if (
            x.start_ts != y.start_ts
            or x.end_ts != y.end_ts
            or x.sample_count != y.sample_count
            or abs(x.mean_temperature - y.mean_temperature) > 1e-9
            or abs(x.mean_humidity - y.mean_humidity) > 1e-9
        ):
            return True
    return False
