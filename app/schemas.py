"""请求与响应模型（OpenAPI 由此生成）。"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Confidence = Literal["baseline", "high", "medium", "low"]


def _to_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


class ReadingIn(BaseModel):
    device_sn: str = Field(..., min_length=1, max_length=64, examples=["CASE-007"])
    ts: dt.datetime = Field(..., description="ISO8601 时间戳；无时区按 UTC 处理", examples=["2026-09-01T08:00:00+08:00"])
    temperature: float = Field(..., ge=-50.0, le=100.0, examples=[20.4])
    humidity: float = Field(..., ge=0.0, le=100.0, examples=[55.2])

    @field_validator("ts")
    @classmethod
    def _utc(cls, v: dt.datetime) -> dt.datetime:
        return _to_utc(v)


class BatchIngest(BaseModel):
    readings: list[ReadingIn] = Field(..., min_length=1, max_length=10000)
    penalty: float | None = Field(None, ge=0.0, description="本次重放使用的惩罚项；缺省沿用展柜已保存值或默认值")
    replay: bool = Field(True, description="入库后是否立即重放受影响展柜的时间线")


class AnchorIn(BaseModel):
    device_sn: str = Field(..., min_length=1, max_length=64)
    ts: dt.datetime
    label: str | None = Field(None, max_length=128)
    ref_temperature: float | None = Field(None, ge=-50.0, le=100.0)
    ref_humidity: float | None = Field(None, ge=0.0, le=100.0)

    @field_validator("ts")
    @classmethod
    def _utc(cls, v: dt.datetime) -> dt.datetime:
        return _to_utc(v)


class SegmentOut(BaseModel):
    index: int
    start_ts: dt.datetime
    end_ts: dt.datetime
    sample_count: int
    mean_temperature: float
    mean_humidity: float
    temperature_std: float
    humidity_std: float
    delta_temperature: float
    delta_humidity: float
    magnitude: float = Field(..., description="缩放加权后的欧氏变化幅度")
    confidence: Confidence
    start_anchored: bool
    end_anchored: bool
    cost: float


class AnchorOut(BaseModel):
    device_sn: str
    ts: dt.datetime
    label: str | None
    ref_temperature: float | None
    ref_humidity: float | None
    aligned: bool
    aligned_ts: dt.datetime | None = Field(None, description="实际对齐到的读数时间戳")


class SegmentationOut(BaseModel):
    device_sn: str
    version: int
    penalty: float
    sample_count: int
    anchor_count: int
    total_cost: float
    replayed: bool
    replayed_at: dt.datetime | None
    unmatched_anchors: list[dt.datetime]
    segments: list[SegmentOut]


class IngestSummary(BaseModel):
    device_sn: str
    inserted: int
    merged: int
    affected: bool = Field(..., description="该展柜是否被重放（存在新读数或合并）")
    before_version: int
    after_version: int
    replay: "SegmentationOut | None" = None


class IngestResponse(BaseModel):
    accepted: int
    inserted: int
    merged: int
    replayed_devices: list[str]
    summaries: list[IngestSummary]


class CaseOut(BaseModel):
    device_sn: str
    version: int
    penalty: float
    reading_count: int
    anchor_count: int
    replayed_at: dt.datetime | None


class ReplayPreviewResponse(BaseModel):
    device_sn: str
    current_version: int
    would_bump_version: bool
    changed: bool = Field(..., description="预览分段与当前已存储分段是否不同")
    penalty: float
    added_readings: int
    added_anchors: int
    affected_from_ts: dt.datetime | None
    affected_to_ts: dt.datetime | None
    preview: SegmentationOut
    current: "SegmentationOut | None" = None


TimelineReason = Literal[
    "range_start",
    "range_end",
    "temp_min",
    "temp_max",
    "humidity_min",
    "humidity_max",
    "aligned_anchor",
    "segment_boundary",
    "lttb",
]


class TimelinePointOut(BaseModel):
    ts: dt.datetime
    temperature: float
    humidity: float
    reasons: list[TimelineReason] = Field(
        ..., description="该点被保留的全部原因，按固定顺序排列（可多个）"
    )


class TimelineOut(BaseModel):
    device_sn: str
    start: dt.datetime | None = Field(None, description="实际生效的起始过滤（UTC），未指定为 null")
    end: dt.datetime | None = Field(None, description="实际生效的结束过滤（UTC），未指定为 null")
    max_points: int
    original_count: int = Field(..., description="过滤后、降采样前的读数数量")
    returned_count: int
    points: list[TimelinePointOut]


class CalibratedPointOut(BaseModel):
    ts: dt.datetime
    temperature: float = Field(..., description="原始温度读数（°C）")
    humidity: float = Field(..., description="原始湿度读数（%RH）")
    temperature_offset: float = Field(
        ..., description="该时刻温度偏移（参考值曲线插值结果，°C）"
    )
    humidity_offset: float = Field(
        ..., description="该时刻湿度偏移（参考值曲线插值结果，%RH）"
    )
    calibrated_temperature: float = Field(..., description="校正温度 = 原值 + 温度偏移")
    calibrated_humidity: float = Field(..., description="校正湿度 = 原值 + 湿度偏移")


class CalibratedTimelineOut(BaseModel):
    device_sn: str
    start: dt.datetime | None = Field(None, description="实际生效的起始过滤（UTC），未指定为 null")
    end: dt.datetime | None = Field(None, description="实际生效的结束过滤（UTC），未指定为 null")
    sample_count: int = Field(..., description="过滤后返回的读数数量")
    aligned_anchor_count: int = Field(
        ..., description="已对齐到读数的锚点数量（未必都参与某一维曲线）"
    )
    points: list[CalibratedPointOut]


ExcursionDimension = Literal["temperature", "humidity"]


class ExcursionEventOut(BaseModel):
    start_ts: dt.datetime
    end_ts: dt.datetime
    dimensions: list[ExcursionDimension] = Field(
        ..., description="触发该事件的维度（temperature 在前，可多个）"
    )
    duration_seconds: float = Field(..., description="事件持续秒数")
    temp_dose: float = Field(..., description="事件内温度越界剂量（°C·秒）")
    humidity_dose: float = Field(..., description="事件内湿度越界剂量（%RH·秒）")
    peak_temperature: float | None = Field(
        None, description="温度峰值：越界幅度最大处的温度（未触发温度时为 null）"
    )
    peak_temperature_ts: dt.datetime | None = None
    peak_humidity: float | None = Field(
        None, description="湿度峰值：越界幅度最大处的湿度（未触发湿度时为 null）"
    )
    peak_humidity_ts: dt.datetime | None = None


class ExcursionsOut(BaseModel):
    device_sn: str
    start: dt.datetime | None = Field(None, description="实际生效的起始过滤（UTC），未指定为 null")
    end: dt.datetime | None = Field(None, description="实际生效的结束过滤（UTC），未指定为 null")
    max_gap_seconds: float = Field(..., description="生效的最大采样间隔（秒）")
    temp_min: float | None = None
    temp_max: float | None = None
    humidity_min: float | None = None
    humidity_max: float | None = None
    sample_count: int = Field(..., description="过滤后参与计算的读数数量")
    event_count: int
    events: list[ExcursionEventOut]


class ExposureBudgetDimensionOut(BaseModel):
    dimension: ExcursionDimension
    budget: float = Field(
        ..., description="该维越界剂量预算（温度 °C·秒 / 湿度 %RH·秒），非负"
    )
    cumulative_dose: float = Field(
        ..., description="查询窗末累计越界剂量（断档区间不计入）"
    )
    remaining: float = Field(
        ..., description="剩余额度（budget − cumulative_dose；超出预算时为负）"
    )
    budget_exhausted: bool = Field(
        ..., description="窗内累计剂量是否已达到预算"
    )
    exhausted_ts: dt.datetime | None = Field(
        None,
        description=(
            "首次达到预算的精确时刻（在线段内按越界幅度线性变化反解）；"
            "零预算在该维首次越界起点耗尽；窗末仍未达到预算为 null"
        ),
    )


class ExposureBudgetOut(BaseModel):
    device_sn: str
    start: dt.datetime | None = Field(None, description="实际生效的起始过滤（UTC），未指定为 null")
    end: dt.datetime | None = Field(None, description="实际生效的结束过滤（UTC），未指定为 null")
    max_gap_seconds: float = Field(..., description="生效的最大采样间隔（秒）")
    temp_min: float | None = None
    temp_max: float | None = None
    humidity_min: float | None = None
    humidity_max: float | None = None
    temp_budget: float | None = Field(None, description="温度剂量预算（未配置该维为 null）")
    humidity_budget: float | None = Field(None, description="湿度剂量预算（未配置该维为 null）")
    sample_count: int = Field(..., description="过滤后参与计算的读数数量")
    dimensions: list[ExposureBudgetDimensionOut] = Field(
        ..., description="各维累计剂量与预算状态（temperature 在前）"
    )


IncidentDirection = Literal["up", "down", "flat"]


class IncidentBoundaryOut(BaseModel):
    device_sn: str
    ts: dt.datetime = Field(..., description="该展柜非基线段段首（突变发生时刻）")
    direction: IncidentDirection = Field(
        ..., description="边界方向：以占主导的温/湿跳变分量符号判定"
    )
    magnitude: float = Field(..., description="该边界的缩放加权变化幅度")
    delta_temperature: float
    delta_humidity: float


class IncidentOut(BaseModel):
    start_ts: dt.datetime = Field(..., description="事件区间起点（成员中最早的边界）")
    end_ts: dt.datetime = Field(..., description="事件区间终点（成员中最晚的边界）")
    duration_seconds: float = Field(..., description="事件跨度（end_ts - start_ts，秒）")
    device_count: int = Field(..., description="参与该事件的 distinct 展柜数")
    total_magnitude: float = Field(..., description="各展柜边界幅度之和")
    boundaries: list[IncidentBoundaryOut] = Field(
        ..., description="各展柜保留的边界（每个展柜一条），按 (ts, device_sn) 排序"
    )


class IncidentsOut(BaseModel):
    start: dt.datetime = Field(..., description="实际生效的起始过滤（UTC）")
    end: dt.datetime = Field(..., description="实际生效的结束过滤（UTC）")
    tolerance_seconds: float = Field(..., description="候选窗允许的最大边界跨度（秒）")
    min_devices: int = Field(..., description="合法候选要求的最少 distinct 展柜数")
    candidate_count: int = Field(
        ..., description="枚举得到的全部合法连续候选窗数量（去重后）"
    )
    boundary_count: int = Field(..., description="过滤后参与检测的非基线段首数量")
    incident_count: int
    incidents: list[IncidentOut]


PropagationDimension = Literal["temperature", "humidity"]
PropagationDirection = Literal[
    "source_to_target",  # 最佳时滞 > 0：source 领先 target
    "target_to_source",  # 最佳时滞 < 0：target 领先 source
    "synchronous",       # 最佳时滞 = 0：同步
    "unavailable",       # 没有任何可用时滞
]


class PropagationLagOut(BaseModel):
    lag_seconds: float = Field(
        ..., description="时滞（秒）：正值表示 source 领先 target，含 0，按 step 等距枚举"
    )
    available: bool = Field(..., description="该时滞是否得到有效相关系数")
    pair_count: int = Field(..., description="参与相关计算的成对有效点数")
    correlation: float | None = Field(
        None, description="皮尔逊相关系数；不可用（样本不足/常量序列/无重叠）时为 null"
    )
    reason: str | None = Field(
        None,
        description="不可用原因：no_overlap / insufficient_samples / constant_series",
    )


class PropagationOut(BaseModel):
    source_sn: str
    target_sn: str
    dimension: PropagationDimension
    start: dt.datetime = Field(..., description="实际生效的起始时间（UTC）")
    end: dt.datetime = Field(..., description="实际生效的结束时间（UTC）")
    overlap_start: dt.datetime | None = Field(
        None, description="两柜读数覆盖范围与查询窗交集的起点（无重叠为 null）"
    )
    overlap_end: dt.datetime | None = Field(
        None, description="两柜读数覆盖范围与查询窗交集的终点（无重叠为 null）"
    )
    step_seconds: float
    max_lag_seconds: float
    max_gap_seconds: float = Field(
        ..., description="允许的最大相邻读数间隔（秒）；超过则该插值点无效"
    )
    grid_point_count: int = Field(..., description="重叠时段内固定步长网格点数")
    source_valid_count: int = Field(..., description="source 侧有效插值点数")
    target_valid_count: int = Field(..., description="target 侧有效插值点数")
    lags: list[PropagationLagOut] = Field(
        ..., description="各枚举时滞的明细，按时滞升序排列（含 0）"
    )
    best_lag_seconds: float | None = Field(
        None, description="最佳时滞（秒）；无可用时滞时为 null"
    )
    best_correlation: float | None = Field(
        None, description="最佳时滞对应的皮尔逊相关系数；无可用时滞时为 null"
    )
    direction: PropagationDirection = Field(
        ...,
        description=(
            "传播方向：source_to_target（source 领先）/ target_to_source"
            "（target 领先）/ synchronous（零时滞）/ unavailable（不可用）"
        ),
    )


class CondensationWindowOut(BaseModel):
    start_ts: dt.datetime = Field(..., description="风险窗口起点：裕量曲线进入阈值（margin = margin_threshold_c）的精确插值时刻")
    end_ts: dt.datetime = Field(..., description="风险窗口终点：裕量曲线离开阈值的精确插值时刻")
    duration_seconds: float = Field(..., description="窗口持续秒数（end_ts − start_ts）")
    min_margin: float = Field(
        ..., description="窗口内最低冷凝裕量（°C）；裕量沿线段线性，必落在某个有效读数时刻"
    )
    min_margin_ts: dt.datetime = Field(..., description="最低裕量对应的读数时刻（并列取最早）")


class CondensationRiskOut(BaseModel):
    device_sn: str
    start: dt.datetime | None = Field(None, description="实际生效的起始过滤（UTC），未指定为 null")
    end: dt.datetime | None = Field(None, description="实际生效的结束过滤（UTC），未指定为 null")
    surface_temperature_c: float = Field(
        ..., description="给定的展柜表面温度（°C），冷凝裕量 = 表面温度 − 露点"
    )
    margin_threshold_c: float = Field(
        ..., description="冷凝裕量风险阈值（°C）：裕量 ≤ 该值视为风险（含等于）"
    )
    max_gap_seconds: float = Field(
        ..., description="允许的最大相邻读数间隔（秒）；超过则曲线断开，未知区间不计风险"
    )
    min_duration_seconds: float = Field(
        ..., description="风险窗口最短持续秒数（含等于）；更短的窗口不返回"
    )
    sample_count: int = Field(..., description="过滤后参与计算的读数数量（含湿度为 0 的无效读数）")
    window_count: int
    windows: list[CondensationWindowOut]


class ReadingOutlierDimensionOut(BaseModel):
    dimension: ExcursionDimension
    median: float = Field(..., description="该点邻点（不含自身）在该维上的中位数")
    deviation: float = Field(
        ..., description="原值相对邻点中位数的绝对偏差（|原值 − 中位数|）"
    )


class ReadingOutlierOut(BaseModel):
    ts: dt.datetime = Field(..., description="疑点读数时间戳（UTC）")
    temperature: float = Field(..., description="原始温度读数（°C，未做任何修改）")
    humidity: float = Field(..., description="原始湿度读数（%RH，未做任何修改）")
    dimensions: list[ReadingOutlierDimensionOut] = Field(
        ..., description="命中异常的维度（temperature 在前，可多个）"
    )


class ReadingOutliersOut(BaseModel):
    device_sn: str
    window_size: int = Field(..., description="每侧最多取的邻点数（不含自身）")
    k: float = Field(..., description="命中阈值倍数：|原值 − 中位数| 严格大于 k×尺度 才标记")
    min_temperature_delta: float = Field(
        ..., description="温度维尺度下限（°C）：scale = max(1.4826×MAD, 该值)"
    )
    min_humidity_delta: float = Field(
        ..., description="湿度维尺度下限（%RH）：scale = max(1.4826×MAD, 该值)"
    )
    max_gap_seconds: float = Field(
        ..., description="允许的最大相邻读数间隔（秒）；超过则切断序列，邻点不跨段取"
    )
    sample_count: int = Field(..., description="该展柜已合并读数总数")
    segment_count: int = Field(..., description="按 max_gap_seconds 切断后得到的连续段数")
    evaluated_count: int = Field(
        ..., description="左右各至少有 2 个同段邻点、实际参与判定的读数数量"
    )
    outlier_count: int = Field(..., description="至少一维命中的疑点读数数量")
    outliers: list[ReadingOutlierOut] = Field(
        ..., description="疑点列表，按时间升序排列"
    )


class SamplingGapOut(BaseModel):
    start_ts: dt.datetime = Field(
        ..., description="缺口起点：连续缺测段首个缺测时隙的时刻（UTC）"
    )
    end_ts: dt.datetime = Field(
        ..., description="缺口终点：连续缺测段末个缺测时隙的时刻（UTC）"
    )
    missing_slots: int = Field(..., description="该缺口内连续缺测的时隙数")


class SamplingCoverageOut(BaseModel):
    device_sn: str
    start: dt.datetime = Field(
        ..., description="实际生效的起始时间（UTC）：首个应采时刻"
    )
    end: dt.datetime = Field(
        ..., description="实际生效的结束时间（UTC）：时隙不晚于该时刻"
    )
    period_seconds: float = Field(..., description="采样周期（秒）")
    tolerance_seconds: float = Field(
        ..., description="匹配容差（秒）：|读数时刻 − 时隙| ≤ 该值（含等于）可覆盖"
    )
    expected_count: int = Field(
        ..., description="应采时隙数：起点为首个，按周期生成不晚于终点的时隙"
    )
    covered_count: int = Field(..., description="已覆盖时隙数（一对一匹配）")
    coverage: float = Field(..., description="覆盖率 = covered_count / expected_count")
    gap_count: int = Field(..., description="连续缺测时隙合并后的缺口数")
    gaps: list[SamplingGapOut] = Field(..., description="缺口列表，按起点时刻升序")


DeviationDirection = Literal[
    "up",    # 峰值处目标值高于对照基线
    "down",  # 峰值处目标值低于对照基线
]


class DeviationPeriodOut(BaseModel):
    start_ts: dt.datetime = Field(..., description="时段起点：首个被标记偏离的网格时刻（UTC）")
    end_ts: dt.datetime = Field(..., description="时段终点：末个被标记偏离的网格时刻（UTC）")
    duration_seconds: float = Field(..., description="持续秒数（end_ts − start_ts）")
    marked_count: int = Field(..., description="该时段内被标记偏离的网格点数")
    direction: DeviationDirection = Field(
        ..., description="峰值偏离方向：up（目标高于基线）/ down（目标低于基线）"
    )
    max_deviation: float = Field(
        ..., description="时段内最大绝对偏离（|目标值 − 对照中位数基线|）"
    )
    peak_ts: dt.datetime = Field(
        ..., description="最大偏离出现的网格时刻（并列取最早）"
    )


class PeerBaselineDeviationOut(BaseModel):
    target_sn: str = Field(..., description="目标展柜设备序号")
    peer_sns: list[str] = Field(..., description="对照展柜设备序号（2～10 个，按请求顺序）")
    dimension: ExcursionDimension = Field(..., description="分析维度：temperature / humidity")
    start: dt.datetime = Field(..., description="实际生效的起始时间（UTC）：等距网格原点")
    end: dt.datetime = Field(..., description="实际生效的结束时间（UTC）：网格时刻不晚于该时刻")
    step_seconds: float = Field(..., description="等距网格步长（秒）")
    max_gap_seconds: float = Field(
        ..., description="允许的最大相邻读数间隔（秒）；超过则该插值点无效"
    )
    deviation_threshold: float = Field(
        ..., description="偏离阈值：|目标值 − 基线| 达到该值（含等于）即标记偏离"
    )
    min_duration_seconds: float = Field(
        ..., description="偏离时段最短持续秒数（含等于）；更短的时段不返回"
    )
    grid_point_count: int = Field(..., description="查询窗内等距网格点数")
    evaluated_count: int = Field(
        ..., description="目标有效且对照有效值不少于两个、实际完成偏离判定的网格点数"
    )
    marked_count: int = Field(..., description="被标记偏离的网格点总数")
    period_count: int = Field(..., description="持续时间达标的偏离时段数")
    periods: list[DeviationPeriodOut] = Field(
        ..., description="偏离时段列表，按起点时刻升序"
    )


class HealthOut(BaseModel):
    status: str
    db: str


RecoveryNotRecoveredReason = Literal[
    "no_post_event_readings",        # 事件后观察窗内没有任何读数
    "never_simultaneously_in_band",  # 从未形成两维同时在带内的正长度区间
    "dwell_interrupted",             # 两维曾同时在带，但出带或断档中断了保持计时
    "dwell_incomplete_at_horizon",   # 观察结束时仍在带内，但连续保持未满 hold_seconds
]


class RecoveryBaselineOut(BaseModel):
    window_start: dt.datetime = Field(
        ..., description="基线窗口起点（event_ts − baseline_seconds，UTC）"
    )
    window_end: dt.datetime = Field(..., description="基线窗口终点（即 event_ts，UTC）")
    sample_count: int = Field(..., description="基线窗口（含两端）内参与中位数的读数数量")
    temperature: float = Field(..., description="基线温度中位数（°C）")
    humidity: float = Field(..., description="基线湿度中位数（%RH）")
    temperature_low: float = Field(..., description="温度容差带下限 = 中位数 − temp_tolerance（°C，含）")
    temperature_high: float = Field(..., description="温度容差带上限 = 中位数 + temp_tolerance（°C，含）")
    humidity_low: float = Field(..., description="湿度容差带下限 = 中位数 − humidity_tolerance（%RH，含）")
    humidity_high: float = Field(..., description="湿度容差带上限 = 中位数 + humidity_tolerance（%RH，含）")


class RecoveryOut(BaseModel):
    device_sn: str
    event_ts: dt.datetime = Field(..., description="扰动事件时刻（UTC）")
    horizon_end: dt.datetime = Field(
        ..., description="事件后观察窗终点（event_ts + horizon_seconds，UTC）"
    )
    baseline_seconds: float = Field(..., description="事件前基线窗口长度（秒）")
    horizon_seconds: float = Field(..., description="事件后观察窗长度（秒）")
    min_baseline_samples: int = Field(..., description="基线最少读数数量要求")
    temp_tolerance: float = Field(..., description="温度容差半宽（°C，含边界）")
    humidity_tolerance: float = Field(..., description="湿度容差半宽（%RH，含边界）")
    hold_seconds: float = Field(
        ..., description="两维同时在带内所需的连续保持秒数（含等于）"
    )
    max_gap_seconds: float = Field(
        ..., description="允许的最大相邻读数间隔（秒）；超过即中断保持计时"
    )
    baseline_sample_count: int = Field(..., description="基线窗口内实际读数数量")
    post_event_sample_count: int = Field(
        ..., description="事件后观察窗 (event_ts, event_ts + horizon_seconds] 内读数数量"
    )
    baseline: RecoveryBaselineOut
    recovered: bool = Field(..., description="两维是否曾同时连续在带内保持满 hold_seconds")
    band_enter_ts: dt.datetime | None = Field(
        None,
        description="成功保持段的起点：两维同时进入容差带的精确插值时刻（未恢复为 null）",
    )
    recovered_at: dt.datetime | None = Field(
        None,
        description="恢复确认时刻 = band_enter_ts + hold_seconds（未恢复为 null）",
    )
    recovery_seconds: float | None = Field(
        None, description="恢复用时（recovered_at − event_ts，秒；未恢复为 null）"
    )
    max_temperature_deviation: float | None = Field(
        None, description="事件后温度相对基线中位数的最大绝对偏差（°C）；无事件后读数为 null"
    )
    max_temperature_deviation_ts: dt.datetime | None = Field(
        None, description="最大温度偏差对应的读数时刻（并列取最早）；无事件后读数为 null"
    )
    max_humidity_deviation: float | None = Field(
        None, description="事件后湿度相对基线中位数的最大绝对偏差（%RH）；无事件后读数为 null"
    )
    max_humidity_deviation_ts: dt.datetime | None = Field(
        None, description="最大湿度偏差对应的读数时刻（并列取最早）；无事件后读数为 null"
    )
    reason: RecoveryNotRecoveredReason | None = Field(
        None, description="未恢复原因；已恢复时为 null"
    )


IngestSummary.model_rebuild()
ReplayPreviewResponse.model_rebuild()
