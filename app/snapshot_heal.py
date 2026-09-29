"""分段快照标准差列的模式迁移与历史快照恢复。

旧版本把 ``temperature_std`` / ``humidity_std`` 计算出来后只在即时
重放响应中返回，**没有随快照保存**；读取快照时又把两维标准差硬编码
为 ``0``，导致重放后的重复查询与服务重启后的查询把非恒定段误报为零
波动。本模块负责：

1. 为旧库的 ``segment_snapshots`` 表幂等补建两列（:func:`migrate_and_heal`）；
2. 从快照对应的**已合并读数**核实快照身份（段边界、样本数、均值必须
   一致，且全部读数恰好被各段覆盖一次），核实通过后恢复两维**总体
   标准差**并写回（:func:`heal_all`）；
3. 为只读查询路径提供按行的即时核实恢复（:func:`segment_std`）。

无法由读数核实的快照一律抛 :class:`UnverifiableSnapshotError`，绝不
把缺失值当作真实零值，也不改动任何分段边界、样本数或版本号。
"""

from __future__ import annotations

import datetime as dt
import logging
import math

from sqlalchemy import inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .database import Reading, SegmentSnapshot, to_utc

logger = logging.getLogger(__name__)

# 均值回读核实容差（物理单位：°C / %RH）。快照均值与读数均值不一致即
# 说明快照已无法由当前读数核实，不得再据此恢复标准差。
MEAN_TOLERANCE = 1e-9

_STD_COLUMNS = ("temperature_std", "humidity_std")


class UnverifiableSnapshotError(Exception):
    """快照统计无法由库内对应读数核实（读数缺失/数量或均值不符）。

    读取分段快照遇到该错误时必须显式报错，不能把缺失的标准差当作 0
    返回，也不允许改动分段或版本来回避。
    """

    def __init__(self, message: str, *, device_sn: str, version: int | None = None,
                 segment_index: int | None = None, reason: str = "unverifiable"):
        super().__init__(message)
        self.device_sn = device_sn
        self.version = version
        self.segment_index = segment_index
        self.reason = reason


def population_std(values: list[float]) -> float:
    """总体标准差 sqrt(Σ(x−mean)²/N)；样本数 < 2 时为 0。"""
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(max(0.0, sum((v - mean) ** 2 for v in values)) / n)


def _segment_readings(
    db: Session, device_sn: str, start_ts: dt.datetime, end_ts: dt.datetime
) -> list[Reading]:
    """快照段 [start_ts, end_ts] 闭区间内的已合并读数（按时间升序）。"""
    return list(
        db.scalars(
            select(Reading)
            .where(
                Reading.device_sn == device_sn,
                Reading.ts >= start_ts,
                Reading.ts <= end_ts,
            )
            .order_by(Reading.ts.asc())
        ).all()
    )


def _recover_segment_std(
    db: Session, row: SegmentSnapshot
) -> tuple[float, float]:
    """由段对应读数核实样本数与均值，返回两维总体标准差。

    核实任一项失败即抛 :class:`UnverifiableSnapshotError`。
    """
    device_sn = row.device_sn
    start_ts = to_utc(row.start_ts)
    end_ts = to_utc(row.end_ts)
    readings = _segment_readings(db, device_sn, start_ts, end_ts)
    idx = row.segment_index
    version = row.version

    if len(readings) != row.sample_count:
        raise UnverifiableSnapshotError(
            f"展柜 {device_sn} 版本 {version} 第 {idx} 段快照需 "
            f"{row.sample_count} 条读数，当前读数仅 {len(readings)} 条可对应，"
            "标准差无法核实，拒绝以零值回显",
            device_sn=device_sn,
            version=version,
            segment_index=idx,
            reason="sample_count_mismatch",
        )

    # 段边界必须是真实读数时刻，否则“由对应读数核实”不成立
    if to_utc(readings[0].ts) != start_ts or to_utc(readings[-1].ts) != end_ts:
        raise UnverifiableSnapshotError(
            f"展柜 {device_sn} 版本 {version} 第 {idx} 段快照边界"
            f" {start_ts.isoformat()} ~ {end_ts.isoformat()} 无法与读数时刻对应",
            device_sn=device_sn,
            version=version,
            segment_index=idx,
            reason="boundary_reading_missing",
        )

    temps = [r.temperature for r in readings]
    hums = [r.humidity for r in readings]
    mean_t = sum(temps) / len(temps)
    mean_h = sum(hums) / len(hums)
    if (
        abs(mean_t - row.mean_temperature) > MEAN_TOLERANCE
        or abs(mean_h - row.mean_humidity) > MEAN_TOLERANCE
    ):
        raise UnverifiableSnapshotError(
            f"展柜 {device_sn} 版本 {version} 第 {idx} 段快照均值"
            f" ({row.mean_temperature}, {row.mean_humidity}) 与当前读数均值"
            f" ({mean_t}, {mean_h}) 不一致，快照统计无法核实",
            device_sn=device_sn,
            version=version,
            segment_index=idx,
            reason="mean_mismatch",
        )
    return population_std(temps), population_std(hums)


def _verify_partition(db: Session, device_sn: str, rows: list[SegmentSnapshot]) -> None:
    """核实该展柜快照集合仍是当前读数的一个完整划分。

    - 段下标连续、相邻段在读数序列上首尾相接；
    - 首段起点为第一条读数、末段终点为最后一条读数；
    - 全部读数恰好被覆盖一次（各段样本数之和等于读数总数）。
    """
    count = len(
        db.scalars(select(Reading).where(Reading.device_sn == device_sn)).all()
    )
    if count == 0:
        raise UnverifiableSnapshotError(
            f"展柜 {device_sn} 存在 {len(rows)} 段快照但没有任何读数，"
            "快照统计无法核实",
            device_sn=device_sn,
            version=rows[0].version if rows else None,
            reason="no_readings",
        )

    for expected_index, row in enumerate(rows):
        if row.segment_index != expected_index:
            raise UnverifiableSnapshotError(
                f"展柜 {device_sn} 版本 {row.version} 快照段下标不连续，"
                f"第 {expected_index} 行实际为 {row.segment_index}，分段完整性无法核实",
                device_sn=device_sn,
                version=row.version,
                segment_index=row.segment_index,
                reason="segment_index_gap",
            )

    if sum(r.sample_count for r in rows) != count:
        raise UnverifiableSnapshotError(
            f"展柜 {device_sn} 版本 {rows[0].version} 各段样本数之和"
            f" {sum(r.sample_count for r in rows)} 与读数总数 {count} 不符，"
            "存在快照未覆盖的读数，分段统计无法核实",
            device_sn=device_sn,
            version=rows[0].version,
            reason="partition_incomplete",
        )

    all_rows = list(
        db.scalars(
            select(Reading)
            .where(Reading.device_sn == device_sn)
            .order_by(Reading.ts.asc())
        ).all()
    )
    first_ts, last_ts = to_utc(all_rows[0].ts), to_utc(all_rows[-1].ts)
    if to_utc(rows[0].start_ts) != first_ts or to_utc(rows[-1].end_ts) != last_ts:
        raise UnverifiableSnapshotError(
            f"展柜 {device_sn} 版本 {rows[0].version} 快照范围"
            f" {to_utc(rows[0].start_ts).isoformat()} ~ "
            f"{to_utc(rows[-1].end_ts).isoformat()} 未覆盖全部读数"
            f"（{first_ts.isoformat()} ~ {last_ts.isoformat()}）",
            device_sn=device_sn,
            version=rows[0].version,
            reason="partition_span_mismatch",
        )

    for prev, row in zip(rows, rows[1:]):
        prev_end = to_utc(prev.end_ts)
        next_start = to_utc(row.start_ts)
        # 段区间不允许重叠（重叠会使同一条读数被计入两段）
        if next_start <= prev_end:
            raise UnverifiableSnapshotError(
                f"展柜 {device_sn} 版本 {row.version} 第 {row.segment_index} 段"
                f"起点 {next_start.isoformat()} 不晚于上一段终点"
                f" {prev_end.isoformat()}，分段区间重叠，无法核实",
                device_sn=device_sn,
                version=row.version,
                segment_index=row.segment_index,
                reason="partition_overlap",
            )
        # 相邻段必须在读数序列上首尾相接：上一段末读数之后的首条读数
        # 就是下一段的起点
        nxt = db.scalar(
            select(Reading).where(
                Reading.device_sn == device_sn,
                Reading.ts > prev_end,
            ).order_by(Reading.ts.asc()).limit(1)
        )
        if nxt is None or to_utc(nxt.ts) != next_start:
            raise UnverifiableSnapshotError(
                f"展柜 {device_sn} 版本 {row.version} 第 {row.segment_index} 段"
                f"起点 {next_start.isoformat()} 与上一段末读数"
                f" {prev_end.isoformat()} 之后的首条读数不衔接，分段无法核实",
                device_sn=device_sn,
                version=row.version,
                segment_index=row.segment_index,
                reason="partition_disjoint",
            )


def segment_std(db: Session, row: SegmentSnapshot) -> tuple[float, float]:
    """读取路径使用的标准差解析：快照已保存则原样返回，否则按读数即时核实恢复。

    旧快照（两列为 NULL）在服务进程中第一次被读取时由此完成核实；
    核实失败抛 :class:`UnverifiableSnapshotError`。该函数不写库——
    持久化恢复只在启动迁移（:func:`heal_all`）时进行，保证重放预览等
    只读接口无任何副作用。
    """
    if row.temperature_std is not None and row.humidity_std is not None:
        return row.temperature_std, row.humidity_std
    return _recover_segment_std(db, row)


def heal_all(session_factory) -> dict[str, int]:
    """恢复全部历史快照缺失的两维总体标准差并写回。

    对每个展柜先核实分段整体仍是当前读数的完整划分，再逐段核实样本数
    与均值；全部通过才把恢复值 UPDATE 回快照行。任一展柜核实失败时该
    展柜保持原状（缺失值仍为 NULL），记录错误日志并继续处理其他展柜——
    该展柜之后被查询时会显式报错，绝不静默改成零或改动分段/版本。

    返回 ``{device_sn: 被恢复的段数}``（只含实际写回的展柜）。
    """
    db: Session = session_factory()
    healed: dict[str, int] = {}
    try:
        device_sns = [
            sn
            for (sn,) in db.execute(
                select(SegmentSnapshot.device_sn).distinct()
            ).all()
        ]
        for device_sn in sorted(device_sns):
            rows = list(
                db.scalars(
                    select(SegmentSnapshot)
                    .where(SegmentSnapshot.device_sn == device_sn)
                    .order_by(SegmentSnapshot.segment_index.asc())
                ).all()
            )
            missing = [
                r for r in rows
                if r.temperature_std is None or r.humidity_std is None
            ]
            if not missing:
                continue
            try:
                _verify_partition(db, device_sn, rows)
                recovered: dict[int, tuple[float, float]] = {}
                for row in missing:
                    recovered[row.id] = _recover_segment_std(db, row)
                for row in missing:
                    t_std, h_std = recovered[row.id]
                    row.temperature_std = t_std
                    row.humidity_std = h_std
                # 每个展柜独立提交：坏展柜回滚不影响其他展柜的恢复结果
                db.commit()
            except UnverifiableSnapshotError as exc:
                db.rollback()
                logger.error(
                    "展柜 %s 的历史快照无法由当前读数核实，标准差未恢复：%s",
                    device_sn,
                    exc,
                )
                continue
            healed[device_sn] = len(missing)
            logger.info(
                "展柜 %s 的 %d 段历史快照标准差已由读数核实恢复",
                device_sn,
                len(missing),
            )
    finally:
        db.close()
    return healed


def migrate_and_heal(engine: Engine) -> None:
    """幂等补建标准差列，然后从读数核实恢复旧快照统计。

    - 全新数据库：``create_all`` 已含新列，无旧行，直接返回；
    - 升级旧库：表存在但缺列时用 ``ALTER TABLE ADD COLUMN`` 补建
      （SQLite 支持，且新列可空，不触碰任何既有数据）；
    - 补列后执行一次全量核实恢复。
    """
    inspector = inspect(engine)
    if "segment_snapshots" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("segment_snapshots")}
    with engine.begin() as conn:
        for column in _STD_COLUMNS:
            if column not in existing:
                conn.exec_driver_sql(
                    f"ALTER TABLE segment_snapshots ADD COLUMN {column} FLOAT"
                )
    # 用传入的 engine 建立会话（不能复用模块级 SessionLocal：测试或多
    # 数据源场景下它可能绑定另一个库，例如默认的内存库）
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    heal_all(session_factory)
