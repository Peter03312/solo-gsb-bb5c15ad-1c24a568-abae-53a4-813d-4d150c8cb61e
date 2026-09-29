"""旧版分段快照的两维标准差恢复与校验。

历史版本的 ``segment_snapshots`` 不保存 ``temperature_std`` /
``humidity_std``，读取时被固定回显为 0——即使段内读数并不恒定。
本模块在启动（``init_db``）与读取快照时，按每个快照覆盖的读数窗口
重算两维**总体标准差**（与 :func:`app.segmentation.segment_samples`
完全相同的口径，除以样本数 n），并严格核对：

* 快照下标连续，分段首尾相接、恰好覆盖该展柜全部读数；
* ``sample_count`` 与窗口内读数数一致；
* 快照保存的温湿度均值、加权代价与窗口读数重算值一致。

全部一致才把两列统计值补写回旧快照——**只补两列**，不改分段边界、
版本号及其余字段。任一项无法对应即视为旧快照不可核实，保留缺失值，
由读取方向调用方明确报错（缺失值绝不能被当作真实零值回显）。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .config import settings
from .database import Reading, SegmentSnapshot, to_utc
from .segmentation import _Prefix

logger = logging.getLogger("app.snapshot_recovery")
# 均值/代价均为同一浮点运算落库的值，允许极小的相对误差
_REL_TOL = 1e-7


class SnapshotStatsUnverifiable(Exception):
    """旧快照缺失统计列，且无法由当前读数核实重算。"""

    def __init__(self, device_sn: str, reason: str):
        self.device_sn = device_sn
        self.reason = reason
        super().__init__(f"展柜 {device_sn} 的旧分段快照统计值无法由现有读数核实：{reason}")


def _close(actual: float, stored: float) -> bool:
    return abs(actual - stored) <= _REL_TOL * max(1.0, abs(stored))


def _load_readings(db: Session, device_sn: str) -> list[Reading]:
    return list(
        db.scalars(
            select(Reading)
            .where(Reading.device_sn == device_sn)
            .order_by(Reading.ts.asc())
        ).all()
    )


def _legacy_rows(db: Session, device_sn: str) -> list[SegmentSnapshot]:
    return list(
        db.scalars(
            select(SegmentSnapshot)
            .where(SegmentSnapshot.device_sn == device_sn)
            .order_by(SegmentSnapshot.segment_index.asc())
        ).all()
    )


def verify_device(
    db: Session, device_sn: str
) -> tuple[list[tuple[SegmentSnapshot, float, float]] | None, str | None]:
    """校验快照可由当前读数核实，并给出每段应补写的两维总体标准差。

    成功返回 ``(pairs, None)``，pairs 为 ``(快照行, 温度std, 湿度std)``；
    失败返回 ``(None, 原因)``，不做任何写入。
    """
    rows = _legacy_rows(db, device_sn)
    if not rows:
        return None, "该展柜没有分段快照"

    readings = _load_readings(db, device_sn)
    if not readings:
        return None, "快照对应的读数已不存在"

    expected_indices = list(range(len(rows)))
    if [r.segment_index for r in rows] != expected_indices:
        return None, "快照 segment_index 不连续，分段结构可能已被改动"

    times = [to_utc(r.ts) for r in readings]
    position = {ts: i for i, ts in enumerate(times)}
    pt = _Prefix([r.temperature for r in readings])
    ph = _Prefix([r.humidity for r in readings])

    pairs: list[tuple[SegmentSnapshot, float, float]] = []
    cursor = 0  # 下一段必须从该下标开始，保证各段首尾相接且不重不漏
    for row in rows:
        start_ts = to_utc(row.start_ts)
        end_ts = to_utc(row.end_ts)
        if start_ts not in position or end_ts not in position:
            return None, "快照边界时间戳无法对应到现有读数"
        a = position[start_ts]
        b = position[end_ts] + 1  # 半开区间 [a, b)
        if a != cursor:
            return None, "快照分段未首尾相接覆盖完整读数时间线"
        if b <= a:
            return None, "快照分段起止顺序异常"
        if b - a != row.sample_count:
            return None, "快照 sample_count 与窗口内读数数量不一致"

        if not _close(pt.mean(a, b), row.mean_temperature):
            return None, "快照 mean_temperature 与窗口读数重算值不一致"
        if not _close(ph.mean(a, b), row.mean_humidity):
            return None, "快照 mean_humidity 与窗口读数重算值不一致"

        expected_cost = (
            settings.TEMP_WEIGHT * pt.sse(a, b)
            + settings.HUMIDITY_WEIGHT * ph.sse(a, b)
        )
        if not _close(expected_cost, row.cost):
            return None, "快照 cost 与窗口读数重算值不一致"

        pairs.append((row, pt.pstd(a, b), ph.pstd(a, b)))
        cursor = b

    if cursor != len(readings):
        return None, "快照未覆盖该展柜的全部现有读数"
    return pairs, None


def backfill_device(db: Session, device_sn: str, commit: bool = False) -> bool:
    """核实并补写单个展柜旧快照缺失的两维标准差。

    仅当缺失（列为 NULL）时尝试；成功补写返回 True，无法核实返回 False
    （保留 NULL，不写零、不改分段与版本）。
    """
    rows = _legacy_rows(db, device_sn)
    if not rows or not _has_legacy_stats(rows):
        return True

    pairs, reason = verify_device(db, device_sn)
    if pairs is None:
        logger.warning("展柜 %s 旧快照统计值恢复失败：%s", device_sn, reason)
        return False

    for row, t_std, h_std in pairs:
        row.temperature_std = t_std
        row.humidity_std = h_std
    db.flush()
    if commit:
        db.commit()
    logger.info("展柜 %s 旧快照两维标准差已按读数核实并补写（%d 段）", device_sn, len(pairs))
    return True


def _has_legacy_stats(rows: Iterable[SegmentSnapshot]) -> bool:
    return any(r.temperature_std is None or r.humidity_std is None for r in rows)


def recover_all(db: Session) -> dict[str, str]:
    """启动期恢复：对所有存在缺失统计列快照的展柜逐柜核实补写。

    返回 ``{device_sn: "recovered" | "unverifiable"}``。单柜失败不影响
    其他展柜，也不阻断服务启动。
    """
    devices = list(
        db.scalars(
            select(SegmentSnapshot.device_sn)
            .distinct()
            .where(
                or_(
                    SegmentSnapshot.temperature_std.is_(None),
                    SegmentSnapshot.humidity_std.is_(None),
                )
            )
        ).all()
    )
    summary: dict[str, str] = {}
    for device_sn in devices:
        summary[device_sn] = (
            "recovered" if backfill_device(db, device_sn, commit=True) else "unverifiable"
        )
    return summary
