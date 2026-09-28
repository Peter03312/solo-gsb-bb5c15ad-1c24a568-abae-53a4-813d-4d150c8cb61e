"""SQLAlchemy 引擎、会话与 ORM 模型。"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from .config import settings


def _build_engine():
    url = settings.DB_URL
    if url.startswith("sqlite"):
        from sqlalchemy.pool import StaticPool

        # 内存库（测试用）需要 StaticPool 才能跨连接共享
        connect_args = {"check_same_thread": False}
        kwargs: dict = {"connect_args": connect_args}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
        engine = create_engine(url, **kwargs)

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

        return engine

    return create_engine(url)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def to_utc(value: dt.datetime) -> dt.datetime:
    """把任意时间戳归一化为带时区的 UTC。"""
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


class Reading(Base):
    """展柜温湿度读数。

    (device_sn, ts) 唯一：同序号同时间戳的重复上报在入库时被合并。
    """

    __tablename__ = "readings"
    __table_args__ = (
        UniqueConstraint("device_sn", "ts", name="uq_reading_device_ts"),
        Index("ix_reading_device_ts", "device_sn", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_sn: Mapped[str] = mapped_column(String(64), nullable=False)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    humidity: Mapped[float] = mapped_column(Float, nullable=False)
    # 合并后的均值对应的原始上报次数
    report_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    anchors: Mapped[list["Anchor"]] = relationship(back_populates="reading", cascade="all, delete-orphan")


class Anchor(Base):
    """锚点：强制的分段边界（校准/人工确认的真实变点）。"""

    __tablename__ = "anchors"
    __table_args__ = (
        UniqueConstraint("device_sn", "ts", name="uq_anchor_device_ts"),
        Index("ix_anchor_device_ts", "device_sn", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_sn: Mapped[str] = mapped_column(String(64), nullable=False)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    # 对齐后实际落在的读数（可能为空：尚无读数时锚点先登记）
    reading_id: Mapped[int | None] = mapped_column(ForeignKey("readings.id", ondelete="SET NULL"), nullable=True)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ref_temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    ref_humidity: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    reading: Mapped["Reading | None"] = relationship(back_populates="anchors")


class SegmentSnapshot(Base):
    """某次分段结果的持久化快照（重放落库后覆盖更新）。"""

    __tablename__ = "segment_snapshots"
    __table_args__ = (Index("ix_snapshot_device_order", "device_sn", "segment_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_sn: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ts: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    end_ts: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_temperature: Mapped[float] = mapped_column(Float, nullable=False)
    mean_humidity: Mapped[float] = mapped_column(Float, nullable=False)
    delta_temperature: Mapped[float] = mapped_column(Float, nullable=False)
    delta_humidity: Mapped[float] = mapped_column(Float, nullable=False)
    magnitude: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    start_anchored: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    end_anchored: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class CaseState(Base):
    """每个展柜的分段版本号与最近一次分段参数。"""

    __tablename__ = "case_states"

    device_sn: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    penalty: Mapped[float] = mapped_column(Float, nullable=False, default=settings.PENALTY)
    reading_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    replayed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)


def init_db() -> None:
    # SQLite 文件库：确保数据目录存在（Docker 镜像已预建 /app/data，
    # 本地裸跑或自定义 CASE_DATA_DIR 时由这里兜底）
    if settings.DB_URL.startswith("sqlite") and ":memory:" not in settings.DB_URL:
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
