"""运行配置。

所有参数都有内置默认值，也可以通过同名大写环境变量覆盖
（例如 ``CASE_PENALTY=6.4``），便于在 Compose 中调整。
"""

from __future__ import annotations

import os
from pathlib import Path


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


class Settings:
    # SQLite 数据库文件位置（文件模式时使用）
    DATA_DIR = Path(os.getenv("CASE_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
    DB_URL = os.getenv(
        "CASE_DB_URL",
        f"sqlite:///{(DATA_DIR / 'cases.db').as_posix()}",
    )

    # —— 变点分段参数 ——
    # 惩罚项 beta：新增一段必须带来不小于 beta 的代价下降，
    # 数值越大分段越少。尺度与特征缩放后的 SSE 可比。
    PENALTY: float = _get_float("CASE_PENALTY", 4.0)
    # 单段最少样本数（受锚点强制切开的小区间会自动放宽）
    MIN_SEGMENT_SIZE: int = _get_int("CASE_MIN_SEGMENT_SIZE", 3)
    # 锚点时间戳与最近读数时间戳允许的对齐偏差（秒）
    ANCHOR_TOLERANCE: float = _get_float("CASE_ANCHOR_TOLERANCE", 300.0)

    # 温湿度在联合代价中的权重（特征先各自 z-score 风格缩放，
    # 再按权重合成 2 维 SSE）
    TEMP_WEIGHT: float = _get_float("CASE_TEMP_WEIGHT", 1.0)
    HUMIDITY_WEIGHT: float = _get_float("CASE_HUMIDITY_WEIGHT", 1.0)

    # 特征缩放（摄氏度 / 百分比相对湿度），使两个量纲可联合
    TEMP_SCALE: float = _get_float("CASE_TEMP_SCALE", 2.0)
    HUMIDITY_SCALE: float = _get_float("CASE_HUMIDITY_SCALE", 5.0)

    # —— 置信标记阈值 ——
    # 变化幅度超过该倍噪声标准差才算显著，逐级提升置信度
    CONF_HIGH_SIGMAS: float = _get_float("CASE_CONF_HIGH_SIGMAS", 3.0)
    CONF_MEDIUM_SIGMAS: float = _get_float("CASE_CONF_MEDIUM_SIGMAS", 1.5)
    # 变化幅度的最小绝对值（缩放后），低于它即使样本多也判 low
    CONF_MIN_MAGNITUDE: float = _get_float("CASE_CONF_MIN_MAGNITUDE", 0.5)
    # 相邻两段样本数都达到该值才可能给 high
    CONF_HIGH_MIN_SAMPLES: int = _get_int("CASE_CONF_HIGH_MIN_SAMPLES", 4)

    SERVICE_PORT: int = _get_int("CASE_PORT", 8080)


settings = Settings()
