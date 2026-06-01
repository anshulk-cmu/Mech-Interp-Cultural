"""Console + per-stage file logging."""
from __future__ import annotations

import logging

from .config import LOGS


def get_logger(stage: str) -> logging.Logger:
    logger = logging.getLogger(stage)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(LOGS / f"{stage}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
