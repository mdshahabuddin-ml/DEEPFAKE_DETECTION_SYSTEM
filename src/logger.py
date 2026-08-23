"""
src/logger.py
Centralized, reusable logging configuration.

Logs go to both the console and a rotating file at logs/application.log.
Call get_logger(__name__) from any module to obtain a configured logger.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.config import LOGS_DIR

LOG_FILE = LOGS_DIR / "application.log"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track configured logger names so handlers are attached only once.
_configured: set[str] = set()


def get_logger(name: str = "deepfake", level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to the console and logs/application.log.

    Safe to call repeatedly with the same name — handlers are added only once.
    """
    logger = logging.getLogger(name)
    if name in _configured:
        return logger

    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Rotating file handler (5 MB x 3 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _configured.add(name)
    return logger
