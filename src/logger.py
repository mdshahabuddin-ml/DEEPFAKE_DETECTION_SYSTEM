"""
src/logger.py
Centralized, reusable logging configuration.

Logs go to both the console and a rotating file at logs/application.log.
Call get_logger(__name__) from any module to obtain a configured logger.

A single RotatingFileHandler is shared across all loggers — opening many
handles on the same file breaks rotation on Windows.
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

# Shared handlers (created once, reused by every logger).
_file_handler: RotatingFileHandler | None = None
_console_handler: logging.StreamHandler | None = None
_formatter: logging.Formatter | None = None


def _get_shared_handlers() -> tuple[logging.Handler, logging.Handler]:
    global _file_handler, _console_handler, _formatter
    if _formatter is None:
        _formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    if _console_handler is None:
        _console_handler = logging.StreamHandler()
        _console_handler.setFormatter(_formatter)
    if _file_handler is None:
        _file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        _file_handler.setFormatter(_formatter)
    return _console_handler, _file_handler


def get_logger(name: str = "deepfake", level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to the console and logs/application.log.

    Safe to call repeatedly with the same name — handlers are added only once.
    All loggers share one console handler and one rotating file handler.
    """
    logger = logging.getLogger(name)
    if name in _configured:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    console, file_handler = _get_shared_handlers()
    logger.addHandler(console)
    logger.addHandler(file_handler)

    _configured.add(name)
    return logger
