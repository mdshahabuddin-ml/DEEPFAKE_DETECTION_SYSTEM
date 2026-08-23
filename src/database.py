"""
src/database.py
SQLite storage for prediction history.

A single `predictions` table records every image/video/webcam prediction made
through the app or the API. The PredictionDB class is intentionally small,
thread-safe (check_same_thread=False + short-lived connections), and dependency
-free (standard library sqlite3 only).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.config import DB_PATH
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionRecord:
    """One stored prediction."""
    file_name: str
    media_type: str          # "image" | "video" | "webcam"
    prediction: str          # "REAL" | "FAKE"
    confidence: float        # confidence in the chosen label, [0, 1]
    fake_probability: float  # raw model fake-probability, [0, 1]
    inference_time: float    # seconds
    source: str = "app"      # "app" | "api"
    created_at: str = ""     # ISO timestamp, filled on insert if empty


_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name       TEXT    NOT NULL,
    media_type      TEXT    NOT NULL,
    prediction      TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    fake_probability REAL   NOT NULL,
    inference_time  REAL    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'app',
    created_at      TEXT    NOT NULL
);
"""


class PredictionDB:
    """Lightweight SQLite wrapper for prediction history."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info("PredictionDB ready at %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- writes --------------------------------------------------------------
    def add(self, record: PredictionRecord) -> int:
        """Insert a record; returns the new row id."""
        if not record.created_at:
            record.created_at = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO predictions
                   (file_name, media_type, prediction, confidence,
                    fake_probability, inference_time, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (record.file_name, record.media_type, record.prediction,
                 record.confidence, record.fake_probability,
                 record.inference_time, record.source, record.created_at),
            )
            row_id = int(cur.lastrowid)
        logger.info("Stored prediction #%d: %s (%s)", row_id,
                    record.prediction, record.file_name)
        return row_id

    # -- reads ---------------------------------------------------------------
    def recent(self, limit: int = 100) -> List[dict]:
        """Return the most recent predictions as a list of dicts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Aggregate counts and averages for the dashboard."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            fake = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE prediction = 'FAKE'"
            ).fetchone()[0]
            avg_conf = conn.execute(
                "SELECT AVG(confidence) FROM predictions"
            ).fetchone()[0]
            avg_time = conn.execute(
                "SELECT AVG(inference_time) FROM predictions"
            ).fetchone()[0]
        return {
            "total": int(total),
            "fake": int(fake),
            "real": int(total - fake),
            "avg_confidence": float(avg_conf) if avg_conf is not None else 0.0,
            "avg_inference_time": float(avg_time) if avg_time is not None else 0.0,
        }

    def clear(self) -> None:
        """Delete all history (used by the UI 'clear history' action)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM predictions")
        logger.warning("Prediction history cleared")


# Module-level singleton for convenient reuse.
_db: Optional[PredictionDB] = None


def get_db() -> PredictionDB:
    """Return a shared PredictionDB instance."""
    global _db
    if _db is None:
        _db = PredictionDB()
    return _db


def record_to_dict(record: PredictionRecord) -> dict:
    return asdict(record)
