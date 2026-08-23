"""
src/results_writer.py
Persist prediction results as timestamped artifacts under results/.

Each call writes a JSON record (and optionally a Grad-CAM overlay PNG) named
with a timestamp so runs never overwrite each other. Complements the SQLite
history in src/database.py — the DB is for querying, these files are for
auditing / sharing individual results (useful for KYC evidence trails).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import RESULTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def _timestamp() -> str:
    """Filesystem-safe timestamp, e.g. 20260724_110301_123456."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def save_result(result: dict, overlay_rgb: Optional[np.ndarray] = None,
                prefix: str = "prediction") -> Path:
    """
    Save a prediction result to results/<timestamp>_<prefix>.json.

    If `overlay_rgb` (an RGB uint8 image, e.g. a Grad-CAM overlay) is provided,
    it is written alongside as a .png and referenced in the JSON.

    Returns the path to the JSON file.
    """
    ts = _timestamp()
    stem = f"{ts}_{prefix}"
    json_path = RESULTS_DIR / f"{stem}.json"

    record = dict(result)
    record.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))

    if overlay_rgb is not None:
        png_path = RESULTS_DIR / f"{stem}_gradcam.png"
        # cv2 writes BGR, so convert from RGB first.
        cv2.imwrite(str(png_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        record["gradcam_image"] = png_path.name

    try:
        json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info("Saved result artifact: %s", json_path.name)
    except Exception as exc:  # pragma: no cover - never fail a prediction on I/O
        logger.warning("Could not save result artifact: %s", exc)

    return json_path


def save_batch_csv(rows: list[dict], prefix: str = "batch") -> Path:
    """Save a list of result dicts to results/<timestamp>_<prefix>.csv."""
    import csv

    ts = _timestamp()
    csv_path = RESULTS_DIR / f"{ts}_{prefix}.csv"
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return csv_path

    # Union of keys keeps the CSV robust to per-row differences.
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info("Saved batch CSV: %s (%d rows)", csv_path.name, len(rows))
    return csv_path
