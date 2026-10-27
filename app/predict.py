"""
app/predict.py
Prediction helpers shared by the Streamlit pages.

Wraps the core DeepfakeDetector so the UI stays thin: save an upload, run
inference, and record the result to the SQLite history in one call.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional

# Ensure the project root is importable when Streamlit runs this as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.database import PredictionRecord, get_db
from src.logger import get_logger

logger = get_logger("streamlit")


def save_temp_upload(uploaded_file) -> str:
    """Persist a Streamlit UploadedFile to a temp path and return it."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def store_result(result: dict, media_type: str) -> None:
    """Record a prediction to history (skips NO_FACE / errors)."""
    if result.get("prediction") in (None, "NO_FACE"):
        return
    try:
        get_db().add(PredictionRecord(
            file_name=result.get("file_name", "unknown"),
            media_type=media_type,
            prediction=result["prediction"],
            confidence=float(result.get("confidence") or 0.0),
            fake_probability=float(result.get("fake_probability") or 0.0),
            inference_time=float(result.get("inference_time") or 0.0),
            source="app",
        ))
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not store prediction: %s", exc)


def run_image(detector, uploaded_file) -> dict:
    """Run image inference on a Streamlit upload and log it."""
    path = save_temp_upload(uploaded_file)
    try:
        result = detector.predict_image(path)
    finally:
        Path(path).unlink(missing_ok=True)
    result["file_name"] = uploaded_file.name
    store_result(result, "image")
    return result


def run_video(detector, uploaded_file, max_frames: Optional[int] = None) -> dict:
    """Run video inference on a Streamlit upload and log it."""
    path = save_temp_upload(uploaded_file)
    try:
        if max_frames is not None:
            result = detector.predict_video(path, max_frames=max_frames)
        else:
            result = detector.predict_video(path)
    finally:
        Path(path).unlink(missing_ok=True)
    result["file_name"] = uploaded_file.name
    store_result(result, "video")
    return result


def run_audio(detector, uploaded_file, denoise: bool = False,
              window_sec: float = 4.0, hop_sec=None) -> dict:
    """Run audio inference on a Streamlit upload and log it.

    The detector is an ``AudioDeepfakeDetector`` (see
    ``src/audio_inference.py``). The result dict uses the same keys as the
    image/video path (prediction, probability_fake / fake_probability,
    confidence, inference_time, file_name) so ``store_result`` and the
    History / Dashboard pages work unchanged with ``media_type="audio"``.
    """
    path = save_temp_upload(uploaded_file)
    try:
        result = detector.predict_audio(path, denoise=denoise,
                                        window_sec=window_sec, hop_sec=hop_sec)
    finally:
        Path(path).unlink(missing_ok=True)
    result["file_name"] = uploaded_file.name
    store_result(result, "audio")
    return result
