"""
app/main.py
FastAPI backend for the Deepfake Detection System.

Wraps the existing src.inference.DeepfakeDetector so the API and the
Streamlit UI share the exact same model-loading and prediction logic —
no duplicated code.

Run from the project root:
    uvicorn app.main:app --reload

Then open:
    http://127.0.0.1:8000/docs   (interactive Swagger UI)
"""

from __future__ import annotations

import sys
import tempfile
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Make the project root importable when uvicorn executes this as a module.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import CHECKPOINT_PATH, DEFAULT_THRESHOLD, DEVICE, MODEL_NAME, VIDEO_MAX_FRAMES
from src.inference import ModelNotFoundError, get_detector

app = FastAPI(
    title="Deepfake Detection API",
    description="AI-powered explainable deepfake detection — image and video inference.",
    version="1.0.0",
)

# Allow the Streamlit app (or any local frontend) to call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_image_upload(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a BGR OpenCV array."""
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")
    return bgr


def _get_detector_or_503():
    """Return the shared detector, or raise a clean 503 if the checkpoint is missing."""
    try:
        return get_detector()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Meta"])
def root():
    """Basic API info and a pointer to the docs."""
    return {
        "name": "Deepfake Detection API",
        "model": MODEL_NAME,
        "device": DEVICE,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Meta"])
def health():
    """
    Health check. Reports whether the model checkpoint is present and loadable
    — useful for uptime checks / container orchestration probes.
    """
    checkpoint_exists = Path(CHECKPOINT_PATH).exists()
    model_loaded = False
    error = None
    if checkpoint_exists:
        try:
            _get_detector_or_503()
            model_loaded = True
        except HTTPException as exc:
            error = exc.detail
    return {
        "status": "ok" if model_loaded else "degraded",
        "checkpoint_found": checkpoint_exists,
        "model_loaded": model_loaded,
        "model": MODEL_NAME,
        "device": DEVICE,
        "error": error,
    }


@app.post("/predict/image", tags=["Prediction"])
async def predict_image(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0,
        description="Override the default decision threshold for this request.",
    ),
):
    """
    Predict REAL / FAKE for a single uploaded image.

    Returns the same result shape used internally by the app
    (prediction, fake_probability, confidence, inference_time, ...).
    """
    detector = _get_detector_or_503()
    if threshold is not None:
        detector.threshold = threshold  # temporary override for this call

    contents = await file.read()
    bgr = _read_image_upload(contents)
    result = detector.predict_image_array(bgr)
    result["file_name"] = file.filename

    if threshold is not None:
        detector.threshold = DEFAULT_THRESHOLD  # restore default

    return JSONResponse(content=result)


@app.post("/predict/video", tags=["Prediction"])
async def predict_video(
    file: UploadFile = File(...),
    max_frames: int = Query(
        VIDEO_MAX_FRAMES, ge=1, le=64,
        description="Number of frames to sample evenly across the video.",
    ),
):
    """
    Predict REAL / FAKE for an uploaded video using frame sampling +
    majority voting, matching the Streamlit Video Detection page.
    """
    detector = _get_detector_or_503()

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = detector.predict_video(tmp_path, max_frames=max_frames)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    result["file_name"] = file.filename
    return JSONResponse(content=result)


@app.post("/predict/localize", tags=["Prediction"])
async def predict_localize(file: UploadFile = File(...)):
    """
    Detect and score EVERY face in an image (not just the largest) —
    useful for group photos or multi-face frames.
    """
    detector = _get_detector_or_503()
    contents = await file.read()
    bgr = _read_image_upload(contents)
    faces = detector.localize_faces(bgr)
    return JSONResponse(content={"file_name": file.filename, "num_faces": len(faces),
                                  "faces": faces})