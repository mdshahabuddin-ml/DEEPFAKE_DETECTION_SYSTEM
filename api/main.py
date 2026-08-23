"""
api/main.py
FastAPI service for the AI-Powered Explainable Deepfake Detection System.

Endpoints:
    POST /predict/image   -> classify an uploaded image (JPG/PNG/JPEG)
    POST /predict/video   -> classify an uploaded video (MP4/AVI/MOV/MKV)
    GET  /health          -> service + model health
    GET  /model/info      -> model metadata
    GET  /history         -> recent predictions from SQLite
    GET  /stats           -> aggregate dashboard statistics

Interactive Swagger docs are auto-generated at /docs (ReDoc at /redoc).

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.config import (
    ALLOWED_IMAGE_EXT, ALLOWED_VIDEO_EXT, API_KEY, API_TITLE, API_VERSION,
    CHECKPOINT_PATH, DEVICE, IMG_SIZE, MAX_UPLOAD_MB, MODEL_NAME,
    RATE_LIMIT_PER_MIN, UPLOADS_DIR,
)
from src.database import PredictionRecord, get_db
from src.inference import ModelNotFoundError, get_detector
from src.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Detect whether an uploaded image or video is REAL or DEEPFAKE.",
)

# Permissive CORS so the Streamlit UI (or any client) can call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Security guard: optional API-key auth + in-memory rate limiting.
# Applies only to inference endpoints; system/docs routes stay open.
# ---------------------------------------------------------------------------
_PROTECTED_PREFIXES = ("/predict", "/analyze", "/report", "/localize")
_rate_lock = threading.Lock()
_request_log: Dict[str, deque] = defaultdict(deque)


def _is_rate_limited(client_key: str) -> bool:
    """Sliding-window limiter: True if `client_key` exceeded the per-minute cap."""
    if RATE_LIMIT_PER_MIN <= 0:
        return False
    now = time.time()
    with _rate_lock:
        q = _request_log[client_key]
        while q and now - q[0] > 60.0:      # drop entries older than 60s
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            return True
        q.append(now)
        return False


@app.middleware("http")
async def security_guard(request: Request, call_next):
    path = request.url.path
    if any(path.startswith(p) for p in _PROTECTED_PREFIXES):
        # 1) API-key auth (only enforced when DEEPFAKE_API_KEY is configured)
        if API_KEY and request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401,
                                content={"detail": "Invalid or missing API key."})
        # 2) Rate limiting (keyed by API key when present, else client IP)
        client_key = request.headers.get("x-api-key") or (
            request.client.host if request.client else "anonymous")
        if _is_rate_limited(client_key):
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded "
                                   f"({RATE_LIMIT_PER_MIN}/min). Try again shortly."})
    return await call_next(request)


# ---------------------------------------------------------------------------
# Response schemas (Pydantic)
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    file_name: str
    media_type: str = Field(..., description="image or video")
    prediction: str = Field(..., description="REAL, FAKE, or NO_FACE")
    confidence: Optional[float] = Field(None, description="Confidence in the label [0,1]")
    fake_probability: Optional[float] = Field(None, description="Raw fake probability [0,1]")
    inference_time: float = Field(..., description="Seconds spent on inference")
    details: dict = Field(default_factory=dict, description="Extra per-media stats")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    checkpoint_present: bool
    auth_required: bool
    rate_limit_per_min: int


class ModelInfoResponse(BaseModel):
    model_name: str
    input_size: int
    device: str
    checkpoint_path: str
    checkpoint_present: bool


class HistoryItem(BaseModel):
    id: int
    file_name: str
    media_type: str
    prediction: str
    confidence: float
    fake_probability: float
    inference_time: float
    source: str
    created_at: str


class TrustReportResponse(BaseModel):
    media_type: str
    file_name: str
    verdict: str
    trust_score: float = Field(..., description="Overall authenticity score, 0-100")
    recommendation: str
    fake_probability: float
    signals: Dict[str, Optional[float]] = Field(
        ..., description="Sub-signals in [0,1]: face_consistency, temporal_consistency, lip_sync, texture"
    )
    faces_detected: int
    manipulated_frames: Optional[List[int]] = None
    frames_analyzed: int
    file_hash: str
    inference_time: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_detector():
    """Return the shared detector or raise 503 if the model is missing."""
    try:
        return get_detector()
    except ModelNotFoundError as exc:
        logger.error("Model not available: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))


def _save_upload(upload: UploadFile, allowed_ext: set) -> Path:
    """Validate extension + size and persist the upload to uploads/."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed_ext)}",
        )

    dest = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    try:
        with dest.open("wb") as out:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.",
                    )
                out.write(chunk)
    finally:
        upload.file.close()

    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return dest


def _store(result: dict, media_type: str) -> None:
    """Persist a successful prediction to the history DB (best-effort)."""
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
            source="api",
        ))
    except Exception as exc:  # pragma: no cover - never fail the request on logging
        logger.warning("Failed to store prediction: %s", exc)


def _to_response(result: dict, media_type: str) -> PredictionResponse:
    known = {"prediction", "fake_probability", "confidence", "inference_time",
             "file_name", "message"}
    details = {k: v for k, v in result.items() if k not in known and k != "threshold"}
    if "message" in result:
        details["message"] = result["message"]
    return PredictionResponse(
        file_name=result.get("file_name", "unknown"),
        media_type=media_type,
        prediction=result["prediction"],
        confidence=result.get("confidence"),
        fake_probability=result.get("fake_probability"),
        inference_time=result["inference_time"],
        details=details,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness + model-availability probe."""
    present = CHECKPOINT_PATH.exists()
    loaded = False
    if present:
        try:
            get_detector()
            loaded = True
        except Exception:
            loaded = False
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        device=DEVICE,
        checkpoint_present=present,
        auth_required=bool(API_KEY),
        rate_limit_per_min=RATE_LIMIT_PER_MIN,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
def model_info() -> ModelInfoResponse:
    """Return model metadata."""
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        input_size=IMG_SIZE,
        device=DEVICE,
        checkpoint_path=str(CHECKPOINT_PATH),
        checkpoint_present=CHECKPOINT_PATH.exists(),
    )


@app.post("/predict/image", response_model=PredictionResponse, tags=["prediction"])
def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    """Classify a single uploaded image as REAL or DEEPFAKE."""
    detector = _require_detector()
    path = _save_upload(file, ALLOWED_IMAGE_EXT)
    try:
        result = detector.predict_image(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    finally:
        path.unlink(missing_ok=True)
    result["file_name"] = file.filename or path.name
    _store(result, "image")
    return _to_response(result, "image")


@app.post("/predict/video", response_model=PredictionResponse, tags=["prediction"])
def predict_video(file: UploadFile = File(...)) -> PredictionResponse:
    """Classify an uploaded video using per-frame inference + majority voting."""
    detector = _require_detector()
    path = _save_upload(file, ALLOWED_VIDEO_EXT)
    try:
        result = detector.predict_video(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid video: {exc}")
    finally:
        path.unlink(missing_ok=True)
    result["file_name"] = file.filename or path.name
    _store(result, "video")
    return _to_response(result, "video")


@app.get("/history", response_model=List[HistoryItem], tags=["history"])
def history(limit: int = 50) -> List[HistoryItem]:
    """Return the most recent predictions from SQLite."""
    rows = get_db().recent(limit=limit)
    return [HistoryItem(**row) for row in rows]


@app.get("/stats", tags=["history"])
def stats() -> dict:
    """Aggregate statistics for dashboards."""
    return get_db().stats()


class FaceLocalization(BaseModel):
    box: List[int] = Field(..., description="[x0, y0, x1, y1]")
    prediction: str
    confidence: float
    fake_probability: float


class LocalizationResponse(BaseModel):
    file_name: str
    faces_detected: int
    faces: List[FaceLocalization]


@app.post("/localize/image", response_model=LocalizationResponse, tags=["prediction"])
def localize_image(file: UploadFile = File(...)) -> LocalizationResponse:
    """Detect and score every face in an image (multi-face localization)."""
    detector = _require_detector()
    path = _save_upload(file, ALLOWED_IMAGE_EXT)
    try:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image.")
        faces = detector.localize_faces(bgr)
    finally:
        path.unlink(missing_ok=True)
    return LocalizationResponse(
        file_name=file.filename or path.name,
        faces_detected=len(faces),
        faces=[FaceLocalization(**f) for f in faces],
    )


@app.post("/analyze/image", response_model=TrustReportResponse, tags=["trust-score"])
def analyze_image(file: UploadFile = File(...)) -> TrustReportResponse:
    """Full multi-signal Trust Score assessment for an image."""
    from src.trust_score import analyze_image as trust_image
    detector = _require_detector()
    path = _save_upload(file, ALLOWED_IMAGE_EXT)
    try:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image.")
        report = trust_image(detector, bgr,
                             file_name=file.filename or path.name, file_path=str(path))
    finally:
        path.unlink(missing_ok=True)
    return TrustReportResponse(**report.to_dict())


@app.post("/analyze/video", response_model=TrustReportResponse, tags=["trust-score"])
def analyze_video_trust(file: UploadFile = File(...)) -> TrustReportResponse:
    """Full multi-signal Trust Score assessment for a video."""
    from src.trust_score import analyze_video as trust_video
    detector = _require_detector()
    path = _save_upload(file, ALLOWED_VIDEO_EXT)
    try:
        report = trust_video(detector, str(path), file_name=file.filename or path.name)
    finally:
        path.unlink(missing_ok=True)
    return TrustReportResponse(**report.to_dict())


@app.post("/report/image", tags=["trust-score"])
def report_image(file: UploadFile = File(...)):
    """Generate and return a downloadable PDF authenticity report for an image."""
    from src.report import generate_pdf_report
    from src.trust_score import analyze_image as trust_image
    detector = _require_detector()
    path = _save_upload(file, ALLOWED_IMAGE_EXT)
    try:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image.")
        report = trust_image(detector, bgr,
                             file_name=file.filename or path.name, file_path=str(path))
        pdf_path = generate_pdf_report(report)
    finally:
        path.unlink(missing_ok=True)
    return FileResponse(str(pdf_path), media_type="application/pdf",
                        filename=pdf_path.name)


@app.get("/", tags=["system"])
def root() -> dict:
    """Landing endpoint pointing to the docs."""
    return {"service": API_TITLE, "version": API_VERSION, "docs": "/docs"}
