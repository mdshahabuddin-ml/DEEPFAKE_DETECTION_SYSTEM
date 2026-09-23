"""
api/advanced.py
FastAPI endpoints for advanced detection modules.

Endpoints:
    POST /advanced/analyze/audio    -> full audio forensics analysis
    POST /advanced/analyze/video    -> full multi-modal video analysis
    POST /advanced/analyze/file     -> unified analysis (auto-detect type)
    GET  /advanced/modules          -> list available modules and status

These endpoints live alongside the existing api/main.py without modifying it.
Mount them by including this router in your FastAPI app:

    from api.advanced import advanced_router
    app.include_router(advanced_router, prefix="/advanced")
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import UPLOADS_DIR, MAX_UPLOAD_MB

advanced_router = APIRouter(tags=["advanced"])

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
ALL_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class AudioAnalysisResponse(BaseModel):
    analysis_id: str
    file_name: str
    overall_prediction: str
    overall_probability: float
    overall_confidence: float
    total_segments: int
    suspicious_segments: int
    evidence: list = Field(default_factory=list)
    recommendations: list = Field(default_factory=list)
    timeline: list = Field(default_factory=list)
    processing_time: float


class VideoAnalysisResponse(BaseModel):
    analysis_id: str
    file_name: str
    overall_verdict: str
    overall_confidence: float
    overall_fake_probability: float
    modules_run: list = Field(default_factory=list)
    modules_succeeded: list = Field(default_factory=list)
    evidence: list = Field(default_factory=list)
    recommendations: list = Field(default_factory=list)
    processing_time: float


class UnifiedAnalysisResponse(BaseModel):
    analysis_id: str
    file_name: str
    file_type: str
    overall_verdict: str
    overall_confidence: float
    overall_fake_probability: float
    modules_run: list = Field(default_factory=list)
    modules_succeeded: list = Field(default_factory=list)
    evidence: list = Field(default_factory=list)
    recommendations: list = Field(default_factory=list)
    processing_time: float


class ModuleStatus(BaseModel):
    name: str
    available: bool
    description: str


def _save_upload(upload: UploadFile, allowed_ext: set) -> str:
    """Validate and save an uploaded file, return path."""
    suffix = os.path.splitext(upload.filename or "")[1].lower()
    if suffix not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_ext}",
        )
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    try:
        with dest.open("wb") as out:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to save upload")
    finally:
        upload.file.close()
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return str(dest)


@advanced_router.post("/analyze/audio", response_model=AudioAnalysisResponse)
def analyze_audio_forensics(file: UploadFile = File(...)) -> AudioAnalysisResponse:
    """Full audio forensics analysis with segmentation and timeline."""
    path = _save_upload(file, AUDIO_EXTENSIONS)
    try:
        from audio_forensics.inference import AudioInferencePipeline
        from audio_forensics.config import AudioForensicsConfig

        pipeline = AudioInferencePipeline(AudioForensicsConfig())
        result = pipeline.analyze(path)

        return AudioAnalysisResponse(
            analysis_id=result.analysis_id,
            file_name=file.filename or os.path.basename(path),
            overall_prediction=result.overall_prediction.value,
            overall_probability=result.overall_probability,
            overall_confidence=result.overall_confidence,
            total_segments=result.total_segments,
            suspicious_segments=result.suspicious_segments,
            evidence=result.evidence,
            recommendations=result.recommendations,
            timeline=result.timeline,
            processing_time=result.processing_time,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    finally:
        os.unlink(path)


@advanced_router.post("/analyze/video", response_model=VideoAnalysisResponse)
def analyze_video_advanced(
    file: UploadFile = File(...),
) -> VideoAnalysisResponse:
    """Full multi-modal video analysis with all advanced modules."""
    path = _save_upload(file, VIDEO_EXTENSIONS)
    try:
        from src.unified_orchestrator import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()
        result = orchestrator.analyze_video(path)

        return VideoAnalysisResponse(
            analysis_id=result.analysis_id,
            file_name=file.filename or os.path.basename(path),
            overall_verdict=result.overall_verdict,
            overall_confidence=result.overall_confidence,
            overall_fake_probability=result.overall_fake_probability,
            modules_run=list(result.module_results.keys()),
            modules_succeeded=[
                k for k, v in result.module_results.items() if v.success
            ],
            evidence=result.evidence,
            recommendations=result.recommendations,
            processing_time=result.processing_time,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    finally:
        os.unlink(path)


@advanced_router.post("/analyze/file", response_model=UnifiedAnalysisResponse)
def analyze_file_unified(file: UploadFile = File(...)) -> UnifiedAnalysisResponse:
    """Unified analysis - auto-detects file type and runs appropriate modules."""
    path = _save_upload(file, ALL_EXTENSIONS)
    try:
        from src.unified_orchestrator import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()
        suffix = os.path.splitext(path)[1].lower()

        if suffix in AUDIO_EXTENSIONS:
            result = orchestrator.analyze_audio(path)
            file_type = "audio"
        else:
            result = orchestrator.analyze_video(path)
            file_type = "video"

        return UnifiedAnalysisResponse(
            analysis_id=result.analysis_id,
            file_name=file.filename or os.path.basename(path),
            file_type=file_type,
            overall_verdict=result.overall_verdict,
            overall_confidence=result.overall_confidence,
            overall_fake_probability=result.overall_fake_probability,
            modules_run=list(result.module_results.keys()),
            modules_succeeded=[
                k for k, v in result.module_results.items() if v.success
            ],
            evidence=result.evidence,
            recommendations=result.recommendations,
            processing_time=result.processing_time,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    finally:
        os.unlink(path)


@advanced_router.get("/modules")
def list_modules() -> Dict[str, Any]:
    """List available advanced modules and their status."""
    modules = []

    # Audio forensics
    try:
        from audio_forensics.inference import AudioInferencePipeline
        available = True
    except ImportError:
        available = False
    modules.append(ModuleStatus(
        name="audio_forensics",
        available=available,
        description="Audio deepfake detection with segmentation and timeline",
    ))

    # Temporal analysis
    try:
        from temporal_analysis.temporal_analyzer import TemporalAnalyzer
        available = True
    except ImportError:
        available = False
    modules.append(ModuleStatus(
        name="temporal_analysis",
        available=available,
        description="Video temporal consistency and anomaly detection",
    ))

    # Multimodal fusion
    try:
        from multimodal_fusion.multimodal_analyzer import MultimodalAnalyzer
        available = True
    except ImportError:
        available = False
    modules.append(ModuleStatus(
        name="multimodal_fusion",
        available=available,
        description="Multi-modal evidence fusion with multiple methods",
    ))

    # Media forensics
    try:
        from media_forensics.forensic_analyzer import ForensicAnalyzer
        available = True
    except ImportError:
        available = False
    modules.append(ModuleStatus(
        name="media_forensics",
        available=available,
        description="File hashing, metadata analysis, integrity verification",
    ))

    # Forensic report
    try:
        from src.forensic_report_builder import ForensicReportBuilder
        available = True
    except ImportError:
        available = False
    modules.append(ModuleStatus(
        name="forensic_report",
        available=available,
        description="PDF forensic report generation with chain-of-custody",
    ))

    return {
        "total_modules": len(modules),
        "available_modules": sum(1 for m in modules if m.available),
        "modules": [m.model_dump() for m in modules],
    }
