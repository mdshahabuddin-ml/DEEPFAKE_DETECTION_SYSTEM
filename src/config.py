"""
src/config.py
Central configuration for the AI-Powered Explainable Deepfake Detection System.

All paths, model settings, and runtime constants live here so every module
(inference, API, Streamlit, webcam, database) shares a single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# config.py lives in <project_root>/src/, so the root is one level up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

MODELS_DIR: Path = PROJECT_ROOT / "models"
UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Create runtime directories on import so nothing downstream has to.
for _d in (MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, LOGS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
MODEL_NAME: str = os.getenv("DEEPFAKE_MODEL_NAME", "xception")
CHECKPOINT_PATH = Path(
    os.getenv(
        "DEEPFAKE_CHECKPOINT",
        str(MODELS_DIR / "best_xception_multidataset.pth")
    )
)
IMG_SIZE: int = 299
DEFAULT_THRESHOLD: float = 0.47 # fake if probability >= threshold

# ---------------------------------------------------------------------------
# Checkpoint auto-download (optional) -- used when the local file is missing.
# Set ONE of these to fetch weights automatically on first startup:
#   DEEPFAKE_CHECKPOINT_URL : a direct https URL to a .pth file
#   DEEPFAKE_HF_REPO        : a Hugging Face Hub repo id (e.g. "user/deepfake-xception")
#                             optionally with DEEPFAKE_HF_FILENAME (defaults to the
#                             checkpoint's basename) and DEEPFAKE_HF_REVISION.
# ---------------------------------------------------------------------------
CHECKPOINT_URL: str = os.getenv("DEEPFAKE_CHECKPOINT_URL", "").strip()
HF_REPO: str = os.getenv("DEEPFAKE_HF_REPO", "").strip()
HF_FILENAME: str = os.getenv("DEEPFAKE_HF_FILENAME", "").strip() or CHECKPOINT_PATH.name
HF_REVISION: str = os.getenv("DEEPFAKE_HF_REVISION", "").strip() or None

# ImageNet normalization used by the training transforms.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
# "mtcnn"  -> facenet-pytorch MTCNN (accurate, used for image/video)
# "haar"   -> OpenCV Haar cascade  (fast, used as a fallback / for webcam)
FACE_DETECTOR: str = os.getenv("DEEPFAKE_FACE_DETECTOR", "mtcnn")
FACE_MARGIN: float = 0.2  # padding ratio around a detected face box

# ---------------------------------------------------------------------------
# Video / webcam
# ---------------------------------------------------------------------------
VIDEO_MAX_FRAMES: int = 16       # frames sampled per uploaded video
WEBCAM_INFER_EVERY_N: int = 2    # run the model every N webcam frames
WEBCAM_SMOOTH_WINDOW: int = 8    # rolling-average window for a stable label

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: Path = Path(os.getenv("DEEPFAKE_DB", str(PROJECT_ROOT / "results" / "predictions.db")))

# ---------------------------------------------------------------------------
# User authentication (login page)
# ---------------------------------------------------------------------------
AUTH_DB: Path = Path(os.getenv("DEEPFAKE_AUTH_DB", str(PROJECT_ROOT / "results" / "users.db")))
# A default admin account is created on first run if none exists. CHANGE THIS
# in production via the environment variables below.
ADMIN_USER: str = os.getenv("DEEPFAKE_ADMIN_USER", "admin")
ADMIN_PASSWORD: str = os.getenv("DEEPFAKE_ADMIN_PASSWORD", "admin123")

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_TITLE: str = "AI-Powered Explainable Deepfake Detection API"
API_VERSION: str = "1.0.0"
MAX_UPLOAD_MB: int = 100

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

# API security (both optional / off by default):
#   DEEPFAKE_API_KEY       : if set, prediction endpoints require header X-API-Key
#   DEEPFAKE_RATE_LIMIT    : max requests/minute per client (0 = unlimited)
API_KEY: str = os.getenv("DEEPFAKE_API_KEY", "").strip()
RATE_LIMIT_PER_MIN: int = int(os.getenv("DEEPFAKE_RATE_LIMIT", "0"))

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Return 'cuda' when a GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE: str = get_device()


def label_from_prob(prob: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Map a fake-probability to a human-readable REAL/FAKE label."""
    return "FAKE" if prob >= threshold else "REAL"


def confidence_from_prob(prob: float, threshold: float = DEFAULT_THRESHOLD) -> float:
    """
    Confidence in the chosen label, in [0, 1].
    FAKE -> prob, REAL -> 1 - prob.
    """
    return prob if prob >= threshold else 1.0 - prob
