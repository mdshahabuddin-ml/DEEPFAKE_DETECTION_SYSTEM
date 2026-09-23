"""
Streamlit web app for the deepfake detection project.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os
import time
import tempfile
from datetime import datetime
from typing import Optional, List, Tuple

import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.append(".")

from src.model import build_model, load_checkpoint
from src.dataset import get_val_transforms
from src.gradcam_utils import generate_gradcam_overlay
from src.inference import FaceDetector
from src.realtime_webcam import detect_largest_face, predict_fake_prob, draw_overlay
from src.audio_infer import (
    load_audio_detector, predict_from_array,
    DEFAULT_AUDIO_MODEL, DEFAULT_AUDIO_CHECKPOINT,
)
from src.audio_dataset import SAMPLE_RATE
from src.fusion import fuse_scores, load_audio_from_video
from src.database import get_db, PredictionRecord
from src.logger import get_logger

logger = get_logger("streamlit_app")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME: str = "xception"
CHECKPOINT_PATH: str = "models/best_xception_celebdf.pth"
IMG_SIZE: int = 299
DEFAULT_THRESHOLD: float = 0.47

AUDIO_MODEL_NAME: str = DEFAULT_AUDIO_MODEL
AUDIO_CHECKPOINT_PATH: str = DEFAULT_AUDIO_CHECKPOINT

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark theme CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
    section[data-testid="stSidebar"] .stRadio label { color: #e0e0e0; font-size: 14px; padding: 4px 0; }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label { color: #e0e0e0; }

    /* Sidebar header */
    .sidebar-header { padding: 0 0 10px 0; border-bottom: 1px solid #333; margin-bottom: 15px; }
    .sidebar-header h2 { color: #fff; margin: 0; font-size: 22px; font-weight: 700; }
    .sidebar-header p { color: #888; margin: 4px 0 0 0; font-size: 13px; }

    /* Nav items */
    .nav-item { padding: 8px 12px; border-radius: 8px; margin: 2px 0; cursor: pointer; font-size: 14px; }
    .nav-item-active { background: rgba(102, 126, 234, 0.2); color: #667eea; font-weight: 600; }
    .nav-item-inactive { color: #888; }

    /* Cards */
    .info-card {
        background: #1e1e2e; padding: 20px; border-radius: 12px;
        border-top: 3px solid #667eea; text-align: center; min-height: 100px;
    }
    .info-card h3 { color: #888; margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
    .info-card p { color: #667eea; margin: 8px 0 0 0; font-size: 22px; font-weight: 700; }

    .cap-card {
        background: #1e1e2e; padding: 25px; border-radius: 12px;
        border: 1px solid #2a2a3e; min-height: 160px;
    }
    .cap-card h4 { color: #fff; margin: 0 0 8px 0; font-size: 16px; }
    .cap-card p { color: #888; margin: 0; font-size: 13px; line-height: 1.5; }

    /* Hero */
    .hero-section {
        background: linear-gradient(135deg, #1a1a3e 0%, #2d2b55 50%, #1a1a3e 100%);
        padding: 40px; border-radius: 16px; margin-bottom: 30px;
        border: 1px solid #333;
    }
    .hero-badge {
        display: inline-block; background: rgba(102,126,234,0.2); color: #667eea;
        padding: 4px 14px; border-radius: 20px; font-size: 11px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;
    }
    .hero-section h1 { color: #4da3ff; margin: 0 0 15px 0; font-size: 32px; font-weight: 700; }
    .hero-section p { color: #b0b0b0; margin: 0; font-size: 15px; line-height: 1.6; }

    /* Verdict badges */
    .verdict-real { color: #28a745; }
    .verdict-fake { color: #dc3545; }

    /* Stat cards for dashboard */
    .stat-card {
        background: #1e1e2e; padding: 25px; border-radius: 12px;
        border-left: 4px solid #667eea; text-align: center;
    }
    .stat-card h2 { color: #667eea; margin: 0; font-size: 36px; font-weight: 700; }
    .stat-card p { color: #888; margin: 8px 0 0 0; font-size: 13px; }

    /* Table styling */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Section headers */
    .section-label { color: #666; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px; font-weight: 600; }

    /* Override default streamlit button */
    .stButton > button { border-radius: 8px; }

    /* Image Detection - Forensics UI */
    .forensics-header {
        padding: 32px 36px; border-radius: 16px; margin-bottom: 28px;
        background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0f1923 100%);
        border: 1px solid #2a3a4e; position: relative; overflow: hidden;
    }
    .forensics-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    .forensics-header .badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(102,126,234,0.12); color: #667eea;
        padding: 5px 14px; border-radius: 20px; font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px;
        border: 1px solid rgba(102,126,234,0.2);
    }
    .forensics-header h1 { color: #e8edf3; margin: 0 0 8px 0; font-size: 26px; font-weight: 700; letter-spacing: -0.3px; }
    .forensics-header p { color: #7a8ba0; margin: 0; font-size: 14px; line-height: 1.5; }

    .upload-forensics-card {
        background: #141c26; border: 1px dashed #2a3a4e; border-radius: 14px;
        padding: 36px 28px; text-align: center; transition: all 0.25s ease;
        cursor: pointer; position: relative; overflow: hidden;
    }
    .upload-forensics-card:hover { border-color: #667eea; background: #182230; }
    .upload-forensics-card .icon-circle {
        width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 18px;
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.15);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .upload-forensics-card h3 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
    .upload-forensics-card .sub { color: #5a6b7e; font-size: 13px; margin: 0 0 14px 0; }
    .upload-forensics-card .divider { color: #3a4a5e; font-size: 12px; margin: 10px 0; }
    .upload-forensics-card .hint { color: #4a5a6e; font-size: 11px; margin: 8px 0 0 0; }

    .workflow-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 16px 0; margin: 20px 0;
    }
    .workflow-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 20px;
    }
    .workflow-step .step-icon {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 14px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .workflow-step .step-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .workflow-step .step-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .workflow-step .step-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #4a5a6e; font-weight: 600; }
    .workflow-step .step-label.active { color: #667eea; }
    .workflow-step .step-label.done { color: #28a745; }
    .workflow-arrow { color: #2a3a4e; font-size: 16px; margin: 0 2px; padding-bottom: 18px; }

    .empty-state-forensics {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 48px 32px; text-align: center; margin-top: 8px;
    }
    .empty-state-forensics .icon { font-size: 40px; margin-bottom: 16px; opacity: 0.4; }
    .empty-state-forensics h3 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
    .empty-state-forensics p { color: #5a6b7e; margin: 0; font-size: 13px; line-height: 1.6; max-width: 400px; margin: 0 auto; }

    .analysis-card {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px; padding: 24px;
    }
    .analysis-card h4 {
        color: #7a8ba0; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px;
        margin: 0 0 16px 0; font-weight: 600; padding-bottom: 10px; border-bottom: 1px solid #1e2e3e;
    }

    .verdict-badge-lg {
        display: inline-flex; align-items: center; gap: 8px; padding: 10px 22px;
        border-radius: 10px; font-size: 18px; font-weight: 700; letter-spacing: 1px;
    }
    .verdict-badge-lg.real { background: rgba(40,167,69,0.1); color: #28a745; border: 1px solid rgba(40,167,69,0.25); }
    .verdict-badge-lg.fake { background: rgba(220,53,69,0.1); color: #dc3545; border: 1px solid rgba(220,53,69,0.25); }
    .verdict-badge-lg.uncertain { background: rgba(255,193,7,0.1); color: #ffc107; border: 1px solid rgba(255,193,7,0.25); }

    .prob-row { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
    .prob-row .label { color: #7a8ba0; font-size: 12px; font-weight: 600; width: 42px; text-align: right; }
    .prob-bar-track { flex: 1; height: 8px; background: #1e2e3e; border-radius: 4px; overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
    .prob-bar-fill.real { background: linear-gradient(90deg, #1a7a34, #28a745); }
    .prob-bar-fill.fake { background: linear-gradient(90deg, #c0392b, #dc3545); }
    .prob-row .value { color: #e0e0e0; font-size: 13px; font-weight: 600; width: 52px; }

    .forensic-detail-row {
        display: flex; justify-content: space-between; padding: 8px 0;
        border-bottom: 1px solid #1a2636; font-size: 13px;
    }
    .forensic-detail-row:last-child { border-bottom: none; }
    .forensic-detail-row .dk { color: #5a6b7e; }
    .forensic-detail-row .dv { color: #c0c8d4; font-weight: 500; }

    .summary-text { color: #7a8ba0; font-size: 13px; line-height: 1.7; margin: 0; }

    .meta-chip {
        display: inline-flex; align-items: center; gap: 5px;
        background: #1a2636; padding: 4px 10px; border-radius: 6px;
        font-size: 11px; color: #7a8ba0; margin: 2px;
    }

    /* Video Detection - Forensics UI */
    .video-upload-card {
        background: #141c26; border: 1px dashed #2a3a4e; border-radius: 14px;
        padding: 36px 28px; text-align: center; transition: all 0.25s ease;
    }
    .video-upload-card:hover { border-color: #667eea; background: #182230; }
    .video-upload-card .icon-circle {
        width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 18px;
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.15);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .video-upload-card h3 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
    .video-upload-card .sub { color: #5a6b7e; font-size: 13px; margin: 0 0 14px 0; }
    .video-upload-card .formats { color: #4a5a6e; font-size: 11px; margin: 8px 0 0 0; }

    .pipeline-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 16px 0; margin: 20px 0; flex-wrap: wrap;
    }
    .pipeline-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 12px;
    }
    .pipeline-step .step-icon {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 14px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .pipeline-step .step-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .pipeline-step .step-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .pipeline-step .step-icon.skip { border-color: #3a4a5e; background: #1a2636; color: #3a4a5e; opacity: 0.5; }
    .pipeline-step .step-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #4a5a6e; font-weight: 600; }
    .pipeline-step .step-label.active { color: #667eea; }
    .pipeline-step .step-label.done { color: #28a745; }
    .pipeline-step .step-label.skip { color: #3a4a5e; opacity: 0.5; }
    .pipeline-arrow { color: #2a3a4e; font-size: 14px; margin: 0 2px; padding-bottom: 18px; }

    .config-card {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px; padding: 24px;
    }
    .config-card h4 {
        color: #7a8ba0; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px;
        margin: 0 0 16px 0; font-weight: 600; padding-bottom: 10px; border-bottom: 1px solid #1e2e3e;
    }

    .fusion-status {
        display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px;
        border-radius: 8px; font-size: 12px; font-weight: 600;
    }
    .fusion-status.enabled { background: rgba(102,126,234,0.1); color: #667eea; border: 1px solid rgba(102,126,234,0.2); }
    .fusion-status.disabled { background: rgba(90,107,126,0.1); color: #5a6b7e; border: 1px solid rgba(90,107,126,0.2); }

    .video-empty-state {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 48px 32px; text-align: center; margin-top: 8px;
    }
    .video-empty-state .icon { font-size: 40px; margin-bottom: 16px; opacity: 0.4; }
    .video-empty-state h3 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
    .video-empty-state p { color: #5a6b7e; margin: 0; font-size: 13px; line-height: 1.6; max-width: 400px; margin: 0 auto; }
    .video-empty-state .status-dot {
        display: inline-flex; align-items: center; gap: 6px; margin-top: 14px;
        color: #28a745; font-size: 12px; font-weight: 500;
    }
    .video-empty-state .status-dot::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%; background: #28a745;
    }

    .frame-stat-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px; padding: 14px 16px; text-align: center;
    }
    .frame-stat-card .val { color: #e0e0e0; font-size: 20px; font-weight: 700; margin: 0; }
    .frame-stat-card .lbl { color: #5a6b7e; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 4px 0 0 0; font-weight: 600; }


    /* Audio Detection - Forensics UI */
    .audio-upload-card {
        background: #141c26; border: 1px dashed #2a3a4e; border-radius: 14px;
        padding: 36px 28px; text-align: center; transition: all 0.25s ease;
    }
    .audio-upload-card:hover { border-color: #667eea; background: #182230; }
    .audio-upload-card .icon-circle {
        width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 18px;
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.15);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .audio-upload-card h3 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
    .audio-upload-card .sub { color: #5a6b7e; font-size: 13px; margin: 0 0 14px 0; }
    .audio-upload-card .formats { color: #4a5a6e; font-size: 11px; margin: 8px 0 0 0; }

    .audio-pipeline-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 16px 0; margin: 20px 0; flex-wrap: wrap;
    }
    .audio-pipeline-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 12px;
    }
    .audio-pipeline-step .step-icon {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 14px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .audio-pipeline-step .step-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .audio-pipeline-step .step-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .audio-pipeline-step .step-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #4a5a6e; font-weight: 600; }
    .audio-pipeline-step .step-label.active { color: #667eea; }
    .audio-pipeline-step .step-label.done { color: #28a745; }
    .audio-pipeline-arrow { color: #2a3a4e; font-size: 14px; margin: 0 2px; padding-bottom: 18px; }

    .audio-empty-state {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 48px 32px; text-align: center; margin-top: 8px;
    }
    .audio-empty-state .icon { font-size: 40px; margin-bottom: 16px; opacity: 0.4; }
    .audio-empty-state h3 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
    .audio-empty-state p { color: #5a6b7e; margin: 0; font-size: 13px; line-height: 1.6; max-width: 400px; margin: 0 auto; }
    .audio-empty-state .status-dot {
        display: inline-flex; align-items: center; gap: 6px; margin-top: 14px;
        color: #28a745; font-size: 12px; font-weight: 500;
    }
    .audio-empty-state .status-dot::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%; background: #28a745;
    }

    .audio-stat-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px; padding: 14px 16px; text-align: center;
    }
    .audio-stat-card .val { color: #e0e0e0; font-size: 18px; font-weight: 700; margin: 0; }
    .audio-stat-card .lbl { color: #5a6b7e; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 4px 0 0 0; font-weight: 600; }

    .audio-interpretation {
        padding: 16px; background: #0f1620; border-radius: 10px; border: 1px solid #1a2636;
    }
    .audio-interpretation p { color: #7a8ba0; font-size: 13px; line-height: 1.7; margin: 0; }


    /* Grad-CAM Explainability - Forensics UI */
    .gradcam-upload-card {
        background: #141c26; border: 1px dashed #2a3a4e; border-radius: 14px;
        padding: 36px 28px; text-align: center; transition: all 0.25s ease;
    }
    .gradcam-upload-card:hover { border-color: #667eea; background: #182230; }
    .gradcam-upload-card .icon-circle {
        width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 18px;
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.15);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .gradcam-upload-card h3 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
    .gradcam-upload-card .sub { color: #5a6b7e; font-size: 13px; margin: 0 0 14px 0; }
    .gradcam-upload-card .formats { color: #4a5a6e; font-size: 11px; margin: 8px 0 0 0; }

    .gradcam-pipeline-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 16px 0; margin: 20px 0; flex-wrap: wrap;
    }
    .gradcam-pipeline-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 12px;
    }
    .gradcam-pipeline-step .step-icon {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 14px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .gradcam-pipeline-step .step-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .gradcam-pipeline-step .step-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .gradcam-pipeline-step .step-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #4a5a6e; font-weight: 600; }
    .gradcam-pipeline-step .step-label.active { color: #667eea; }
    .gradcam-pipeline-step .step-label.done { color: #28a745; }
    .gradcam-pipeline-arrow { color: #2a3a4e; font-size: 14px; margin: 0 2px; padding-bottom: 18px; }

    .gradcam-empty-state {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 48px 32px; text-align: center; margin-top: 8px;
    }
    .gradcam-empty-state .icon { font-size: 40px; margin-bottom: 16px; opacity: 0.4; }
    .gradcam-empty-state h3 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
    .gradcam-empty-state p { color: #5a6b7e; margin: 0; font-size: 13px; line-height: 1.6; max-width: 400px; margin: 0 auto; }
    .gradcam-empty-state .status-dot {
        display: inline-flex; align-items: center; gap: 6px; margin-top: 14px;
        color: #28a745; font-size: 12px; font-weight: 500;
    }
    .gradcam-empty-state .status-dot::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%; background: #28a745;
    }

    .gradcam-viz-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .gradcam-viz-card .label {
        color: #7a8ba0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
        margin: 0 0 8px 0; font-weight: 600;
    }
    .gradcam-viz-card .desc {
        color: #5a6b7e; font-size: 11px; margin: 8px 0 0 0; line-height: 1.4;
    }

    .legend-item {
        display: flex; align-items: center; gap: 8px; padding: 6px 0;
    }
    .legend-swatch {
        width: 32px; height: 12px; border-radius: 3px;
    }
    .legend-label {
        color: #7a8ba0; font-size: 12px;
    }

    .gradcam-interpretation {
        padding: 16px; background: #0f1620; border-radius: 10px; border: 1px solid #1a2636;
    }
    .gradcam-interpretation p { color: #7a8ba0; font-size: 13px; line-height: 1.7; margin: 0; }


    /* About Page - Forensics UI */
    .about-hero {
        background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0f1923 100%);
        border: 1px solid #1e2e3e; border-radius: 16px; padding: 36px;
        margin-bottom: 24px; text-align: center;
    }
    .about-hero h1 { color: #e8edf3; margin: 0 0 8px 0; font-size: 26px; font-weight: 700; }
    .about-hero p { color: #7a8ba0; margin: 0; font-size: 14px; line-height: 1.5; }

    .about-intro-card {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 28px; margin-bottom: 20px;
    }
    .about-intro-card h3 { color: #e0e0e0; margin: 0 0 12px 0; font-size: 18px; font-weight: 600; }
    .about-intro-card p { color: #7a8ba0; font-size: 14px; line-height: 1.7; margin: 0; }

    .capability-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 20px; text-align: left; transition: border-color 0.2s ease;
        min-height: 140px;
    }
    .capability-card:hover { border-color: #2a3a4e; }
    .capability-card .cap-icon {
        width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
        background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.2); margin-bottom: 12px;
    }
    .capability-card h4 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 14px; font-weight: 600; }
    .capability-card p { color: #5a6b7e; font-size: 12px; line-height: 1.5; margin: 0; }

    .tech-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; text-align: center;
    }
    .tech-card .tech-name { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }
    .tech-card .tech-desc { color: #5a6b7e; font-size: 11px; margin: 4px 0 0 0; }

    .principle-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 24px; text-align: center;
    }
    .principle-card .p-icon {
        width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
        margin: 0 auto 14px; font-size: 20px;
    }
    .principle-card h4 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 15px; font-weight: 600; }
    .principle-card p { color: #5a6b7e; font-size: 12px; line-height: 1.6; margin: 0; }

    .workflow-step {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .workflow-step .step-num {
        color: #667eea; font-size: 11px; font-weight: 700; margin: 0 0 6px 0;
        letter-spacing: 1px;
    }
    .workflow-step .step-title { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }

    .highlight-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; display: flex; align-items: center; gap: 10px;
    }
    .highlight-card .h-dot {
        width: 6px; height: 6px; border-radius: 50%; background: #667eea; flex-shrink: 0;
    }
    .highlight-card .h-text { color: #c0c8d4; font-size: 13px; }

    .about-section-title {
        color: #e0e0e0; font-size: 18px; font-weight: 700; margin: 0 0 8px 0;
    }
    .about-section-sub {
        color: #5a6b7e; font-size: 12px; margin: 0 0 20px 0;
    }

    .config-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 0; border-bottom: 1px solid #1a2636; font-size: 13px;
    }
    .config-row:last-child { border-bottom: none; }
    .config-row .ck { color: #5a6b7e; }
    .config-row .cv { color: #c0c8d4; font-weight: 500; }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> Tuple[torch.nn.Module, str]:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(MODEL_NAME, pretrained=False)
        model = load_checkpoint(model, CHECKPOINT_PATH)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise


@st.cache_resource
def load_face_detector() -> FaceDetector:
    try:
        return FaceDetector(backend="auto")
    except Exception as e:
        logger.error("Failed to load face detector: %s", e)
        raise


@st.cache_resource
def load_audio_model():
    try:
        return load_audio_detector(AUDIO_MODEL_NAME, AUDIO_CHECKPOINT_PATH)
    except Exception as e:
        logger.error("Failed to load audio model: %s", e)
        raise


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def extract_face(bgr_image: np.ndarray, face_detector: FaceDetector) -> Optional[np.ndarray]:
    try:
        return face_detector.crop_largest(bgr_image)
    except Exception as e:
        logger.warning("Face extraction failed: %s", e)
        return None


def predict(face_bgr: np.ndarray, model: torch.nn.Module, device: str, transform) -> float:
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        augmented = transform(image=face_rgb)
        tensor = augmented["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        return prob
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return 0.0


def sample_video_frames(video_path: str, max_frames: int = 16) -> List[np.ndarray]:
    frames = []
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return frames
        idxs = np.linspace(0, total - 1, min(max_frames, total)).astype(int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    except Exception as e:
        logger.error("Frame sampling failed: %s", e)
    finally:
        if cap is not None:
            cap.release()
    return frames


def store_prediction(file_name: str, media_type: str, prediction: str,
                     confidence: float, fake_prob: float, inference_time: float):
    """Store a prediction to the history database."""
    try:
        db = get_db()
        db.add(PredictionRecord(
            file_name=file_name,
            media_type=media_type,
            prediction=prediction,
            confidence=confidence,
            fake_probability=fake_prob,
            inference_time=inference_time,
            source="app",
        ))
    except Exception as e:
        logger.warning("Failed to store prediction: %s", e)


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
if not os.path.exists(CHECKPOINT_PATH):
    st.error(f"Checkpoint not found at `{CHECKPOINT_PATH}`.")
    st.stop()

model, device = load_model()
face_cascade = load_face_detector()
transform = get_val_transforms(IMG_SIZE)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>Deepfake Detector</h2>
    </div>
    """, unsafe_allow_html=True)

    # Navigation via radio
    page = st.radio(
        "Navigation",
        ["Home", "Image Detection", "Video Detection", "Audio Detection",
         "Webcam Detection", "Grad-CAM", "Prediction History", "Dashboard", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Decision threshold
    st.markdown('<p class="section-label">Decision Threshold</p>', unsafe_allow_html=True)
    threshold = st.slider("Decision threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
                          label_visibility="collapsed")
    st.markdown(
        f'<p style="color:#888;font-size:12px;margin:0;">'
        f'Verdict fires at <span style="color:#667eea;font-weight:700;">{threshold:.2f}</span> '
        f'&mdash; lower values flag more samples as fake</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<br>', unsafe_allow_html=True)

    # Device and model info
    st.markdown(
        f'<p style="color:#888;font-size:12px;">'
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#28a745;margin-right:6px;"></span>'
        f'Device: <code>{device}</code> &nbsp;|&nbsp; Model: <code>{MODEL_NAME}</code></p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Logout / Change password placeholders
    if st.button("Log out", use_container_width=True):
        st.info("Logout functionality requires authentication setup.")
    if st.button("Change my password", use_container_width=True):
        st.info("Password change requires authentication setup.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

# --- HOME ---
if page == "Home":
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">AI SECURITY</div>
        <h1>AI-Powered Explainable Deepfake Detection</h1>
        <p>
            Detect whether an image, video, or live webcam stream is authentic or manipulated.
            Confidence scores and Grad-CAM explainability provide transparent, auditable
            decision support for media verification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model info cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>Model Architecture</h3>
            <p>Xception</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>Input Resolution</h3>
            <p>299 x 299 px</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="info-card">
            <h3>Inference Device</h3>
            <p>{device.upper()}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br><p class="section-label">Capabilities</p>', unsafe_allow_html=True)

    # Capability cards
    row1 = st.columns(3)
    with row1[0]:
        st.markdown("""
        <div class="cap-card">
            <h4>Image Detection</h4>
            <p>Upload a photo to receive an instant REAL or DEEPFAKE verdict with confidence score and face localization.</p>
        </div>
        """, unsafe_allow_html=True)
    with row1[1]:
        st.markdown("""
        <div class="cap-card">
            <h4>Video Detection</h4>
            <p>Frame-by-frame analysis with majority voting, per-frame probability tracking, and temporal statistics.</p>
        </div>
        """, unsafe_allow_html=True)
    with row1[2]:
        st.markdown("""
        <div class="cap-card">
            <h4>Real-Time Webcam</h4>
            <p>Live detection directly from your camera with bounding boxes, smoothed predictions, and real-time FPS display.</p>
        </div>
        """, unsafe_allow_html=True)

    row2 = st.columns(3)
    with row2[0]:
        st.markdown("""
        <div class="cap-card">
            <h4>Grad-CAM Explainability</h4>
            <p>Visual heatmaps reveal which facial regions influenced the model's decision, enabling transparent audit trails.</p>
        </div>
        """, unsafe_allow_html=True)
    with row2[1]:
        st.markdown("""
        <div class="cap-card">
            <h4>Prediction History</h4>
            <p>Every analysis is logged to SQLite with timestamps, confidence scores, and source metadata for review.</p>
        </div>
        """, unsafe_allow_html=True)
    with row2[2]:
        st.markdown("""
        <div class="cap-card">
            <h4>Analytics Dashboard</h4>
            <p>Aggregate statistics including real vs. fake distribution, average confidence, and inference performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)


# --- IMAGE DETECTION ---
elif page == "Image Detection":
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
    
    /* Video Detection - Forensics UI */
    .video-upload-card {
        background: #141c26; border: 1px dashed #2a3a4e; border-radius: 14px;
        padding: 36px 28px; text-align: center; transition: all 0.25s ease;
    }
    .video-upload-card:hover { border-color: #667eea; background: #182230; }
    .video-upload-card .icon-circle {
        width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 18px;
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.15);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .video-upload-card h3 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
    .video-upload-card .sub { color: #5a6b7e; font-size: 13px; margin: 0 0 14px 0; }
    .video-upload-card .formats { color: #4a5a6e; font-size: 11px; margin: 8px 0 0 0; }

    .pipeline-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 16px 0; margin: 20px 0; flex-wrap: wrap;
    }
    .pipeline-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 12px;
    }
    .pipeline-step .step-icon {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 14px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .pipeline-step .step-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .pipeline-step .step-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .pipeline-step .step-icon.skip { border-color: #3a4a5e; background: #1a2636; color: #3a4a5e; opacity: 0.5; }
    .pipeline-step .step-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #4a5a6e; font-weight: 600; }
    .pipeline-step .step-label.active { color: #667eea; }
    .pipeline-step .step-label.done { color: #28a745; }
    .pipeline-step .step-label.skip { color: #3a4a5e; opacity: 0.5; }
    .pipeline-arrow { color: #2a3a4e; font-size: 14px; margin: 0 2px; padding-bottom: 18px; }

    .config-card {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px; padding: 24px;
    }
    .config-card h4 {
        color: #7a8ba0; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px;
        margin: 0 0 16px 0; font-weight: 600; padding-bottom: 10px; border-bottom: 1px solid #1e2e3e;
    }

    .fusion-status {
        display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px;
        border-radius: 8px; font-size: 12px; font-weight: 600;
    }
    .fusion-status.enabled { background: rgba(102,126,234,0.1); color: #667eea; border: 1px solid rgba(102,126,234,0.2); }
    .fusion-status.disabled { background: rgba(90,107,126,0.1); color: #5a6b7e; border: 1px solid rgba(90,107,126,0.2); }

    .video-empty-state {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 48px 32px; text-align: center; margin-top: 8px;
    }
    .video-empty-state .icon { font-size: 40px; margin-bottom: 16px; opacity: 0.4; }
    .video-empty-state h3 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
    .video-empty-state p { color: #5a6b7e; margin: 0; font-size: 13px; line-height: 1.6; max-width: 400px; margin: 0 auto; }
    .video-empty-state .status-dot {
        display: inline-flex; align-items: center; gap: 6px; margin-top: 14px;
        color: #28a745; font-size: 12px; font-weight: 500;
    }
    .video-empty-state .status-dot::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%; background: #28a745;
    }

    .frame-stat-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px; padding: 14px 16px; text-align: center;
    }
    .frame-stat-card .val { color: #e0e0e0; font-size: 20px; font-weight: 700; margin: 0; }
    .frame-stat-card .lbl { color: #5a6b7e; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 4px 0 0 0; font-weight: 600; }


    /* Audio Detection - Forensics UI */
    .audio-upload-card {
        background: #141c26; border: 1px dashed #2a3a4e; border-radius: 14px;
        padding: 36px 28px; text-align: center; transition: all 0.25s ease;
    }
    .audio-upload-card:hover { border-color: #667eea; background: #182230; }
    .audio-upload-card .icon-circle {
        width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 18px;
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.15);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .audio-upload-card h3 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
    .audio-upload-card .sub { color: #5a6b7e; font-size: 13px; margin: 0 0 14px 0; }
    .audio-upload-card .formats { color: #4a5a6e; font-size: 11px; margin: 8px 0 0 0; }

    .audio-pipeline-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 16px 0; margin: 20px 0; flex-wrap: wrap;
    }
    .audio-pipeline-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 12px;
    }
    .audio-pipeline-step .step-icon {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 14px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .audio-pipeline-step .step-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .audio-pipeline-step .step-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .audio-pipeline-step .step-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #4a5a6e; font-weight: 600; }
    .audio-pipeline-step .step-label.active { color: #667eea; }
    .audio-pipeline-step .step-label.done { color: #28a745; }
    .audio-pipeline-arrow { color: #2a3a4e; font-size: 14px; margin: 0 2px; padding-bottom: 18px; }

    .audio-empty-state {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 48px 32px; text-align: center; margin-top: 8px;
    }
    .audio-empty-state .icon { font-size: 40px; margin-bottom: 16px; opacity: 0.4; }
    .audio-empty-state h3 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
    .audio-empty-state p { color: #5a6b7e; margin: 0; font-size: 13px; line-height: 1.6; max-width: 400px; margin: 0 auto; }
    .audio-empty-state .status-dot {
        display: inline-flex; align-items: center; gap: 6px; margin-top: 14px;
        color: #28a745; font-size: 12px; font-weight: 500;
    }
    .audio-empty-state .status-dot::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%; background: #28a745;
    }

    .audio-stat-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px; padding: 14px 16px; text-align: center;
    }
    .audio-stat-card .val { color: #e0e0e0; font-size: 18px; font-weight: 700; margin: 0; }
    .audio-stat-card .lbl { color: #5a6b7e; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 4px 0 0 0; font-weight: 600; }

    .audio-interpretation {
        padding: 16px; background: #0f1620; border-radius: 10px; border: 1px solid #1a2636;
    }
    .audio-interpretation p { color: #7a8ba0; font-size: 13px; line-height: 1.7; margin: 0; }


    /* Grad-CAM Explainability - Forensics UI */
    .gradcam-upload-card {
        background: #141c26; border: 1px dashed #2a3a4e; border-radius: 14px;
        padding: 36px 28px; text-align: center; transition: all 0.25s ease;
    }
    .gradcam-upload-card:hover { border-color: #667eea; background: #182230; }
    .gradcam-upload-card .icon-circle {
        width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 18px;
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.15);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .gradcam-upload-card h3 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
    .gradcam-upload-card .sub { color: #5a6b7e; font-size: 13px; margin: 0 0 14px 0; }
    .gradcam-upload-card .formats { color: #4a5a6e; font-size: 11px; margin: 8px 0 0 0; }

    .gradcam-pipeline-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 16px 0; margin: 20px 0; flex-wrap: wrap;
    }
    .gradcam-pipeline-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 12px;
    }
    .gradcam-pipeline-step .step-icon {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 14px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .gradcam-pipeline-step .step-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .gradcam-pipeline-step .step-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .gradcam-pipeline-step .step-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #4a5a6e; font-weight: 600; }
    .gradcam-pipeline-step .step-label.active { color: #667eea; }
    .gradcam-pipeline-step .step-label.done { color: #28a745; }
    .gradcam-pipeline-arrow { color: #2a3a4e; font-size: 14px; margin: 0 2px; padding-bottom: 18px; }

    .gradcam-empty-state {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 48px 32px; text-align: center; margin-top: 8px;
    }
    .gradcam-empty-state .icon { font-size: 40px; margin-bottom: 16px; opacity: 0.4; }
    .gradcam-empty-state h3 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
    .gradcam-empty-state p { color: #5a6b7e; margin: 0; font-size: 13px; line-height: 1.6; max-width: 400px; margin: 0 auto; }
    .gradcam-empty-state .status-dot {
        display: inline-flex; align-items: center; gap: 6px; margin-top: 14px;
        color: #28a745; font-size: 12px; font-weight: 500;
    }
    .gradcam-empty-state .status-dot::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%; background: #28a745;
    }

    .gradcam-viz-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .gradcam-viz-card .label {
        color: #7a8ba0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
        margin: 0 0 8px 0; font-weight: 600;
    }
    .gradcam-viz-card .desc {
        color: #5a6b7e; font-size: 11px; margin: 8px 0 0 0; line-height: 1.4;
    }

    .legend-item {
        display: flex; align-items: center; gap: 8px; padding: 6px 0;
    }
    .legend-swatch {
        width: 32px; height: 12px; border-radius: 3px;
    }
    .legend-label {
        color: #7a8ba0; font-size: 12px;
    }

    .gradcam-interpretation {
        padding: 16px; background: #0f1620; border-radius: 10px; border: 1px solid #1a2636;
    }
    .gradcam-interpretation p { color: #7a8ba0; font-size: 13px; line-height: 1.7; margin: 0; }


    /* About Page - Forensics UI */
    .about-hero {
        background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0f1923 100%);
        border: 1px solid #1e2e3e; border-radius: 16px; padding: 36px;
        margin-bottom: 24px; text-align: center;
    }
    .about-hero h1 { color: #e8edf3; margin: 0 0 8px 0; font-size: 26px; font-weight: 700; }
    .about-hero p { color: #7a8ba0; margin: 0; font-size: 14px; line-height: 1.5; }

    .about-intro-card {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 28px; margin-bottom: 20px;
    }
    .about-intro-card h3 { color: #e0e0e0; margin: 0 0 12px 0; font-size: 18px; font-weight: 600; }
    .about-intro-card p { color: #7a8ba0; font-size: 14px; line-height: 1.7; margin: 0; }

    .capability-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 20px; text-align: left; transition: border-color 0.2s ease;
        min-height: 140px;
    }
    .capability-card:hover { border-color: #2a3a4e; }
    .capability-card .cap-icon {
        width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
        background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.2); margin-bottom: 12px;
    }
    .capability-card h4 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 14px; font-weight: 600; }
    .capability-card p { color: #5a6b7e; font-size: 12px; line-height: 1.5; margin: 0; }

    .tech-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; text-align: center;
    }
    .tech-card .tech-name { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }
    .tech-card .tech-desc { color: #5a6b7e; font-size: 11px; margin: 4px 0 0 0; }

    .principle-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 24px; text-align: center;
    }
    .principle-card .p-icon {
        width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
        margin: 0 auto 14px; font-size: 20px;
    }
    .principle-card h4 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 15px; font-weight: 600; }
    .principle-card p { color: #5a6b7e; font-size: 12px; line-height: 1.6; margin: 0; }

    .workflow-step {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .workflow-step .step-num {
        color: #667eea; font-size: 11px; font-weight: 700; margin: 0 0 6px 0;
        letter-spacing: 1px;
    }
    .workflow-step .step-title { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }

    .highlight-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; display: flex; align-items: center; gap: 10px;
    }
    .highlight-card .h-dot {
        width: 6px; height: 6px; border-radius: 50%; background: #667eea; flex-shrink: 0;
    }
    .highlight-card .h-text { color: #c0c8d4; font-size: 13px; }

    .about-section-title {
        color: #e0e0e0; font-size: 18px; font-weight: 700; margin: 0 0 8px 0;
    }
    .about-section-sub {
        color: #5a6b7e; font-size: 12px; margin: 0 0 20px 0;
    }

    .config-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 0; border-bottom: 1px solid #1a2636; font-size: 13px;
    }
    .config-row:last-child { border-bottom: none; }
    .config-row .ck { color: #5a6b7e; }
    .config-row .cv { color: #c0c8d4; font-weight: 500; }

</style>
    """, unsafe_allow_html=True)

    # ---- Page Header ----
    st.markdown("""
    <div class="forensics-header">
        <div class="badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            IMAGE FORENSICS
        </div>
        <h1>Image Deepfake Detection</h1>
        <p>Analyze an image for AI-generated or manipulated visual content.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Upload Card ----
    st.markdown("""
    <div class="upload-forensics-card" id="upload-zone">
        <div class="icon-circle">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="1.5">
                <rect x="3" y="3" width="18" height="18" rx="3"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <path d="M21 15l-5-5L5 21"/>
            </svg>
        </div>
        <h3>Upload Image</h3>
        <p class="sub">Upload a JPG, JPEG, or PNG image for forensic analysis.</p>
        <div class="divider">or drag and drop your image here</div>
        <p class="hint">Maximum file size: 200 MB</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img",
                                   label_visibility="collapsed")

    # ---- Workflow Indicator ----
    step_upload = "done" if uploaded_img else "active"
    step_analyze = "active" if uploaded_img else ""
    step_result = ""
    st.markdown(f"""
    <div class="workflow-bar">
        <div class="workflow-step">
            <div class="step-icon {step_upload}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            </div>
            <span class="step-label {step_upload}">Upload</span>
        </div>
        <div class="workflow-arrow">&#8594;</div>
        <div class="workflow-step">
            <div class="step-icon {step_analyze}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            </div>
            <span class="step-label {step_analyze}">Analyze</span>
        </div>
        <div class="workflow-arrow">&#8594;</div>
        <div class="workflow-step">
            <div class="step-icon">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            </div>
            <span class="step-label">Result</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Empty State ----
    if uploaded_img is None:
        st.markdown("""
        <div class="empty-state-forensics">
            <div class="icon">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3a4a5e" stroke-width="1.2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
            </div>
            <h3>Ready for Analysis</h3>
            <p>Upload an image to begin deepfake detection and forensic analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    # ---- Analysis After Upload ----
    if uploaded_img is not None:
        t0 = time.time()
        try:
            pil_img = Image.open(uploaded_img).convert("RGB")
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            img_w, img_h = pil_img.size
            file_size = uploaded_img.size

            face = extract_face(bgr_img, face_cascade)

            if face is None:
                st.warning("No face detected -- try a clearer, front-facing photo.")
            else:
                prob = predict(face, model, device, transform)
                elapsed = time.time() - t0
                label = "FAKE" if prob >= threshold else "REAL"
                real_prob = 1.0 - prob
                confidence = abs(prob - 0.5) * 2

                # ---- Two-Column Layout ----
                col_img, col_result = st.columns([1, 1], gap="large")

                with col_img:
                    st.markdown("""
                    <div class="analysis-card">
                        <h4>Input Image</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(pil_img, use_container_width=True)

                    size_kb = file_size / 1024
                    size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{file_size/1024/1024:.1f} MB"
                    st.markdown(f"""
                    <div style="padding: 4px 0;">
                        <span class="meta-chip">{uploaded_img.name}</span>
                        <span class="meta-chip">{uploaded_img.type}</span>
                        <span class="meta-chip">{img_w}x{img_h}</span>
                        <span class="meta-chip">{size_str}</span>
                    </div>
                    """, unsafe_allow_html=True)

                with col_result:
                    verdict_class = "fake" if label == "FAKE" else "real"
                    verdict_icon = "&#10006;" if label == "FAKE" else "&#10004;"
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4>Detection Result</h4>
                        <p style="color:#5a6b7e;font-size:12px;margin:0 0 16px 0;">Authenticity Assessment</p>
                        <div class="verdict-badge-lg {verdict_class}">
                            {verdict_icon}&nbsp; {label}
                        </div>
                        <div style="margin-top:20px;">
                            <div class="prob-row">
                                <span class="label">Fake</span>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill fake" style="width:{prob * 100}%;"></div>
                                </div>
                                <span class="value">{prob:.1%}</span>
                            </div>
                            <div class="prob-row">
                                <span class="label">Real</span>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill real" style="width:{real_prob * 100}%;"></div>
                                </div>
                                <span class="value">{real_prob:.1%}</span>
                            </div>
                        </div>
                        <div style="margin-top:16px;padding-top:14px;border-top:1px solid #1e2e3e;">
                            <div class="forensic-detail-row"><span class="dk">Confidence</span><span class="dv">{confidence:.1%}</span></div>
                            <div class="forensic-detail-row"><span class="dk">Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ---- AI Forensic Analysis Card ----
                if label == "FAKE":
                    assessment_html = '<strong style="color:#dc3545;">Potentially manipulated</strong> content was detected.'
                else:
                    assessment_html = '<strong style="color:#28a745;">Likely authentic</strong> content was detected.'

                st.markdown(f"""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>AI Forensic Analysis</h4>
                    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">
                        <span class="meta-chip">Model: {MODEL_NAME.upper()}</span>
                        <span class="meta-chip">Device: {device.upper()}</span>
                        <span class="meta-chip">Threshold: {threshold:.2f}</span>
                    </div>
                    <div style="padding:16px;background:#0f1620;border-radius:10px;border:1px solid #1a2636;">
                        <p class="summary-text">
                            The uploaded image was analyzed using the configured deepfake detection model.
                            {assessment_html}
                            The model confidence for this assessment is {confidence:.1%}.
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ---- Forensic Details Card ----
                st.markdown(f"""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Forensic Details</h4>
                    <div class="forensic-detail-row"><span class="dk">Model</span><span class="dv">{MODEL_NAME.upper()}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Input Format</span><span class="dv">JPEG / RGB 299x299</span></div>
                    <div class="forensic-detail-row"><span class="dk">Resolution</span><span class="dv">{img_w}x{img_h} px</span></div>
                    <div class="forensic-detail-row"><span class="dk">Processing Device</span><span class="dv">{device.upper()}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Confidence</span><span class="dv">{confidence:.1%}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Inference Time</span><span class="dv">{elapsed:.3f}s</span></div>
                    <div class="forensic-detail-row"><span class="dk">Analysis Status</span><span class="dv" style="color:#28a745;">Completed</span></div>
                </div>
                """, unsafe_allow_html=True)

                store_prediction(
                    file_name=uploaded_img.name,
                    media_type="image",
                    prediction=label,
                    confidence=confidence,
                    fake_prob=prob,
                    inference_time=elapsed,
                )

        except Exception as e:
            logger.error("Image processing failed: %s", e)
            st.error(f"Error processing image: {e}")

# --- VIDEO DETECTION ---
elif page == "Video Detection":
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
    
    /* Audio Detection - Forensics UI */
    .audio-upload-card {
        background: #141c26; border: 1px dashed #2a3a4e; border-radius: 14px;
        padding: 36px 28px; text-align: center; transition: all 0.25s ease;
    }
    .audio-upload-card:hover { border-color: #667eea; background: #182230; }
    .audio-upload-card .icon-circle {
        width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 18px;
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.15);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .audio-upload-card h3 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
    .audio-upload-card .sub { color: #5a6b7e; font-size: 13px; margin: 0 0 14px 0; }
    .audio-upload-card .formats { color: #4a5a6e; font-size: 11px; margin: 8px 0 0 0; }

    .audio-pipeline-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 16px 0; margin: 20px 0; flex-wrap: wrap;
    }
    .audio-pipeline-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 12px;
    }
    .audio-pipeline-step .step-icon {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 14px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .audio-pipeline-step .step-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .audio-pipeline-step .step-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .audio-pipeline-step .step-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #4a5a6e; font-weight: 600; }
    .audio-pipeline-step .step-label.active { color: #667eea; }
    .audio-pipeline-step .step-label.done { color: #28a745; }
    .audio-pipeline-arrow { color: #2a3a4e; font-size: 14px; margin: 0 2px; padding-bottom: 18px; }

    .audio-empty-state {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 48px 32px; text-align: center; margin-top: 8px;
    }
    .audio-empty-state .icon { font-size: 40px; margin-bottom: 16px; opacity: 0.4; }
    .audio-empty-state h3 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
    .audio-empty-state p { color: #5a6b7e; margin: 0; font-size: 13px; line-height: 1.6; max-width: 400px; margin: 0 auto; }
    .audio-empty-state .status-dot {
        display: inline-flex; align-items: center; gap: 6px; margin-top: 14px;
        color: #28a745; font-size: 12px; font-weight: 500;
    }
    .audio-empty-state .status-dot::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%; background: #28a745;
    }

    .audio-stat-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px; padding: 14px 16px; text-align: center;
    }
    .audio-stat-card .val { color: #e0e0e0; font-size: 18px; font-weight: 700; margin: 0; }
    .audio-stat-card .lbl { color: #5a6b7e; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 4px 0 0 0; font-weight: 600; }

    .audio-interpretation {
        padding: 16px; background: #0f1620; border-radius: 10px; border: 1px solid #1a2636;
    }
    .audio-interpretation p { color: #7a8ba0; font-size: 13px; line-height: 1.7; margin: 0; }


    /* Grad-CAM Explainability - Forensics UI */
    .gradcam-upload-card {
        background: #141c26; border: 1px dashed #2a3a4e; border-radius: 14px;
        padding: 36px 28px; text-align: center; transition: all 0.25s ease;
    }
    .gradcam-upload-card:hover { border-color: #667eea; background: #182230; }
    .gradcam-upload-card .icon-circle {
        width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 18px;
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.15);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .gradcam-upload-card h3 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
    .gradcam-upload-card .sub { color: #5a6b7e; font-size: 13px; margin: 0 0 14px 0; }
    .gradcam-upload-card .formats { color: #4a5a6e; font-size: 11px; margin: 8px 0 0 0; }

    .gradcam-pipeline-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 16px 0; margin: 20px 0; flex-wrap: wrap;
    }
    .gradcam-pipeline-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 12px;
    }
    .gradcam-pipeline-step .step-icon {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 14px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .gradcam-pipeline-step .step-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .gradcam-pipeline-step .step-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .gradcam-pipeline-step .step-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #4a5a6e; font-weight: 600; }
    .gradcam-pipeline-step .step-label.active { color: #667eea; }
    .gradcam-pipeline-step .step-label.done { color: #28a745; }
    .gradcam-pipeline-arrow { color: #2a3a4e; font-size: 14px; margin: 0 2px; padding-bottom: 18px; }

    .gradcam-empty-state {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 48px 32px; text-align: center; margin-top: 8px;
    }
    .gradcam-empty-state .icon { font-size: 40px; margin-bottom: 16px; opacity: 0.4; }
    .gradcam-empty-state h3 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
    .gradcam-empty-state p { color: #5a6b7e; margin: 0; font-size: 13px; line-height: 1.6; max-width: 400px; margin: 0 auto; }
    .gradcam-empty-state .status-dot {
        display: inline-flex; align-items: center; gap: 6px; margin-top: 14px;
        color: #28a745; font-size: 12px; font-weight: 500;
    }
    .gradcam-empty-state .status-dot::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%; background: #28a745;
    }

    .gradcam-viz-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .gradcam-viz-card .label {
        color: #7a8ba0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
        margin: 0 0 8px 0; font-weight: 600;
    }
    .gradcam-viz-card .desc {
        color: #5a6b7e; font-size: 11px; margin: 8px 0 0 0; line-height: 1.4;
    }

    .legend-item {
        display: flex; align-items: center; gap: 8px; padding: 6px 0;
    }
    .legend-swatch {
        width: 32px; height: 12px; border-radius: 3px;
    }
    .legend-label {
        color: #7a8ba0; font-size: 12px;
    }

    .gradcam-interpretation {
        padding: 16px; background: #0f1620; border-radius: 10px; border: 1px solid #1a2636;
    }
    .gradcam-interpretation p { color: #7a8ba0; font-size: 13px; line-height: 1.7; margin: 0; }


    /* About Page - Forensics UI */
    .about-hero {
        background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0f1923 100%);
        border: 1px solid #1e2e3e; border-radius: 16px; padding: 36px;
        margin-bottom: 24px; text-align: center;
    }
    .about-hero h1 { color: #e8edf3; margin: 0 0 8px 0; font-size: 26px; font-weight: 700; }
    .about-hero p { color: #7a8ba0; margin: 0; font-size: 14px; line-height: 1.5; }

    .about-intro-card {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 28px; margin-bottom: 20px;
    }
    .about-intro-card h3 { color: #e0e0e0; margin: 0 0 12px 0; font-size: 18px; font-weight: 600; }
    .about-intro-card p { color: #7a8ba0; font-size: 14px; line-height: 1.7; margin: 0; }

    .capability-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 20px; text-align: left; transition: border-color 0.2s ease;
        min-height: 140px;
    }
    .capability-card:hover { border-color: #2a3a4e; }
    .capability-card .cap-icon {
        width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
        background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.2); margin-bottom: 12px;
    }
    .capability-card h4 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 14px; font-weight: 600; }
    .capability-card p { color: #5a6b7e; font-size: 12px; line-height: 1.5; margin: 0; }

    .tech-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; text-align: center;
    }
    .tech-card .tech-name { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }
    .tech-card .tech-desc { color: #5a6b7e; font-size: 11px; margin: 4px 0 0 0; }

    .principle-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 24px; text-align: center;
    }
    .principle-card .p-icon {
        width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
        margin: 0 auto 14px; font-size: 20px;
    }
    .principle-card h4 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 15px; font-weight: 600; }
    .principle-card p { color: #5a6b7e; font-size: 12px; line-height: 1.6; margin: 0; }

    .workflow-step {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .workflow-step .step-num {
        color: #667eea; font-size: 11px; font-weight: 700; margin: 0 0 6px 0;
        letter-spacing: 1px;
    }
    .workflow-step .step-title { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }

    .highlight-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; display: flex; align-items: center; gap: 10px;
    }
    .highlight-card .h-dot {
        width: 6px; height: 6px; border-radius: 50%; background: #667eea; flex-shrink: 0;
    }
    .highlight-card .h-text { color: #c0c8d4; font-size: 13px; }

    .about-section-title {
        color: #e0e0e0; font-size: 18px; font-weight: 700; margin: 0 0 8px 0;
    }
    .about-section-sub {
        color: #5a6b7e; font-size: 12px; margin: 0 0 20px 0;
    }

    .config-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 0; border-bottom: 1px solid #1a2636; font-size: 13px;
    }
    .config-row:last-child { border-bottom: none; }
    .config-row .ck { color: #5a6b7e; }
    .config-row .cv { color: #c0c8d4; font-weight: 500; }

</style>
    """, unsafe_allow_html=True)

    # ---- Page Header ----
    st.markdown("""
    <div class="forensics-header">
        <div class="badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
            VIDEO FORENSICS
        </div>
        <h1>Video Deepfake Detection</h1>
        <p>Analyze video frames and audio signals for potential synthetic or manipulated media.</p>
        <p style="color:#5a6b7e;font-size:12px;margin:8px 0 0 0;">AI Forensic Engine &bull; Ready</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Upload Card ----
    st.markdown("""
    <div class="video-upload-card">
        <div class="icon-circle">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="1.5">
                <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
            </svg>
        </div>
        <h3>Upload Video Evidence</h3>
        <p class="sub">Upload a video for frame-level and optional audio-visual forensic analysis.</p>
        <p class="formats">MP4 &bull; MOV &bull; AVI &bull; MKV &bull; MPEG4</p>
        <p class="formats" style="margin-top:4px;">Maximum file size: 200 MB</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], key="vid",
                                   label_visibility="collapsed")

    # ---- Analysis Configuration ----
    audio_ready = os.path.exists(AUDIO_CHECKPOINT_PATH)

    col_slider, col_fusion = st.columns([1, 1], gap="large")
    with col_slider:
        st.markdown("""
        <div class="config-card">
            <h4>Analysis Configuration</h4>
        </div>
        """, unsafe_allow_html=True)
        max_frames = st.slider("Frames to Sample", 4, 32, 16)
        st.caption(f"Currently sampling {max_frames} frames. Controls how many video frames are analyzed during detection.")

    with col_fusion:
        st.markdown("""
        <div class="config-card">
            <h4>Multimodal Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        use_fusion = st.checkbox(
            "Enable audio + video fusion", value=audio_ready,
            disabled=not audio_ready,
        )
        if use_fusion:
            st.markdown('<span class="fusion-status enabled"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg> Audio + Video Fusion: Enabled</span>', unsafe_allow_html=True)
        else:
            if not audio_ready:
                st.caption("No audio checkpoint found. Face-only analysis.")
            st.markdown('<span class="fusion-status disabled"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg> Visual Analysis Only</span>', unsafe_allow_html=True)
        st.caption("Combine visual frame analysis with available audio evidence for a multimodal assessment.")

    # ---- Analysis Pipeline ----
    pipeline_audio = "active" if use_fusion else "skip"
    st.markdown(f"""
    <div class="pipeline-bar">
        <div class="pipeline-step">
            <div class="step-icon {"done" if uploaded_vid else "active"}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
            </div>
            <span class="step-label {"done" if uploaded_vid else "active"}">Video Input</span>
        </div>
        <div class="pipeline-arrow">&#8595;</div>
        <div class="pipeline-step">
            <div class="step-icon active">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
            </div>
            <span class="step-label active">Frame Analysis</span>
        </div>
        <div class="pipeline-arrow">&#8595;</div>
        <div class="pipeline-step">
            <div class="step-icon {pipeline_audio}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
            </div>
            <span class="step-label {pipeline_audio}">Audio Analysis</span>
        </div>
        <div class="pipeline-arrow">&#8595;</div>
        <div class="pipeline-step">
            <div class="step-icon {pipeline_audio}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
            </div>
            <span class="step-label {pipeline_audio}">Multimodal Fusion</span>
        </div>
        <div class="pipeline-arrow">&#8595;</div>
        <div class="pipeline-step">
            <div class="step-icon">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            </div>
            <span class="step-label">Forensic Result</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Empty State ----
    if uploaded_vid is None:
        st.markdown("""
        <div class="video-empty-state">
            <div class="icon">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3a4a5e" stroke-width="1.2">
                    <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                </svg>
            </div>
            <h3>Ready for Video Analysis</h3>
            <p>Upload video evidence to begin frame-level deepfake detection.</p>
            <div class="status-dot">Detection engine ready</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- After Video Upload ----
    if uploaded_vid is not None:
        video_path = None
        t0 = time.time()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1]) as tmp:
                tmp.write(uploaded_vid.read())
                video_path = tmp.name

            # Extract video metadata
            vid_cap = cv2.VideoCapture(video_path)
            vid_fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap.isOpened() else 0
            vid_frames_total = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if vid_cap.isOpened() else 0
            vid_w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if vid_cap.isOpened() else 0
            vid_h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if vid_cap.isOpened() else 0
            vid_duration = vid_frames_total / vid_fps if vid_fps > 0 else 0
            vid_cap.release()

            file_size = uploaded_vid.size
            size_mb = file_size / (1024 * 1024)

            # ---- Two-Column Layout ----
            col_vid, col_result = st.columns([1, 1], gap="large")

            with col_vid:
                st.markdown("""
                <div class="analysis-card">
                    <h4>Video Evidence</h4>
                </div>
                """, unsafe_allow_html=True)
                st.video(uploaded_vid)

                st.markdown(f"""
                <div style="padding: 4px 0;">
                    <span class="meta-chip">{uploaded_vid.name}</span>
                    <span class="meta-chip">{uploaded_vid.type}</span>
                    <span class="meta-chip">{size_mb:.1f} MB</span>
                </div>
                <div style="padding: 2px 0;">
                    <span class="meta-chip">{vid_w}x{vid_h}</span>
                    <span class="meta-chip">{vid_fps:.1f} FPS</span>
                    <span class="meta-chip">{vid_duration:.1f}s</span>
                    <span class="meta-chip">{vid_frames_total} frames</span>
                </div>
                """, unsafe_allow_html=True)

            with col_result:
                st.markdown("""
                <div class="analysis-card">
                    <h4>Forensic Assessment</h4>
                </div>
                """, unsafe_allow_html=True)

            # ---- Run Analysis ----
            with st.spinner("Sampling frames and running inference..."):
                frames = sample_video_frames(video_path, max_frames=max_frames)
                probs = []
                for frame in frames:
                    face = extract_face(frame, face_cascade)
                    if face is not None:
                        probs.append(predict(face, model, device, transform))
            video_prob = float(np.mean(probs)) if probs else None

            audio_prob = None
            if use_fusion and audio_ready:
                with st.spinner("Extracting and analyzing the audio track..."):
                    wav = load_audio_from_video(video_path, SAMPLE_RATE)
                    if wav is not None:
                        audio_model, audio_device = load_audio_model()
                        audio_prob = predict_from_array(wav, audio_model, audio_device)

            fused = fuse_scores(video_prob, audio_prob, w_video=0.5)
            elapsed = time.time() - t0

            if fused is None:
                st.warning("No face detected in any sampled frame, and no usable audio track.")
            else:
                label = "FAKE" if fused >= threshold else "REAL"
                real_prob = 1.0 - fused
                confidence = abs(fused - 0.5) * 2
                verdict_class = "fake" if label == "FAKE" else "real"
                verdict_icon = "&#10006;" if label == "FAKE" else "&#10004;"

                # ---- Right Column: Forensic Assessment ----
                with col_result:
                    st.markdown(f"""
                    <div style="margin-top:48px;">
                        <p style="color:#5a6b7e;font-size:12px;margin:0 0 16px 0;">Authenticity Assessment</p>
                        <div class="verdict-badge-lg {verdict_class}">
                            {verdict_icon}&nbsp; {label}
                        </div>
                        <div style="margin-top:20px;">
                            <div class="prob-row">
                                <span class="label">Fake</span>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill fake" style="width:{fused * 100}%;"></div>
                                </div>
                                <span class="value">{fused:.1%}</span>
                            </div>
                            <div class="prob-row">
                                <span class="label">Real</span>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill real" style="width:{real_prob * 100}%;"></div>
                                </div>
                                <span class="value">{real_prob:.1%}</span>
                            </div>
                        </div>
                        <div style="margin-top:16px;padding-top:14px;border-top:1px solid #1e2e3e;">
                            <div class="forensic-detail-row"><span class="dk">Confidence</span><span class="dv">{confidence:.1%}</span></div>
                            <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ---- Frame-Level Analysis ----
                fake_count = sum(1 for p in probs if p >= threshold)
                real_count = len(probs) - fake_count
                avg_prob = float(np.mean(probs)) if probs else 0
                max_prob = float(np.max(probs)) if probs else 0
                min_prob = float(np.min(probs)) if probs else 0

                st.markdown(f"""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Frame-Level Analysis</h4>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;">
                        <div class="frame-stat-card"><p class="val">{max_frames}</p><p class="lbl">Frames Sampled</p></div>
                        <div class="frame-stat-card"><p class="val">{len(probs)}</p><p class="lbl">Frames Analyzed</p></div>
                        <div class="frame-stat-card"><p class="val" style="color:#dc3545;">{fake_count}</p><p class="lbl">Fake Frames</p></div>
                        <div class="frame-stat-card"><p class="val" style="color:#28a745;">{real_count}</p><p class="lbl">Real Frames</p></div>
                        <div class="frame-stat-card"><p class="val">{avg_prob:.3f}</p><p class="lbl">Avg Fake Prob</p></div>
                        <div class="frame-stat-card"><p class="val">{max_prob:.3f}</p><p class="lbl">Max Fake Prob</p></div>
                        <div class="frame-stat-card"><p class="val">{min_prob:.3f}</p><p class="lbl">Min Fake Prob</p></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ---- Temporal Analysis ----
                st.markdown("""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Temporal Authenticity Analysis</h4>
                </div>
                """, unsafe_allow_html=True)
                if probs:
                    import pandas as pd
                    chart_data = pd.DataFrame({
                        "Frame": list(range(1, len(probs) + 1)),
                        "Fake Probability": probs,
                    })
                    st.line_chart(chart_data.set_index("Frame"))
                    st.caption(f"Decision threshold: {threshold:.2f}")
                else:
                    st.info("Frame-level probability visualization will appear after analysis.")

                # ---- Multimodal Evidence ----
                st.markdown(f"""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Multimodal Evidence</h4>
                    <div class="forensic-detail-row"><span class="dk">Visual Evidence</span><span class="dv">{video_prob:.3f if video_prob else 'n/a'}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Audio Evidence</span><span class="dv">{audio_prob:.3f if audio_prob else 'Unavailable'}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Fused Assessment</span><span class="dv">{fused:.3f}</span></div>
                    {"<p style='color:#5a6b7e;font-size:12px;margin:10px 0 0 0;'>Audio evidence unavailable &mdash; assessment based on visual evidence.</p>" if audio_prob is None and not use_fusion else ""}
                </div>
                """, unsafe_allow_html=True)

                # ---- Assessment Interpretation ----
                if label == "FAKE":
                    interp = "Potentially Manipulated"
                    interp_desc = "The model detected patterns associated with manipulated or synthetic media."
                elif confidence < 0.15:
                    interp = "Requires Review"
                    interp_desc = "The model confidence is close to the decision threshold. Additional forensic review is recommended."
                else:
                    interp = "Likely Authentic"
                    interp_desc = "The model did not detect strong evidence of manipulation under the current analysis settings."

                st.markdown(f"""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Assessment Interpretation</h4>
                    <div style="padding:16px;background:#0f1620;border-radius:10px;border:1px solid #1a2636;">
                        <p style="color:#e0e0e0;font-size:14px;font-weight:600;margin:0 0 8px 0;">{interp}</p>
                        <p class="summary-text">{interp_desc}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ---- Forensic Summary ----
                st.markdown(f"""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Forensic Analysis Summary</h4>
                    <div class="forensic-detail-row"><span class="dk">Analysis Status</span><span class="dv" style="color:#28a745;">Completed</span></div>
                    <div class="forensic-detail-row"><span class="dk">Evidence Type</span><span class="dv">Video</span></div>
                    <div class="forensic-detail-row"><span class="dk">Analysis Method</span><span class="dv">Frame-level AI analysis</span></div>
                    <div class="forensic-detail-row"><span class="dk">Multimodal Analysis</span><span class="dv">{"Enabled" if use_fusion else "Disabled"}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Model</span><span class="dv">{MODEL_NAME.upper()}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Device</span><span class="dv">{device.upper()}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Threshold</span><span class="dv">{threshold:.2f}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Inference Time</span><span class="dv">{elapsed:.2f}s</span></div>
                </div>
                """, unsafe_allow_html=True)

                store_prediction(
                    file_name=uploaded_vid.name,
                    media_type="video",
                    prediction=label,
                    confidence=confidence,
                    fake_prob=fused,
                    inference_time=elapsed,
                )

        except Exception as e:
            logger.error("Video processing failed: %s", e)
            st.error(f"Error processing video: {e}")
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)

# --- WEBCAM DETECTION ---
elif page == "Webcam Detection":
    st.markdown("## Webcam Detection")
    st.markdown('<p class="section-label">Real-time deepfake detection from your camera</p>', unsafe_allow_html=True)

    try:
        from collections import deque
        import av
        from streamlit_webrtc import webrtc_streamer, WebRtcMode

        st.caption("Allow camera access when your browser asks, then click **START**.")
        prob_history = deque(maxlen=8)

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            box = detect_largest_face(img, face_cascade)
            prob = None
            if box is not None:
                x0, y0, x1, y1 = box
                face = img[y0:y1, x0:x1]
                if face.size > 0:
                    prob = predict_fake_prob(face, model, device, transform)
                    prob_history.append(prob)
            smoothed = float(np.mean(prob_history)) if prob_history else 0.0
            draw_overlay(img, box, prob, threshold, smoothed, fps=0.0, show_help=False)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="deepfake-live",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        st.caption("Green box = REAL, red box = FAKE. Prediction is smoothed over recent frames.")

    except ImportError:
        st.info(
            "In-browser live video needs `streamlit-webrtc`:\n\n"
            "```\npip install streamlit-webrtc\n```\n\n"
            "You can still take a **snapshot** below."
        )
        snap = st.camera_input("Take a snapshot to analyze")
        if snap is not None:
            try:
                pil_snap = Image.open(snap).convert("RGB")
                bgr_snap = cv2.cvtColor(np.array(pil_snap), cv2.COLOR_RGB2BGR)
                face = extract_face(bgr_snap, face_cascade)
                if face is None:
                    st.warning("No face detected.")
                else:
                    prob = predict(face, model, device, transform)
                    label = "FAKE" if prob >= threshold else "REAL"
                    col1, col2 = st.columns(2)
                    col1.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
                               caption="Detected face", use_container_width=True)
                    col2.subheader(f"{label}")
                    col2.progress(min(max(prob, 0.0), 1.0))
                    col2.write(f"Fake probability: **{prob:.4f}**")
                    store_prediction("webcam_snapshot", "webcam", label,
                                     abs(prob - 0.5) * 2, prob, 0.0)
            except Exception as e:
                st.error(f"Error: {e}")


# --- GRAD-CAM ---
elif page == "Grad-CAM":
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
    
    /* About Page - Forensics UI */
    .about-hero {
        background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0f1923 100%);
        border: 1px solid #1e2e3e; border-radius: 16px; padding: 36px;
        margin-bottom: 24px; text-align: center;
    }
    .about-hero h1 { color: #e8edf3; margin: 0 0 8px 0; font-size: 26px; font-weight: 700; }
    .about-hero p { color: #7a8ba0; margin: 0; font-size: 14px; line-height: 1.5; }

    .about-intro-card {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 28px; margin-bottom: 20px;
    }
    .about-intro-card h3 { color: #e0e0e0; margin: 0 0 12px 0; font-size: 18px; font-weight: 600; }
    .about-intro-card p { color: #7a8ba0; font-size: 14px; line-height: 1.7; margin: 0; }

    .capability-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 20px; text-align: left; transition: border-color 0.2s ease;
        min-height: 140px;
    }
    .capability-card:hover { border-color: #2a3a4e; }
    .capability-card .cap-icon {
        width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
        background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.2); margin-bottom: 12px;
    }
    .capability-card h4 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 14px; font-weight: 600; }
    .capability-card p { color: #5a6b7e; font-size: 12px; line-height: 1.5; margin: 0; }

    .tech-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; text-align: center;
    }
    .tech-card .tech-name { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }
    .tech-card .tech-desc { color: #5a6b7e; font-size: 11px; margin: 4px 0 0 0; }

    .principle-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 24px; text-align: center;
    }
    .principle-card .p-icon {
        width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
        margin: 0 auto 14px; font-size: 20px;
    }
    .principle-card h4 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 15px; font-weight: 600; }
    .principle-card p { color: #5a6b7e; font-size: 12px; line-height: 1.6; margin: 0; }

    .workflow-step {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .workflow-step .step-num {
        color: #667eea; font-size: 11px; font-weight: 700; margin: 0 0 6px 0;
        letter-spacing: 1px;
    }
    .workflow-step .step-title { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }

    .highlight-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; display: flex; align-items: center; gap: 10px;
    }
    .highlight-card .h-dot {
        width: 6px; height: 6px; border-radius: 50%; background: #667eea; flex-shrink: 0;
    }
    .highlight-card .h-text { color: #c0c8d4; font-size: 13px; }

    .about-section-title {
        color: #e0e0e0; font-size: 18px; font-weight: 700; margin: 0 0 8px 0;
    }
    .about-section-sub {
        color: #5a6b7e; font-size: 12px; margin: 0 0 20px 0;
    }

    .config-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 0; border-bottom: 1px solid #1a2636; font-size: 13px;
    }
    .config-row:last-child { border-bottom: none; }
    .config-row .ck { color: #5a6b7e; }
    .config-row .cv { color: #c0c8d4; font-weight: 500; }

</style>
    """, unsafe_allow_html=True)

    # ---- Page Header ----
    st.markdown("""
    <div class="forensics-header">
        <div class="badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
            EXPLAINABLE AI
        </div>
        <h1>Grad-CAM Explainability</h1>
        <p>Visualize the facial regions that influenced the deepfake detection model's prediction.</p>
        <p style="color:#5a6b7e;font-size:12px;margin:8px 0 0 0;">Explainability Engine &bull; Ready</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Introduction Card ----
    st.markdown("""
    <div class="analysis-card" style="margin-bottom:20px;">
        <div style="display:flex;align-items:flex-start;gap:14px;">
            <div style="min-width:36px;height:36px;border-radius:8px;background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.2);display:flex;align-items:center;justify-content:center;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
            </div>
            <div>
                <h4 style="color:#e0e0e0;margin:0 0 4px 0;font-size:14px;text-transform:none;letter-spacing:0;">Understand the Model Decision</h4>
                <p style="color:#7a8ba0;font-size:13px;margin:0;line-height:1.5;">Grad-CAM highlights image regions that contributed most strongly to the model's prediction, providing a visual explanation of the AI assessment.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Upload Card ----
    st.markdown("""
    <div class="gradcam-upload-card">
        <div class="icon-circle">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="1.5">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <path d="M21 15l-5-5L5 21"/>
            </svg>
        </div>
        <h3>Upload Image for Explanation</h3>
        <p class="sub">Upload an image to generate a visual explanation of the model prediction.</p>
        <p class="formats">JPG &bull; JPEG &bull; PNG</p>
        <p class="formats" style="margin-top:4px;">Maximum file size: 200 MB</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="gradcam",
                                   label_visibility="collapsed")

    # ---- Explainability Workflow ----
    gradcam_uploaded = uploaded_img is not None
    st.markdown(f"""
    <div class="gradcam-pipeline-bar">
        <div class="gradcam-pipeline-step">
            <div class="step-icon {"done" if gradcam_uploaded else "active"}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
            </div>
            <span class="step-label {"done" if gradcam_uploaded else "active"}">Image Input</span>
        </div>
        <div class="gradcam-pipeline-arrow">&#8595;</div>
        <div class="gradcam-pipeline-step">
            <div class="step-icon active">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 1 0 20 10 10 0 0 1 0-20z"/><path d="M12 6v6l4 2"/></svg>
            </div>
            <span class="step-label active">Model Analysis</span>
        </div>
        <div class="gradcam-pipeline-arrow">&#8595;</div>
        <div class="gradcam-pipeline-step">
            <div class="step-icon active">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
            </div>
            <span class="step-label active">Grad-CAM Generation</span>
        </div>
        <div class="gradcam-pipeline-arrow">&#8595;</div>
        <div class="gradcam-pipeline-step">
            <div class="step-icon">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            </div>
            <span class="step-label">Visual Explanation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Empty State ----
    if uploaded_img is None:
        st.markdown("""
        <div class="gradcam-empty-state">
            <div class="icon">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3a4a5e" stroke-width="1.2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <path d="M21 15l-5-5L5 21"/>
                </svg>
            </div>
            <h3>Ready to Generate Explanation</h3>
            <p>Upload an image to visualize the regions that influenced the model prediction.</p>
            <div class="status-dot">Grad-CAM engine ready</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- After Image Upload ----
    if uploaded_img is not None:
        try:
            pil_img = Image.open(uploaded_img).convert("RGB")
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            face = extract_face(bgr_img, face_cascade)

            if face is None:
                st.markdown("""
                <div class="gradcam-empty-state" style="border-color:#dc3545;">
                    <div class="icon">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#dc3545" stroke-width="1.2">
                            <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
                        </svg>
                    </div>
                    <h3>No Face Detected</h3>
                    <p>The model could not detect a face in the uploaded image. Please try another image with a clearly visible face.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                augmented = transform(image=face_rgb)
                input_tensor = augmented["image"].unsqueeze(0).to(device)
                rgb_face_float = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

                overlay, prob = generate_gradcam_overlay(model, input_tensor, rgb_face_float, device)

                label = "FAKE" if prob >= threshold else "REAL"
                confidence = abs(prob - 0.5) * 2
                real_prob = 1.0 - prob
                verdict_class = "fake" if label == "FAKE" else "real"
                verdict_icon = "&#10006;" if label == "FAKE" else "&#10004;"

                file_size = uploaded_img.size
                size_mb = file_size / (1024 * 1024)
                img_w, img_h = pil_img.size
                fmt = os.path.splitext(uploaded_img.name)[1].upper()

                # ---- Two-Column Layout ----
                col_input, col_result = st.columns([1, 1], gap="large")

                with col_input:
                    st.markdown("""
                    <div class="analysis-card">
                        <h4>Input Evidence</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(pil_img, caption="Uploaded Image", use_container_width=True)
                    st.markdown(f"""
                    <div style="padding: 4px 0;">
                        <span class="meta-chip">{uploaded_img.name}</span>
                        <span class="meta-chip">{fmt}</span>
                        <span class="meta-chip">{size_mb:.2f} MB</span>
                        <span class="meta-chip">{img_w}x{img_h}</span>
                    </div>
                    """, unsafe_allow_html=True)

                with col_result:
                    st.markdown("""
                    <div class="analysis-card">
                        <h4>Model Assessment</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="margin-top:16px;">
                        <p style="color:#5a6b7e;font-size:12px;margin:0 0 12px 0;">Authenticity Prediction</p>
                        <div class="verdict-badge-lg {verdict_class}">
                            {verdict_icon}&nbsp; {label}
                        </div>
                        <div style="margin-top:20px;">
                            <div class="prob-row">
                                <span class="label">Fake</span>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill fake" style="width:{prob * 100}%;"></div>
                                </div>
                                <span class="value">{prob:.1%}</span>
                            </div>
                            <div class="prob-row">
                                <span class="label">Real</span>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill real" style="width:{real_prob * 100}%;"></div>
                                </div>
                                <span class="value">{real_prob:.1%}</span>
                            </div>
                        </div>
                        <div style="margin-top:16px;padding-top:14px;border-top:1px solid #1e2e3e;">
                            <div class="forensic-detail-row"><span class="dk">Confidence</span><span class="dv">{confidence:.1%}</span></div>
                            <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ---- Grad-CAM Visualization ----
                st.markdown("""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Model Attention Visualization</h4>
                </div>
                """, unsafe_allow_html=True)

                col_orig, col_face, col_overlay = st.columns(3)
                with col_orig:
                    st.markdown("""
                    <div class="gradcam-viz-card">
                        <p class="label">Original Image</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(pil_img, caption="Original input provided to the model", use_container_width=True)

                with col_face:
                    st.markdown("""
                    <div class="gradcam-viz-card">
                        <p class="label">Detected Face</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(face_rgb, caption="Face region extracted for analysis", use_container_width=True)

                with col_overlay:
                    st.markdown("""
                    <div class="gradcam-viz-card">
                        <p class="label">Grad-CAM Overlay</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(overlay, caption="Heatmap overlaid on the original image for interpretation", use_container_width=True)

                # ---- Attention Intensity Legend ----
                st.markdown("""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Attention Intensity</h4>
                    <div style="display:flex;flex-direction:column;gap:4px;max-width:300px;">
                        <div class="legend-item">
                            <div class="legend-swatch" style="background:linear-gradient(90deg,#313695,#4575b4);"></div>
                            <span class="legend-label">Low</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-swatch" style="background:linear-gradient(90deg,#fee090,#fdae61);"></div>
                            <span class="legend-label">Medium</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-swatch" style="background:linear-gradient(90deg,#f46d43,#a50026);"></div>
                            <span class="legend-label">High</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ---- Model Information ----
                st.markdown(f"""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Model Information</h4>
                    <div class="forensic-detail-row"><span class="dk">Model</span><span class="dv">{MODEL_NAME.upper()}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Device</span><span class="dv">{device.upper()}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Analysis</span><span class="dv">Grad-CAM</span></div>
                </div>
                """, unsafe_allow_html=True)

                # ---- Explanation Summary ----
                st.markdown("""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Explanation Summary</h4>
                    <div class="gradcam-interpretation">
                        <p>The Grad-CAM visualization identifies image regions that contributed to the model's prediction. These visualizations provide model interpretability and should be considered supporting evidence rather than definitive proof of manipulation.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ---- Detection Assessment ----
                st.markdown(f"""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Detection Assessment</h4>
                    <div class="forensic-detail-row"><span class="dk">Prediction</span><span class="dv" style="color:{"#dc3545" if label == "FAKE" else "#28a745"};">{label}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Fake Probability</span><span class="dv">{prob:.1%}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Real Probability</span><span class="dv">{real_prob:.1%}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Confidence</span><span class="dv">{confidence:.1%}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Threshold</span><span class="dv">{threshold:.2f}</span></div>
                </div>
                """, unsafe_allow_html=True)

                store_prediction(uploaded_img.name, "image", label,
                                 abs(prob - 0.5) * 2, prob, 0.0)

        except Exception as e:
            logger.error("Grad-CAM processing failed: %s", e)
            st.markdown("""
            <div class="gradcam-empty-state" style="border-color:#dc3545;">
                <div class="icon">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#dc3545" stroke-width="1.2">
                        <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
                    </svg>
                </div>
                <h3>Explanation Unavailable</h3>
                <p>Grad-CAM could not be generated for this input. Please try another compatible image.</p>
            </div>
            """, unsafe_allow_html=True)


# --- PREDICTION HISTORY ---
elif page == "Prediction History":
    st.markdown("## Prediction History")
    st.markdown('<p class="section-label">Review all past analyses stored in the local database</p>', unsafe_allow_html=True)

    db = get_db()
    records = db.recent(limit=200)

    if not records:
        st.info("No predictions yet. Run a detection to see results here.")
    else:
        # Stats
        stats = db.stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", stats["total"])
        c2.metric("Fake", stats["fake"])
        c3.metric("Real", stats["real"])
        c4.metric("Avg Confidence", f"{stats['avg_confidence']:.2%}")

        st.markdown("---")

        # Table
        import pandas as pd
        df = pd.DataFrame(records)
        df = df[["id", "file_name", "media_type", "prediction", "confidence",
                  "fake_probability", "inference_time", "created_at"]]
        df.columns = ["ID", "File", "Type", "Prediction", "Confidence",
                       "Fake Prob", "Time (s)", "Timestamp"]

        # Color the prediction column
        def color_prediction(val):
            if val == "FAKE":
                return "color: #dc3545; font-weight: bold;"
            return "color: #28a745; font-weight: bold;"

        st.dataframe(
            df.style.applymap(color_prediction, subset=["Prediction"]),
            use_container_width=True,
            height=400,
        )

        if st.button("Clear History", type="secondary"):
            db.clear()
            st.success("History cleared.")
            st.rerun()


# --- DASHBOARD ---
elif page == "Dashboard":
    st.markdown("## Analytics Dashboard")
    st.markdown('<p class="section-label">Aggregate statistics from all detections</p>', unsafe_allow_html=True)

    db = get_db()
    stats = db.stats()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{stats['total']}</h2>
            <p>Total Analyses</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card" style="border-left-color: #dc3545;">
            <h2 style="color:#dc3545;">{stats['fake']}</h2>
            <p>Fake Detected</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card" style="border-left-color: #28a745;">
            <h2 style="color:#28a745;">{stats['real']}</h2>
            <p>Authentic</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card" style="border-left-color: #ffc107;">
            <h2 style="color:#ffc107;">{stats['avg_inference_time']:.2f}s</h2>
            <p>Avg Inference Time</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    records = db.recent(limit=200)
    if records:
        import pandas as pd
        df = pd.DataFrame(records)

        # Distribution chart
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Prediction Distribution")
            counts = df["prediction"].value_counts()
            st.bar_chart(counts)

        with col_b:
            st.markdown("### Media Type Breakdown")
            type_counts = df["media_type"].value_counts()
            st.bar_chart(type_counts)

        # Confidence histogram
        st.markdown("### Confidence Distribution")
        st.bar_chart(df["confidence"].hist(bins=20, edgecolor="#1a2636").sort_values(), use_container_width=True)
    else:
        st.info("No data yet. Run some detections to see analytics.")


# --- AUDIO DETECTION ---
elif page == "Audio Detection":
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
    
    /* Grad-CAM Explainability - Forensics UI */
    .gradcam-upload-card {
        background: #141c26; border: 1px dashed #2a3a4e; border-radius: 14px;
        padding: 36px 28px; text-align: center; transition: all 0.25s ease;
    }
    .gradcam-upload-card:hover { border-color: #667eea; background: #182230; }
    .gradcam-upload-card .icon-circle {
        width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 18px;
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.15);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .gradcam-upload-card h3 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
    .gradcam-upload-card .sub { color: #5a6b7e; font-size: 13px; margin: 0 0 14px 0; }
    .gradcam-upload-card .formats { color: #4a5a6e; font-size: 11px; margin: 8px 0 0 0; }

    .gradcam-pipeline-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 16px 0; margin: 20px 0; flex-wrap: wrap;
    }
    .gradcam-pipeline-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 12px;
    }
    .gradcam-pipeline-step .step-icon {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 14px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .gradcam-pipeline-step .step-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .gradcam-pipeline-step .step-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .gradcam-pipeline-step .step-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #4a5a6e; font-weight: 600; }
    .gradcam-pipeline-step .step-label.active { color: #667eea; }
    .gradcam-pipeline-step .step-label.done { color: #28a745; }
    .gradcam-pipeline-arrow { color: #2a3a4e; font-size: 14px; margin: 0 2px; padding-bottom: 18px; }

    .gradcam-empty-state {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 48px 32px; text-align: center; margin-top: 8px;
    }
    .gradcam-empty-state .icon { font-size: 40px; margin-bottom: 16px; opacity: 0.4; }
    .gradcam-empty-state h3 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
    .gradcam-empty-state p { color: #5a6b7e; margin: 0; font-size: 13px; line-height: 1.6; max-width: 400px; margin: 0 auto; }
    .gradcam-empty-state .status-dot {
        display: inline-flex; align-items: center; gap: 6px; margin-top: 14px;
        color: #28a745; font-size: 12px; font-weight: 500;
    }
    .gradcam-empty-state .status-dot::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%; background: #28a745;
    }

    .gradcam-viz-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .gradcam-viz-card .label {
        color: #7a8ba0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
        margin: 0 0 8px 0; font-weight: 600;
    }
    .gradcam-viz-card .desc {
        color: #5a6b7e; font-size: 11px; margin: 8px 0 0 0; line-height: 1.4;
    }

    .legend-item {
        display: flex; align-items: center; gap: 8px; padding: 6px 0;
    }
    .legend-swatch {
        width: 32px; height: 12px; border-radius: 3px;
    }
    .legend-label {
        color: #7a8ba0; font-size: 12px;
    }

    .gradcam-interpretation {
        padding: 16px; background: #0f1620; border-radius: 10px; border: 1px solid #1a2636;
    }
    .gradcam-interpretation p { color: #7a8ba0; font-size: 13px; line-height: 1.7; margin: 0; }


    /* About Page - Forensics UI */
    .about-hero {
        background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0f1923 100%);
        border: 1px solid #1e2e3e; border-radius: 16px; padding: 36px;
        margin-bottom: 24px; text-align: center;
    }
    .about-hero h1 { color: #e8edf3; margin: 0 0 8px 0; font-size: 26px; font-weight: 700; }
    .about-hero p { color: #7a8ba0; margin: 0; font-size: 14px; line-height: 1.5; }

    .about-intro-card {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 28px; margin-bottom: 20px;
    }
    .about-intro-card h3 { color: #e0e0e0; margin: 0 0 12px 0; font-size: 18px; font-weight: 600; }
    .about-intro-card p { color: #7a8ba0; font-size: 14px; line-height: 1.7; margin: 0; }

    .capability-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 20px; text-align: left; transition: border-color 0.2s ease;
        min-height: 140px;
    }
    .capability-card:hover { border-color: #2a3a4e; }
    .capability-card .cap-icon {
        width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
        background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.2); margin-bottom: 12px;
    }
    .capability-card h4 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 14px; font-weight: 600; }
    .capability-card p { color: #5a6b7e; font-size: 12px; line-height: 1.5; margin: 0; }

    .tech-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; text-align: center;
    }
    .tech-card .tech-name { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }
    .tech-card .tech-desc { color: #5a6b7e; font-size: 11px; margin: 4px 0 0 0; }

    .principle-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 24px; text-align: center;
    }
    .principle-card .p-icon {
        width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
        margin: 0 auto 14px; font-size: 20px;
    }
    .principle-card h4 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 15px; font-weight: 600; }
    .principle-card p { color: #5a6b7e; font-size: 12px; line-height: 1.6; margin: 0; }

    .workflow-step {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .workflow-step .step-num {
        color: #667eea; font-size: 11px; font-weight: 700; margin: 0 0 6px 0;
        letter-spacing: 1px;
    }
    .workflow-step .step-title { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }

    .highlight-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; display: flex; align-items: center; gap: 10px;
    }
    .highlight-card .h-dot {
        width: 6px; height: 6px; border-radius: 50%; background: #667eea; flex-shrink: 0;
    }
    .highlight-card .h-text { color: #c0c8d4; font-size: 13px; }

    .about-section-title {
        color: #e0e0e0; font-size: 18px; font-weight: 700; margin: 0 0 8px 0;
    }
    .about-section-sub {
        color: #5a6b7e; font-size: 12px; margin: 0 0 20px 0;
    }

    .config-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 0; border-bottom: 1px solid #1a2636; font-size: 13px;
    }
    .config-row:last-child { border-bottom: none; }
    .config-row .ck { color: #5a6b7e; }
    .config-row .cv { color: #c0c8d4; font-weight: 500; }

</style>
    """, unsafe_allow_html=True)

    # ---- Page Header ----
    st.markdown("""
    <div class="forensics-header">
        <div class="badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
            AUDIO FORENSICS
        </div>
        <h1>Audio Deepfake Detection</h1>
        <p>Analyze speech recordings for patterns associated with synthetic or manipulated audio.</p>
        <p style="color:#5a6b7e;font-size:12px;margin:8px 0 0 0;">AI Audio Forensics Engine &bull; Ready</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Check backends ----
    _AF_AVAIL = False
    _SIMPLE_AVAIL = False
    try:
        from audio_forensics.config import AudioForensicsConfig
        from audio_forensics.inference import AudioInferencePipeline
        _AF_AVAIL = True
    except Exception:
        pass
    try:
        from src.audio_infer import load_audio_detector, predict_from_array as _pf
        _SIMPLE_AVAIL = True
    except Exception:
        pass

    if not _AF_AVAIL and not _SIMPLE_AVAIL:
        st.error("No audio detection backend available. Train the audio model first: `python -m src.audio_train`")
        st.info("Or run the standalone audio app: `streamlit run run_audio_detection.py --server.port 8502`")
    else:
        # ---- Upload Card ----
        st.markdown("""
        <div class="audio-upload-card">
            <div class="icon-circle">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="1.5">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/>
                    <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
            </div>
            <h3>Upload Audio Evidence</h3>
            <p class="sub">Upload a voice recording for AI-powered authenticity analysis.</p>
            <p class="formats">WAV &bull; MP3 &bull; FLAC &bull; OGG &bull; M4A</p>
            <p class="formats" style="margin-top:4px;">Maximum file size: 200 MB</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_audio = st.file_uploader("Upload Audio File",
                                          type=["wav", "mp3", "flac", "ogg", "m4a"],
                                          key="audio_det_upload",
                                          label_visibility="collapsed")

        # ---- Analysis Pipeline (visual only) ----
        audio_uploaded = uploaded_audio is not None
        st.markdown(f"""
        <div class="audio-pipeline-bar">
            <div class="audio-pipeline-step">
                <div class="step-icon {"done" if audio_uploaded else "active"}">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>
                </div>
                <span class="step-label {"done" if audio_uploaded else "active"}">Audio Input</span>
            </div>
            <div class="audio-pipeline-arrow">&#8595;</div>
            <div class="audio-pipeline-step">
                <div class="step-icon active">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h7"/></svg>
                </div>
                <span class="step-label active">Signal Processing</span>
            </div>
            <div class="audio-pipeline-arrow">&#8595;</div>
            <div class="audio-pipeline-step">
                <div class="step-icon active">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>
                </div>
                <span class="step-label active">Feature Extraction</span>
            </div>
            <div class="audio-pipeline-arrow">&#8595;</div>
            <div class="audio-pipeline-step">
                <div class="step-icon active">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 1 0 20 10 10 0 0 1 0-20z"/><path d="M12 6v6l4 2"/></svg>
                </div>
                <span class="step-label active">AI Analysis</span>
            </div>
            <div class="audio-pipeline-arrow">&#8595;</div>
            <div class="audio-pipeline-step">
                <div class="step-icon">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                </div>
                <span class="step-label">Authenticity Assessment</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Analysis Configuration ----
        st.markdown(f"""
        <div class="config-card" style="margin-bottom:20px;">
            <h4>Analysis Configuration</h4>
            <div class="forensic-detail-row">
                <span class="dk">Decision Threshold</span>
                <span class="dv">{threshold:.2f}</span>
            </div>
            <p style="color:#5a6b7e;font-size:11px;margin:10px 0 0 0;">Predictions near the decision boundary should be interpreted with additional caution.</p>
        </div>
        """, unsafe_allow_html=True)

        # ---- Empty State ----
        if uploaded_audio is None:
            st.markdown("""
            <div class="audio-empty-state">
                <div class="icon">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3a4a5e" stroke-width="1.2">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/>
                        <line x1="8" y1="23" x2="16" y2="23"/>
                    </svg>
                </div>
                <h3>Ready for Audio Analysis</h3>
                <p>Upload a voice recording to begin synthetic speech detection.</p>
                <div class="status-dot">Detection engine ready</div>
            </div>
            """, unsafe_allow_html=True)

        # ---- After Audio Upload ----
        if uploaded_audio is not None:
            t0 = time.time()
            suffix = os.path.splitext(uploaded_audio.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_audio.read())
                tmp_path = tmp.name

            try:
                # ---- Two-Column Layout ----
                col_audio, col_result = st.columns([1, 1], gap="large")

                with col_audio:
                    st.markdown("""
                    <div class="analysis-card">
                        <h4>Audio Evidence</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.audio(uploaded_audio)
                    st.caption("Review the uploaded recording before analysis.")

                    # File metadata
                    file_size = uploaded_audio.size
                    size_mb = file_size / (1024 * 1024)
                    fmt = suffix.upper().lstrip(".")

                    st.markdown(f"""
                    <div style="padding: 4px 0;">
                        <span class="meta-chip">{uploaded_audio.name}</span>
                        <span class="meta-chip">{fmt}</span>
                        <span class="meta-chip">{size_mb:.2f} MB</span>
                    </div>
                    """, unsafe_allow_html=True)

                with col_result:
                    st.markdown("""
                    <div class="analysis-card">
                        <h4>Authenticity Assessment</h4>
                    </div>
                    """, unsafe_allow_html=True)

                # ---- Run Analysis ----
                if _AF_AVAIL:
                    with st.spinner("Running audio forensics analysis..."):
                        config = AudioForensicsConfig(fake_threshold=threshold)
                        pipeline = AudioInferencePipeline(config)
                        result = pipeline.analyze(tmp_path)
                    elapsed = time.time() - t0
                    pred_val = result.overall_prediction.value
                    prob = result.overall_probability
                    conf = result.overall_confidence

                    if pred_val in ("POTENTIAL_MANIPULATION", "SUSPICIOUS"):
                        label = "FAKE"
                    elif pred_val == "LOW_SUSPICION":
                        label = "UNCERTAIN"
                    else:
                        label = "REAL"

                    fake_prob = prob if label == "FAKE" else (1.0 - prob if label == "REAL" else prob)
                    real_prob = 1.0 - fake_prob
                    confidence = abs(fake_prob - 0.5) * 2
                    verdict_class = "fake" if label == "FAKE" else ("real" if label == "REAL" else "uncertain")
                    verdict_icon = "&#10006;" if label == "FAKE" else ("&#10004;" if label == "REAL" else "&#9888;")

                    with col_result:
                        st.markdown(f"""
                        <div style="margin-top:48px;">
                            <p style="color:#5a6b7e;font-size:12px;margin:0 0 16px 0;">Authenticity Assessment</p>
                            <div class="verdict-badge-lg {verdict_class}">
                                {verdict_icon}&nbsp; {label}
                            </div>
                            <div style="margin-top:20px;">
                                <div class="prob-row">
                                    <span class="label">Fake</span>
                                    <div class="prob-bar-track">
                                        <div class="prob-bar-fill fake" style="width:{fake_prob * 100}%;"></div>
                                    </div>
                                    <span class="value">{fake_prob:.1%}</span>
                                </div>
                                <div class="prob-row">
                                    <span class="label">Real</span>
                                    <div class="prob-bar-track">
                                        <div class="prob-bar-fill real" style="width:{real_prob * 100}%;"></div>
                                    </div>
                                    <span class="value">{real_prob:.1%}</span>
                                </div>
                            </div>
                            <div style="margin-top:16px;padding-top:14px;border-top:1px solid #1e2e3e;">
                                <div class="forensic-detail-row"><span class="dk">Confidence</span><span class="dv">{confidence:.1%}</span></div>
                                <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ---- Model Confidence ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Model Confidence</h4>
                        <div class="prob-row">
                            <span class="label">Real</span>
                            <div class="prob-bar-track">
                                <div class="prob-bar-fill real" style="width:{real_prob * 100}%;"></div>
                            </div>
                            <span class="value">{real_prob:.1%}</span>
                        </div>
                        <div class="prob-row">
                            <span class="label">Fake</span>
                            <div class="prob-bar-track">
                                <div class="prob-bar-fill fake" style="width:{fake_prob * 100}%;"></div>
                            </div>
                            <span class="value">{fake_prob:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Audio Forensic Analysis ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Audio Forensic Analysis</h4>
                        <div class="forensic-detail-row"><span class="dk">Analysis Status</span><span class="dv" style="color:#28a745;">Completed</span></div>
                        <div class="forensic-detail-row"><span class="dk">Evidence Type</span><span class="dv">Audio</span></div>
                        <div class="forensic-detail-row"><span class="dk">Model</span><span class="dv">{AUDIO_MODEL_NAME.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Device</span><span class="dv">{device.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Inference Time</span><span class="dv">{elapsed:.2f}s</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Segment Analysis (if available) ----
                    if result.segments:
                        st.markdown("""
                        <div class="analysis-card" style="margin-top:16px;">
                            <h4>Audio Forensic Timeline</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for seg in result.segments:
                            sc = "#dc3545" if seg.prediction.value in ("SUSPICIOUS", "POTENTIAL_MANIPULATION") else "#ffc107" if seg.prediction.value == "LOW_SUSPICION" else "#28a745"
                            st.markdown(f'<div style="padding:6px 0;border-bottom:1px solid #1a2636;"><span style="color:#5a6b7e;font-size:12px;">{seg.start_time:.1f}s - {seg.end_time:.1f}s</span> &nbsp;<span style="color:{sc};font-size:12px;font-weight:600;">{seg.prediction.value}</span> <span style="color:#7a8ba0;font-size:12px;">({seg.probability:.1%})</span></div>', unsafe_allow_html=True)

                    # ---- Evidence (if available) ----
                    if result.evidence:
                        st.markdown("""
                        <div class="analysis-card" style="margin-top:16px;">
                            <h4>Forensic Evidence</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for ev in result.evidence:
                            st.markdown(f'<div style="padding:4px 0;color:#7a8ba0;font-size:13px;">&bull; {ev}</div>', unsafe_allow_html=True)

                    # ---- Assessment Interpretation ----
                    if label == "FAKE":
                        interp = "Potentially Synthetic Audio"
                        interp_desc = "The model detected patterns associated with synthetic or manipulated speech."
                    elif label == "UNCERTAIN":
                        interp = "Requires Further Review"
                        interp_desc = "The model confidence is close to the decision threshold. Additional analysis is recommended."
                    else:
                        interp = "Likely Authentic Audio"
                        interp_desc = "The model did not detect strong evidence of synthetic speech under the current analysis settings."

                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Assessment Interpretation</h4>
                        <div class="audio-interpretation">
                            <p style="color:#e0e0e0;font-size:14px;font-weight:600;margin:0 0 8px 0;">{interp}</p>
                            <p>{interp_desc}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Forensic Summary ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Forensic Summary</h4>
                        <div class="forensic-detail-row"><span class="dk">Evidence Type</span><span class="dv">Audio</span></div>
                        <div class="forensic-detail-row"><span class="dk">Analysis</span><span class="dv">AI-based speech authenticity assessment</span></div>
                        <div class="forensic-detail-row"><span class="dk">Model</span><span class="dv">{AUDIO_MODEL_NAME.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Device</span><span class="dv">{device.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Status</span><span class="dv" style="color:#28a745;">Completed</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    store_prediction(
                        file_name=uploaded_audio.name,
                        media_type="audio",
                        prediction=label,
                        confidence=confidence,
                        fake_prob=fake_prob,
                        inference_time=elapsed,
                    )

                elif _SIMPLE_AVAIL:
                    with st.spinner("Analyzing audio..."):
                        import librosa
                        audio_model, audio_device = load_audio_detector()
                        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
                        prob = _pf(y, audio_model, audio_device)
                    elapsed = time.time() - t0
                    label = "FAKE" if prob >= threshold else "REAL"
                    fake_prob = float(prob)
                    real_prob = 1.0 - fake_prob
                    confidence = abs(fake_prob - 0.5) * 2
                    verdict_class = "fake" if label == "FAKE" else "real"
                    verdict_icon = "&#10006;" if label == "FAKE" else "&#10004;"

                    with col_result:
                        st.markdown(f"""
                        <div style="margin-top:48px;">
                            <p style="color:#5a6b7e;font-size:12px;margin:0 0 16px 0;">Authenticity Assessment</p>
                            <div class="verdict-badge-lg {verdict_class}">
                                {verdict_icon}&nbsp; {label}
                            </div>
                            <div style="margin-top:20px;">
                                <div class="prob-row">
                                    <span class="label">Fake</span>
                                    <div class="prob-bar-track">
                                        <div class="prob-bar-fill fake" style="width:{fake_prob * 100}%;"></div>
                                    </div>
                                    <span class="value">{fake_prob:.1%}</span>
                                </div>
                                <div class="prob-row">
                                    <span class="label">Real</span>
                                    <div class="prob-bar-track">
                                        <div class="prob-bar-fill real" style="width:{real_prob * 100}%;"></div>
                                    </div>
                                    <span class="value">{real_prob:.1%}</span>
                                </div>
                            </div>
                            <div style="margin-top:16px;padding-top:14px;border-top:1px solid #1e2e3e;">
                                <div class="forensic-detail-row"><span class="dk">Confidence</span><span class="dv">{confidence:.1%}</span></div>
                                <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ---- Model Confidence ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Model Confidence</h4>
                        <div class="prob-row">
                            <span class="label">Real</span>
                            <div class="prob-bar-track">
                                <div class="prob-bar-fill real" style="width:{real_prob * 100}%;"></div>
                            </div>
                            <span class="value">{real_prob:.1%}</span>
                        </div>
                        <div class="prob-row">
                            <span class="label">Fake</span>
                            <div class="prob-bar-track">
                                <div class="prob-bar-fill fake" style="width:{fake_prob * 100}%;"></div>
                            </div>
                            <span class="value">{fake_prob:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Audio Forensic Analysis ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Audio Forensic Analysis</h4>
                        <div class="forensic-detail-row"><span class="dk">Analysis Status</span><span class="dv" style="color:#28a745;">Completed</span></div>
                        <div class="forensic-detail-row"><span class="dk">Evidence Type</span><span class="dv">Audio</span></div>
                        <div class="forensic-detail-row"><span class="dk">Model</span><span class="dv">{AUDIO_MODEL_NAME.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Device</span><span class="dv">{device.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Inference Time</span><span class="dv">{elapsed:.2f}s</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Audio Forensic Timeline (not available in simple mode) ----
                    st.markdown("""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Audio Forensic Timeline</h4>
                        <div class="audio-interpretation">
                            <p>Segment-level analysis will appear when temporal predictions are available.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Assessment Interpretation ----
                    if label == "FAKE":
                        interp = "Potentially Synthetic Audio"
                        interp_desc = "The model detected patterns associated with synthetic or manipulated speech."
                    else:
                        interp = "Likely Authentic Audio"
                        interp_desc = "The model did not detect strong evidence of synthetic speech under the current analysis settings."

                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Assessment Interpretation</h4>
                        <div class="audio-interpretation">
                            <p style="color:#e0e0e0;font-size:14px;font-weight:600;margin:0 0 8px 0;">{interp}</p>
                            <p>{interp_desc}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Forensic Summary ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Forensic Summary</h4>
                        <div class="forensic-detail-row"><span class="dk">Evidence Type</span><span class="dv">Audio</span></div>
                        <div class="forensic-detail-row"><span class="dk">Analysis</span><span class="dv">AI-based speech authenticity assessment</span></div>
                        <div class="forensic-detail-row"><span class="dk">Model</span><span class="dv">{AUDIO_MODEL_NAME.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Device</span><span class="dv">{device.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Status</span><span class="dv" style="color:#28a745;">Completed</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    store_prediction(
                        file_name=uploaded_audio.name,
                        media_type="audio",
                        prediction=label,
                        confidence=confidence,
                        fake_prob=fake_prob,
                        inference_time=elapsed,
                    )

            except Exception as e:
                st.error(f"Analysis failed: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# --- ABOUT ---
elif page == "About":
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

    # ---- Page Header ----
    st.markdown("""
    <div class="forensics-header">
        <div class="badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            AI FORENSICS PLATFORM
        </div>
        <h1>About Deepfake Detector</h1>
        <p>AI-powered multimodal media authenticity and forensic analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- System Introduction ----
    st.markdown("""
    <div class="analysis-card" style="margin-bottom:20px;">
        <div style="display:flex;align-items:flex-start;gap:14px;">
            <div style="min-width:36px;height:36px;border-radius:8px;background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.2);display:flex;align-items:center;justify-content:center;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
            </div>
            <div>
                <h4 style="color:#e0e0e0;margin:0 0 6px 0;font-size:16px;text-transform:none;letter-spacing:0;">AI-Powered Explainable Deepfake Detection</h4>
                <p style="color:#7a8ba0;font-size:14px;margin:0;line-height:1.6;">An AI-powered platform designed to analyze images, videos, and audio for patterns associated with synthetic or manipulated media, with explainability and forensic evidence supporting model assessments.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- System Overview ----
    st.markdown("""
    <div class="analysis-card" style="margin-bottom:20px;">
        <h4>System Overview</h4>
        <p style="color:#7a8ba0;font-size:14px;line-height:1.7;margin:0 0 10px 0;">
            This deepfake detection system uses state-of-the-art deep learning models to analyze images, videos, and audio for signs of synthetic manipulation. The system provides explainable AI through Grad-CAM heatmaps, giving transparent visual evidence for every decision.
        </p>
        <p style="color:#7a8ba0;font-size:14px;line-height:1.7;margin:0;">
            The platform supports multimodal analysis including image forensics, video frame-level assessment, audio deepfake detection, and audio-video fusion for combined video verdicts. Results are model assessments rather than absolute proof.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Model Architecture ----
    st.markdown("""
    <div class="analysis-card" style="margin-bottom:20px;">
        <h4>Model Architecture</h4>
        <div style="display:flex;align-items:flex-start;gap:14px;">
            <div style="min-width:44px;height:44px;border-radius:10px;background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.2);display:flex;align-items:center;justify-content:center;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
            </div>
            <div>
                <p style="color:#e0e0e0;font-size:15px;font-weight:600;margin:0 0 8px 0;">Xception</p>
                <p style="color:#7a8ba0;font-size:13px;line-height:1.6;margin:0;">A deep CNN architecture based on depthwise separable convolutions, achieving state-of-the-art performance on face forgery detection. Trained on Celeb-DF v2 dataset with input resolution of 299x299 pixels.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Core Capabilities ----
    st.markdown("""
    <p class="about-section-title">Core Capabilities</p>
    <p class="about-section-sub">Multimodal detection across image, video, and audio media.</p>
    """, unsafe_allow_html=True)

    caps = [
        ("Image Detection", "Analyze images for patterns associated with manipulated or synthetic visual content.", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>'),
        ("Video Detection", "Analyze sampled video frames and temporal evidence for potential manipulation.", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>'),
        ("Audio Detection", "Analyze speech recordings for patterns associated with synthetic or manipulated audio.", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>'),
        ("Real-Time Webcam", "Perform real-time visual analysis using the webcam detection pipeline.", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>'),
        ("Grad-CAM Explainability", "Visualize image regions that contributed to the model prediction.", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>'),
        ("Audio-Video Fusion", "Combine available visual and audio evidence for multimodal video analysis.", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42"/></svg>'),
        ("Prediction History", "Maintain analysis records using the existing prediction history system.", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'),
        ("Analytics Dashboard", "Present aggregate detection statistics and model analysis information.", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>'),
    ]

    cols = st.columns(4)
    for i, (title, desc, icon) in enumerate(caps):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="capability-card">
                <div class="cap-icon">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Multimodal Analysis ----
    st.markdown("""
    <div class="analysis-card" style="margin-bottom:20px;">
        <h4>Multimodal Deepfake Analysis</h4>
    </div>
    """, unsafe_allow_html=True)

    flow_cols = st.columns([1, 0.3, 1, 0.3, 1, 0.3, 1])
    with flow_cols[0]:
        st.markdown("""
        <div class="capability-card" style="text-align:center;min-height:120px;">
            <div class="cap-icon" style="margin:0 auto 10px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg></div>
            <h4>Image</h4>
            <p>Visual Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    with flow_cols[1]:
        st.markdown('<p style="text-align:center;color:#2a3a4e;font-size:20px;margin-top:40px;">&#8595;</p>', unsafe_allow_html=True)
    with flow_cols[2]:
        st.markdown("""
        <div class="capability-card" style="text-align:center;min-height:120px;">
            <div class="cap-icon" style="margin:0 auto 10px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg></div>
            <h4>Video</h4>
            <p>Frame-Level + Temporal</p>
        </div>
        """, unsafe_allow_html=True)
    with flow_cols[3]:
        st.markdown('<p style="text-align:center;color:#2a3a4e;font-size:20px;margin-top:40px;">&#8595;</p>', unsafe_allow_html=True)
    with flow_cols[4]:
        st.markdown("""
        <div class="capability-card" style="text-align:center;min-height:120px;">
            <div class="cap-icon" style="margin:0 auto 10px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg></div>
            <h4>Audio</h4>
            <p>Audio Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    with flow_cols[5]:
        st.markdown('<p style="text-align:center;color:#2a3a4e;font-size:20px;margin-top:40px;">+</p>', unsafe_allow_html=True)
    with flow_cols[6]:
        st.markdown("""
        <div class="capability-card" style="text-align:center;min-height:120px;">
            <div class="cap-icon" style="margin:0 auto 10px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42"/></svg></div>
            <h4>Fusion</h4>
            <p>Multimodal Assessment</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p style="text-align:center;color:#2a3a4e;font-size:20px;margin:8px 0;">&#8595;</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:12px 24px;background:#141c26;border:1px solid #1e2e3e;border-radius:10px;max-width:300px;margin:0 auto;">
        <span style="color:#e0e0e0;font-size:14px;font-weight:600;">Forensic Assessment</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Explainable AI ----
    st.markdown("""
    <div class="analysis-card" style="margin-bottom:20px;">
        <h4>Explainable AI</h4>
        <p style="color:#7a8ba0;font-size:14px;line-height:1.7;margin:0 0 12px 0;">
            Grad-CAM provides visual insight into regions that influenced the model prediction, helping users understand the model's attention.
        </p>
        <div style="padding:14px;background:#0f1620;border-radius:10px;border:1px solid #1a2636;">
            <p style="color:#ffc107;font-size:13px;margin:0;">Grad-CAM is an interpretability tool and should be treated as supporting evidence rather than definitive proof of manipulation.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Analysis Workflow ----
    st.markdown("""
    <p class="about-section-title">Analysis Workflow</p>
    <p class="about-section-sub">Step-by-step process from media input to forensic assessment.</p>
    """, unsafe_allow_html=True)

    workflow_steps = [
        ("01", "Upload Media"),
        ("02", "Preprocess Evidence"),
        ("03", "AI Model Analysis"),
        ("04", "Generate Evidence"),
        ("05", "Review Assessment"),
        ("06", "Store History"),
    ]

    wf_cols = st.columns(6)
    for i, (num, title) in enumerate(workflow_steps):
        with wf_cols[i]:
            st.markdown(f"""
            <div class="workflow-step">
                <p class="step-num">{num}</p>
                <p class="step-title">{title}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Technology Stack ----
    st.markdown("""
    <p class="about-section-title">Technology Stack</p>
    <p class="about-section-sub">Core technologies powering the detection platform.</p>
    """, unsafe_allow_html=True)

    techs = [
        ("Python", "Core language"),
        ("PyTorch", "Deep learning"),
        ("Xception", "Detection CNN"),
        ("OpenCV", "Image processing"),
        ("Streamlit", "Web interface"),
        ("SQLite", "Data storage"),
        ("Grad-CAM", "Explainability"),
        ("librosa", "Audio analysis"),
        ("EfficientNet", "Audio model"),
    ]

    tech_cols = st.columns(3)
    for i, (name, desc) in enumerate(techs):
        with tech_cols[i % 3]:
            st.markdown(f"""
            <div class="tech-card">
                <p class="tech-name">{name}</p>
                <p class="tech-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Forensic Analysis Principles ----
    st.markdown("""
    <p class="about-section-title">Forensic Analysis Principles</p>
    <p class="about-section-sub">Guiding principles for responsible forensic analysis.</p>
    """, unsafe_allow_html=True)

    principles = [
        ("Evidence-Based", "Present model outputs and available evidence clearly.", '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>'),
        ("Explainable", "Provide visual explanations where supported by the model.", '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>'),
        ("Responsible", "Treat automated predictions as model assessments requiring appropriate context and review.", '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'),
    ]

    princ_cols = st.columns(3)
    for i, (title, desc, icon) in enumerate(principles):
        with princ_cols[i]:
            st.markdown(f"""
            <div class="principle-card">
                <div class="p-icon" style="background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.2);">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Project Highlights ----
    st.markdown("""
    <p class="about-section-title">Project Highlights</p>
    <p class="about-section-sub">Key features and capabilities of the platform.</p>
    """, unsafe_allow_html=True)

    highlights = [
        "Multimodal deepfake detection",
        "Explainable AI with Grad-CAM",
        "Video frame-level analysis",
        "Audio deepfake detection",
        "Audio-video fusion",
        "Real-time webcam analysis",
        "Prediction history",
        "Analytics dashboard",
    ]

    hl_cols = st.columns(4)
    for i, h in enumerate(highlights):
        with hl_cols[i % 4]:
            st.markdown(f"""
            <div class="highlight-card">
                <div class="h-dot"></div>
                <span class="h-text">{h}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Responsible Use ----
    st.markdown("""
    <div class="analysis-card" style="margin-bottom:20px;">
        <h4>Responsible Use</h4>
        <p style="color:#7a8ba0;font-size:14px;line-height:1.7;margin:0;">
            Deepfake detection models provide probabilistic assessments and may produce incorrect results depending on media quality, manipulation techniques, dataset characteristics, and other factors. Results should be interpreted alongside relevant context and additional evidence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- System Configuration ----
    st.markdown(f"""
    <div class="analysis-card" style="margin-bottom:20px;">
        <h4>Current System Configuration</h4>
        <div class="config-row"><span class="ck">Model</span><span class="cv">{MODEL_NAME.upper()}</span></div>
        <div class="config-row"><span class="ck">Device</span><span class="cv">{device.upper()}</span></div>
        <div class="config-row"><span class="ck">Decision Threshold</span><span class="cv">{threshold:.2f}</span></div>
        <div class="config-row"><span class="ck">Supported Media</span><span class="cv">Image, Video, Audio</span></div>
    </div>
    """, unsafe_allow_html=True)


# --- AUDIO DETECTION ---
elif page == "Audio Detection":
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
    
    /* Grad-CAM Explainability - Forensics UI */
    .gradcam-upload-card {
        background: #141c26; border: 1px dashed #2a3a4e; border-radius: 14px;
        padding: 36px 28px; text-align: center; transition: all 0.25s ease;
    }
    .gradcam-upload-card:hover { border-color: #667eea; background: #182230; }
    .gradcam-upload-card .icon-circle {
        width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 18px;
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.15);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .gradcam-upload-card h3 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; }
    .gradcam-upload-card .sub { color: #5a6b7e; font-size: 13px; margin: 0 0 14px 0; }
    .gradcam-upload-card .formats { color: #4a5a6e; font-size: 11px; margin: 8px 0 0 0; }

    .gradcam-pipeline-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 16px 0; margin: 20px 0; flex-wrap: wrap;
    }
    .gradcam-pipeline-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 12px;
    }
    .gradcam-pipeline-step .step-icon {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 14px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .gradcam-pipeline-step .step-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .gradcam-pipeline-step .step-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .gradcam-pipeline-step .step-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #4a5a6e; font-weight: 600; }
    .gradcam-pipeline-step .step-label.active { color: #667eea; }
    .gradcam-pipeline-step .step-label.done { color: #28a745; }
    .gradcam-pipeline-arrow { color: #2a3a4e; font-size: 14px; margin: 0 2px; padding-bottom: 18px; }

    .gradcam-empty-state {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 48px 32px; text-align: center; margin-top: 8px;
    }
    .gradcam-empty-state .icon { font-size: 40px; margin-bottom: 16px; opacity: 0.4; }
    .gradcam-empty-state h3 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
    .gradcam-empty-state p { color: #5a6b7e; margin: 0; font-size: 13px; line-height: 1.6; max-width: 400px; margin: 0 auto; }
    .gradcam-empty-state .status-dot {
        display: inline-flex; align-items: center; gap: 6px; margin-top: 14px;
        color: #28a745; font-size: 12px; font-weight: 500;
    }
    .gradcam-empty-state .status-dot::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%; background: #28a745;
    }

    .gradcam-viz-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .gradcam-viz-card .label {
        color: #7a8ba0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
        margin: 0 0 8px 0; font-weight: 600;
    }
    .gradcam-viz-card .desc {
        color: #5a6b7e; font-size: 11px; margin: 8px 0 0 0; line-height: 1.4;
    }

    .legend-item {
        display: flex; align-items: center; gap: 8px; padding: 6px 0;
    }
    .legend-swatch {
        width: 32px; height: 12px; border-radius: 3px;
    }
    .legend-label {
        color: #7a8ba0; font-size: 12px;
    }

    .gradcam-interpretation {
        padding: 16px; background: #0f1620; border-radius: 10px; border: 1px solid #1a2636;
    }
    .gradcam-interpretation p { color: #7a8ba0; font-size: 13px; line-height: 1.7; margin: 0; }


    /* About Page - Forensics UI */
    .about-hero {
        background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0f1923 100%);
        border: 1px solid #1e2e3e; border-radius: 16px; padding: 36px;
        margin-bottom: 24px; text-align: center;
    }
    .about-hero h1 { color: #e8edf3; margin: 0 0 8px 0; font-size: 26px; font-weight: 700; }
    .about-hero p { color: #7a8ba0; margin: 0; font-size: 14px; line-height: 1.5; }

    .about-intro-card {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
        padding: 28px; margin-bottom: 20px;
    }
    .about-intro-card h3 { color: #e0e0e0; margin: 0 0 12px 0; font-size: 18px; font-weight: 600; }
    .about-intro-card p { color: #7a8ba0; font-size: 14px; line-height: 1.7; margin: 0; }

    .capability-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 20px; text-align: left; transition: border-color 0.2s ease;
        min-height: 140px;
    }
    .capability-card:hover { border-color: #2a3a4e; }
    .capability-card .cap-icon {
        width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
        background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.2); margin-bottom: 12px;
    }
    .capability-card h4 { color: #e0e0e0; margin: 0 0 6px 0; font-size: 14px; font-weight: 600; }
    .capability-card p { color: #5a6b7e; font-size: 12px; line-height: 1.5; margin: 0; }

    .tech-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; text-align: center;
    }
    .tech-card .tech-name { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }
    .tech-card .tech-desc { color: #5a6b7e; font-size: 11px; margin: 4px 0 0 0; }

    .principle-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 12px;
        padding: 24px; text-align: center;
    }
    .principle-card .p-icon {
        width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
        margin: 0 auto 14px; font-size: 20px;
    }
    .principle-card h4 { color: #e0e0e0; margin: 0 0 8px 0; font-size: 15px; font-weight: 600; }
    .principle-card p { color: #5a6b7e; font-size: 12px; line-height: 1.6; margin: 0; }

    .workflow-step {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .workflow-step .step-num {
        color: #667eea; font-size: 11px; font-weight: 700; margin: 0 0 6px 0;
        letter-spacing: 1px;
    }
    .workflow-step .step-title { color: #e0e0e0; font-size: 13px; font-weight: 600; margin: 0; }

    .highlight-card {
        background: #0f1620; border: 1px solid #1a2636; border-radius: 10px;
        padding: 14px 16px; display: flex; align-items: center; gap: 10px;
    }
    .highlight-card .h-dot {
        width: 6px; height: 6px; border-radius: 50%; background: #667eea; flex-shrink: 0;
    }
    .highlight-card .h-text { color: #c0c8d4; font-size: 13px; }

    .about-section-title {
        color: #e0e0e0; font-size: 18px; font-weight: 700; margin: 0 0 8px 0;
    }
    .about-section-sub {
        color: #5a6b7e; font-size: 12px; margin: 0 0 20px 0;
    }

    .config-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 0; border-bottom: 1px solid #1a2636; font-size: 13px;
    }
    .config-row:last-child { border-bottom: none; }
    .config-row .ck { color: #5a6b7e; }
    .config-row .cv { color: #c0c8d4; font-weight: 500; }

</style>
    """, unsafe_allow_html=True)

    # ---- Page Header ----
    st.markdown("""
    <div class="forensics-header">
        <div class="badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
            AUDIO FORENSICS
        </div>
        <h1>Audio Deepfake Detection</h1>
        <p>Analyze speech recordings for patterns associated with synthetic or manipulated audio.</p>
        <p style="color:#5a6b7e;font-size:12px;margin:8px 0 0 0;">AI Audio Forensics Engine &bull; Ready</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Check backends ----
    _AF_AVAIL = False
    _SIMPLE_AVAIL = False
    try:
        from audio_forensics.config import AudioForensicsConfig
        from audio_forensics.inference import AudioInferencePipeline
        _AF_AVAIL = True
    except Exception:
        pass
    try:
        from src.audio_infer import load_audio_detector, predict_from_array as _pf
        _SIMPLE_AVAIL = True
    except Exception:
        pass

    if not _AF_AVAIL and not _SIMPLE_AVAIL:
        st.error("No audio detection backend available. Train the audio model first: `python -m src.audio_train`")
        st.info("Or run the standalone audio app: `streamlit run run_audio_detection.py --server.port 8502`")
    else:
        # ---- Upload Card ----
        st.markdown("""
        <div class="audio-upload-card">
            <div class="icon-circle">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="1.5">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/>
                    <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
            </div>
            <h3>Upload Audio Evidence</h3>
            <p class="sub">Upload a voice recording for AI-powered authenticity analysis.</p>
            <p class="formats">WAV &bull; MP3 &bull; FLAC &bull; OGG &bull; M4A</p>
            <p class="formats" style="margin-top:4px;">Maximum file size: 200 MB</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_audio = st.file_uploader("Upload Audio File",
                                          type=["wav", "mp3", "flac", "ogg", "m4a"],
                                          key="audio_det_upload",
                                          label_visibility="collapsed")

        # ---- Analysis Pipeline (visual only) ----
        audio_uploaded = uploaded_audio is not None
        st.markdown(f"""
        <div class="audio-pipeline-bar">
            <div class="audio-pipeline-step">
                <div class="step-icon {"done" if audio_uploaded else "active"}">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>
                </div>
                <span class="step-label {"done" if audio_uploaded else "active"}">Audio Input</span>
            </div>
            <div class="audio-pipeline-arrow">&#8595;</div>
            <div class="audio-pipeline-step">
                <div class="step-icon active">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h7"/></svg>
                </div>
                <span class="step-label active">Signal Processing</span>
            </div>
            <div class="audio-pipeline-arrow">&#8595;</div>
            <div class="audio-pipeline-step">
                <div class="step-icon active">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>
                </div>
                <span class="step-label active">Feature Extraction</span>
            </div>
            <div class="audio-pipeline-arrow">&#8595;</div>
            <div class="audio-pipeline-step">
                <div class="step-icon active">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 1 0 20 10 10 0 0 1 0-20z"/><path d="M12 6v6l4 2"/></svg>
                </div>
                <span class="step-label active">AI Analysis</span>
            </div>
            <div class="audio-pipeline-arrow">&#8595;</div>
            <div class="audio-pipeline-step">
                <div class="step-icon">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                </div>
                <span class="step-label">Authenticity Assessment</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Analysis Configuration ----
        st.markdown(f"""
        <div class="config-card" style="margin-bottom:20px;">
            <h4>Analysis Configuration</h4>
            <div class="forensic-detail-row">
                <span class="dk">Decision Threshold</span>
                <span class="dv">{threshold:.2f}</span>
            </div>
            <p style="color:#5a6b7e;font-size:11px;margin:10px 0 0 0;">Predictions near the decision boundary should be interpreted with additional caution.</p>
        </div>
        """, unsafe_allow_html=True)

        # ---- Empty State ----
        if uploaded_audio is None:
            st.markdown("""
            <div class="audio-empty-state">
                <div class="icon">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3a4a5e" stroke-width="1.2">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/>
                        <line x1="8" y1="23" x2="16" y2="23"/>
                    </svg>
                </div>
                <h3>Ready for Audio Analysis</h3>
                <p>Upload a voice recording to begin synthetic speech detection.</p>
                <div class="status-dot">Detection engine ready</div>
            </div>
            """, unsafe_allow_html=True)

        # ---- After Audio Upload ----
        if uploaded_audio is not None:
            t0 = time.time()
            suffix = os.path.splitext(uploaded_audio.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_audio.read())
                tmp_path = tmp.name

            try:
                # ---- Two-Column Layout ----
                col_audio, col_result = st.columns([1, 1], gap="large")

                with col_audio:
                    st.markdown("""
                    <div class="analysis-card">
                        <h4>Audio Evidence</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.audio(uploaded_audio)
                    st.caption("Review the uploaded recording before analysis.")

                    # File metadata
                    file_size = uploaded_audio.size
                    size_mb = file_size / (1024 * 1024)
                    fmt = suffix.upper().lstrip(".")

                    st.markdown(f"""
                    <div style="padding: 4px 0;">
                        <span class="meta-chip">{uploaded_audio.name}</span>
                        <span class="meta-chip">{fmt}</span>
                        <span class="meta-chip">{size_mb:.2f} MB</span>
                    </div>
                    """, unsafe_allow_html=True)

                with col_result:
                    st.markdown("""
                    <div class="analysis-card">
                        <h4>Authenticity Assessment</h4>
                    </div>
                    """, unsafe_allow_html=True)

                # ---- Run Analysis ----
                if _AF_AVAIL:
                    with st.spinner("Running audio forensics analysis..."):
                        config = AudioForensicsConfig(fake_threshold=threshold)
                        pipeline = AudioInferencePipeline(config)
                        result = pipeline.analyze(tmp_path)
                    elapsed = time.time() - t0
                    pred_val = result.overall_prediction.value
                    prob = result.overall_probability
                    conf = result.overall_confidence

                    if pred_val in ("POTENTIAL_MANIPULATION", "SUSPICIOUS"):
                        label = "FAKE"
                    elif pred_val == "LOW_SUSPICION":
                        label = "UNCERTAIN"
                    else:
                        label = "REAL"

                    fake_prob = prob if label == "FAKE" else (1.0 - prob if label == "REAL" else prob)
                    real_prob = 1.0 - fake_prob
                    confidence = abs(fake_prob - 0.5) * 2
                    verdict_class = "fake" if label == "FAKE" else ("real" if label == "REAL" else "uncertain")
                    verdict_icon = "&#10006;" if label == "FAKE" else ("&#10004;" if label == "REAL" else "&#9888;")

                    with col_result:
                        st.markdown(f"""
                        <div style="margin-top:48px;">
                            <p style="color:#5a6b7e;font-size:12px;margin:0 0 16px 0;">Authenticity Assessment</p>
                            <div class="verdict-badge-lg {verdict_class}">
                                {verdict_icon}&nbsp; {label}
                            </div>
                            <div style="margin-top:20px;">
                                <div class="prob-row">
                                    <span class="label">Fake</span>
                                    <div class="prob-bar-track">
                                        <div class="prob-bar-fill fake" style="width:{fake_prob * 100}%;"></div>
                                    </div>
                                    <span class="value">{fake_prob:.1%}</span>
                                </div>
                                <div class="prob-row">
                                    <span class="label">Real</span>
                                    <div class="prob-bar-track">
                                        <div class="prob-bar-fill real" style="width:{real_prob * 100}%;"></div>
                                    </div>
                                    <span class="value">{real_prob:.1%}</span>
                                </div>
                            </div>
                            <div style="margin-top:16px;padding-top:14px;border-top:1px solid #1e2e3e;">
                                <div class="forensic-detail-row"><span class="dk">Confidence</span><span class="dv">{confidence:.1%}</span></div>
                                <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ---- Model Confidence ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Model Confidence</h4>
                        <div class="prob-row">
                            <span class="label">Real</span>
                            <div class="prob-bar-track">
                                <div class="prob-bar-fill real" style="width:{real_prob * 100}%;"></div>
                            </div>
                            <span class="value">{real_prob:.1%}</span>
                        </div>
                        <div class="prob-row">
                            <span class="label">Fake</span>
                            <div class="prob-bar-track">
                                <div class="prob-bar-fill fake" style="width:{fake_prob * 100}%;"></div>
                            </div>
                            <span class="value">{fake_prob:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Audio Forensic Analysis ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Audio Forensic Analysis</h4>
                        <div class="forensic-detail-row"><span class="dk">Analysis Status</span><span class="dv" style="color:#28a745;">Completed</span></div>
                        <div class="forensic-detail-row"><span class="dk">Evidence Type</span><span class="dv">Audio</span></div>
                        <div class="forensic-detail-row"><span class="dk">Model</span><span class="dv">{AUDIO_MODEL_NAME.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Device</span><span class="dv">{device.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Inference Time</span><span class="dv">{elapsed:.2f}s</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Segment Analysis (if available) ----
                    if result.segments:
                        st.markdown("""
                        <div class="analysis-card" style="margin-top:16px;">
                            <h4>Audio Forensic Timeline</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for seg in result.segments:
                            sc = "#dc3545" if seg.prediction.value in ("SUSPICIOUS", "POTENTIAL_MANIPULATION") else "#ffc107" if seg.prediction.value == "LOW_SUSPICION" else "#28a745"
                            st.markdown(f'<div style="padding:6px 0;border-bottom:1px solid #1a2636;"><span style="color:#5a6b7e;font-size:12px;">{seg.start_time:.1f}s - {seg.end_time:.1f}s</span> &nbsp;<span style="color:{sc};font-size:12px;font-weight:600;">{seg.prediction.value}</span> <span style="color:#7a8ba0;font-size:12px;">({seg.probability:.1%})</span></div>', unsafe_allow_html=True)

                    # ---- Evidence (if available) ----
                    if result.evidence:
                        st.markdown("""
                        <div class="analysis-card" style="margin-top:16px;">
                            <h4>Forensic Evidence</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for ev in result.evidence:
                            st.markdown(f'<div style="padding:4px 0;color:#7a8ba0;font-size:13px;">&bull; {ev}</div>', unsafe_allow_html=True)

                    # ---- Assessment Interpretation ----
                    if label == "FAKE":
                        interp = "Potentially Synthetic Audio"
                        interp_desc = "The model detected patterns associated with synthetic or manipulated speech."
                    elif label == "UNCERTAIN":
                        interp = "Requires Further Review"
                        interp_desc = "The model confidence is close to the decision threshold. Additional analysis is recommended."
                    else:
                        interp = "Likely Authentic Audio"
                        interp_desc = "The model did not detect strong evidence of synthetic speech under the current analysis settings."

                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Assessment Interpretation</h4>
                        <div class="audio-interpretation">
                            <p style="color:#e0e0e0;font-size:14px;font-weight:600;margin:0 0 8px 0;">{interp}</p>
                            <p>{interp_desc}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Forensic Summary ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Forensic Summary</h4>
                        <div class="forensic-detail-row"><span class="dk">Evidence Type</span><span class="dv">Audio</span></div>
                        <div class="forensic-detail-row"><span class="dk">Analysis</span><span class="dv">AI-based speech authenticity assessment</span></div>
                        <div class="forensic-detail-row"><span class="dk">Model</span><span class="dv">{AUDIO_MODEL_NAME.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Device</span><span class="dv">{device.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Status</span><span class="dv" style="color:#28a745;">Completed</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    store_prediction(
                        file_name=uploaded_audio.name,
                        media_type="audio",
                        prediction=label,
                        confidence=confidence,
                        fake_prob=fake_prob,
                        inference_time=elapsed,
                    )

                elif _SIMPLE_AVAIL:
                    with st.spinner("Analyzing audio..."):
                        import librosa
                        audio_model, audio_device = load_audio_detector()
                        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
                        prob = _pf(y, audio_model, audio_device)
                    elapsed = time.time() - t0
                    label = "FAKE" if prob >= threshold else "REAL"
                    fake_prob = float(prob)
                    real_prob = 1.0 - fake_prob
                    confidence = abs(fake_prob - 0.5) * 2
                    verdict_class = "fake" if label == "FAKE" else "real"
                    verdict_icon = "&#10006;" if label == "FAKE" else "&#10004;"

                    with col_result:
                        st.markdown(f"""
                        <div style="margin-top:48px;">
                            <p style="color:#5a6b7e;font-size:12px;margin:0 0 16px 0;">Authenticity Assessment</p>
                            <div class="verdict-badge-lg {verdict_class}">
                                {verdict_icon}&nbsp; {label}
                            </div>
                            <div style="margin-top:20px;">
                                <div class="prob-row">
                                    <span class="label">Fake</span>
                                    <div class="prob-bar-track">
                                        <div class="prob-bar-fill fake" style="width:{fake_prob * 100}%;"></div>
                                    </div>
                                    <span class="value">{fake_prob:.1%}</span>
                                </div>
                                <div class="prob-row">
                                    <span class="label">Real</span>
                                    <div class="prob-bar-track">
                                        <div class="prob-bar-fill real" style="width:{real_prob * 100}%;"></div>
                                    </div>
                                    <span class="value">{real_prob:.1%}</span>
                                </div>
                            </div>
                            <div style="margin-top:16px;padding-top:14px;border-top:1px solid #1e2e3e;">
                                <div class="forensic-detail-row"><span class="dk">Confidence</span><span class="dv">{confidence:.1%}</span></div>
                                <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ---- Model Confidence ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Model Confidence</h4>
                        <div class="prob-row">
                            <span class="label">Real</span>
                            <div class="prob-bar-track">
                                <div class="prob-bar-fill real" style="width:{real_prob * 100}%;"></div>
                            </div>
                            <span class="value">{real_prob:.1%}</span>
                        </div>
                        <div class="prob-row">
                            <span class="label">Fake</span>
                            <div class="prob-bar-track">
                                <div class="prob-bar-fill fake" style="width:{fake_prob * 100}%;"></div>
                            </div>
                            <span class="value">{fake_prob:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Audio Forensic Analysis ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Audio Forensic Analysis</h4>
                        <div class="forensic-detail-row"><span class="dk">Analysis Status</span><span class="dv" style="color:#28a745;">Completed</span></div>
                        <div class="forensic-detail-row"><span class="dk">Evidence Type</span><span class="dv">Audio</span></div>
                        <div class="forensic-detail-row"><span class="dk">Model</span><span class="dv">{AUDIO_MODEL_NAME.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Device</span><span class="dv">{device.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Decision Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Inference Time</span><span class="dv">{elapsed:.2f}s</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Audio Forensic Timeline (not available in simple mode) ----
                    st.markdown("""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Audio Forensic Timeline</h4>
                        <div class="audio-interpretation">
                            <p>Segment-level analysis will appear when temporal predictions are available.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Assessment Interpretation ----
                    if label == "FAKE":
                        interp = "Potentially Synthetic Audio"
                        interp_desc = "The model detected patterns associated with synthetic or manipulated speech."
                    else:
                        interp = "Likely Authentic Audio"
                        interp_desc = "The model did not detect strong evidence of synthetic speech under the current analysis settings."

                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Assessment Interpretation</h4>
                        <div class="audio-interpretation">
                            <p style="color:#e0e0e0;font-size:14px;font-weight:600;margin:0 0 8px 0;">{interp}</p>
                            <p>{interp_desc}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Forensic Summary ----
                    st.markdown(f"""
                    <div class="analysis-card" style="margin-top:16px;">
                        <h4>Forensic Summary</h4>
                        <div class="forensic-detail-row"><span class="dk">Evidence Type</span><span class="dv">Audio</span></div>
                        <div class="forensic-detail-row"><span class="dk">Analysis</span><span class="dv">AI-based speech authenticity assessment</span></div>
                        <div class="forensic-detail-row"><span class="dk">Model</span><span class="dv">{AUDIO_MODEL_NAME.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Device</span><span class="dv">{device.upper()}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        <div class="forensic-detail-row"><span class="dk">Status</span><span class="dv" style="color:#28a745;">Completed</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    store_prediction(
                        file_name=uploaded_audio.name,
                        media_type="audio",
                        prediction=label,
                        confidence=confidence,
                        fake_prob=fake_prob,
                        inference_time=elapsed,
                    )

            except Exception as e:
                st.error(f"Analysis failed: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

