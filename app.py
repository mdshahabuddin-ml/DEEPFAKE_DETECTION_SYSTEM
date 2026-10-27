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
    /* Global — Deepfake Detector dark futuristic theme (reference match) */
    .stApp { background-color: #05080f; color: #dbe4f0; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
    section[data-testid="stSidebar"] { background-color: #0a0f1d; border-right: 1px solid #141f33; }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    section[data-testid="stSidebar"] .stRadio label { color: #c4cfdf !important; font-size: 13.5px !important; padding: 0 !important; }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 2px; }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background: transparent; border: 1px solid transparent; border-radius: 10px;
        padding: 9px 12px !important; margin: 1px 0 !important; transition: all .18s ease;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: rgba(47,123,255,.08); border-color: rgba(47,123,255,.18);
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, #1d5cff 0%, #1a3a9e 100%) !important;
        border-color: rgba(80,140,255,.5) !important;
        box-shadow: 0 4px 16px rgba(29,92,255,.35);
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] p {
        color: #fff !important; font-weight: 700 !important;
    }
    /* nav icons via nth-child */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1) p::before { content: "⌂  "; font-weight: 700; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(2) p::before { content: "◫  "; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(3) p::before { content: "▶  "; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(4) p::before { content: "♫  "; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(5) p::before { content: "◉  "; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(6) p::before { content: "((•))  "; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(7) p::before { content: "⊕  "; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(8) p::before { content: "◷  "; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(9) p::before { content: "▂▄  "; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(10) p::before { content: "ⓘ  "; }

    /* Sidebar header — shield logo */
    .sidebar-header { padding: 6px 4px 14px 4px; border-bottom: 1px solid #16223a; margin-bottom: 14px; }
    .sidebar-brand { display: flex; align-items: center; gap: 10px; }
    .sidebar-brand .shield {
        width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0;
        background: linear-gradient(135deg, #12305e, #0a1a35);
        border: 1.5px solid #2f7bff; display: flex; align-items: center; justify-content: center;
        box-shadow: 0 0 14px rgba(47,123,255,.4);
    }
    .sidebar-brand h2 { color: #fff; margin: 0; font-size: 17px; font-weight: 800; letter-spacing: -.2px; }
    .sidebar-brand p { color: #5a6b82; margin: 2px 0 0 0; font-size: 11px; }

    /* Sidebar threshold slider — red accent like reference */
    section[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] div[role="slider"] { background-color: #ef4444 !important; border-color: #ef4444 !important; }
    section[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] div > div > div { background: linear-gradient(90deg, #ef4444, #3a2330) !important; }
    .sidebar-threshold-label { color: #8b9bb0; font-size: 10.5px; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin: 0 0 2px 0; }
    .sidebar-threshold-val { color: #ef4444; font-size: 12px; font-weight: 800; text-align: center; margin: 0 0 4px 0; }
    .sidebar-threshold-hint { color: #6b7d95; font-size: 11.5px; line-height: 1.5; margin: 6px 0 0 0; }
    .sidebar-device { color: #6b7d95; font-size: 12px; margin: 10px 0 0 0; }
    .sidebar-device code { color: #22c55e; background: transparent; padding: 0; font-size: 12px; font-weight: 700; }
    .sidebar-device .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #22c55e; margin-right: 6px; box-shadow: 0 0 8px rgba(34,197,94,.8); }

    /* Sidebar bottom buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: #0f1729 !important; color: #c4cfdf !important;
        border: 1px solid #1e2f4a !important; border-radius: 10px !important;
        font-size: 12.5px !important; font-weight: 600 !important; padding: 9px 12px !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover { border-color: #2f7bff !important; color: #fff !important; }

    /* Nav items (legacy) */
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

    /* Image Detection - Compact Forensics UI */
    .img-hdr {
        background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0f1923 100%);
        border: 1px solid #2a3a4e; border-radius: 12px; padding: 18px 24px;
        margin-bottom: 20px; position: relative; overflow: hidden;
    }
    .img-hdr::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    .img-badge {
        display: inline-flex; align-items: center; gap: 5px;
        background: rgba(102,126,234,0.12); color: #667eea;
        padding: 3px 10px; border-radius: 12px; font-size: 9px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px;
        border: 1px solid rgba(102,126,234,0.2);
    }
    .img-title { color: #e8edf3; margin: 0 0 4px 0; font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }
    .img-sub { color: #7a8ba0; margin: 0; font-size: 12px; line-height: 1.4; }

    .img-upload-card {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 12px;
        padding: 18px 22px; margin-bottom: 16px;
    }
    .img-upload-card .img-upload-head {
        display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
    }
    .img-upload-card .img-upload-icon {
        width: 36px; height: 36px; border-radius: 8px; flex-shrink: 0;
        background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.2);
        display: flex; align-items: center; justify-content: center;
    }
    .img-upload-card h3 { color: #e0e0e0; margin: 0 0 2px 0; font-size: 14px; font-weight: 600; }
    .img-upload-card .img-upload-sub { color: #5a6b7e; font-size: 11px; margin: 0; }
    .img-formats {
        display: flex; gap: 16px; margin-top: 6px; font-size: 11px; color: #4a5a6e;
    }

    .forensics-bar {
        display: flex; align-items: center; justify-content: center; gap: 0;
        padding: 8px 0; margin: 0 0 16px 0;
    }
    .forensics-bar .fb-step {
        display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 0 12px;
    }
    .forensics-bar .fb-icon {
        width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 11px; border: 1.5px solid #2a3a4e; background: #141c26; color: #4a5a6e;
    }
    .forensics-bar .fb-icon.active { border-color: #667eea; background: rgba(102,126,234,0.12); color: #667eea; }
    .forensics-bar .fb-icon.done { border-color: #28a745; background: rgba(40,167,69,0.12); color: #28a745; }
    .forensics-bar .fb-label {
        font-size: 8px; text-transform: uppercase; letter-spacing: 0.8px; color: #4a5a6e; font-weight: 600;
    }
    .forensics-bar .fb-label.active { color: #667eea; }
    .forensics-bar .fb-label.done { color: #28a745; }
    .forensics-bar .fb-arrow { color: #2a3a4e; font-size: 11px; margin: 0 1px; padding-bottom: 14px; }

    .img-empty {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 12px;
        padding: 24px 20px; text-align: center; margin-bottom: 16px;
    }
    .img-empty .img-empty-icon { font-size: 28px; margin-bottom: 8px; opacity: 0.35; }
    .img-empty h3 { color: #e0e0e0; margin: 0 0 4px 0; font-size: 14px; font-weight: 600; }
    .img-empty p { color: #5a6b7e; margin: 0; font-size: 12px; line-height: 1.5; }

    .img-card {
        background: #141c26; border: 1px solid #1e2e3e; border-radius: 12px; padding: 16px 18px;
    }
    .img-card-title {
        color: #7a8ba0; font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px;
        margin: 0 0 12px 0; font-weight: 600; padding-bottom: 8px; border-bottom: 1px solid #1e2e3e;
    }

    .img-verdict {
        display: inline-flex; align-items: center; gap: 6px; padding: 8px 18px;
        border-radius: 8px; font-size: 16px; font-weight: 700; letter-spacing: 1px;
    }
    .img-verdict.real { background: rgba(40,167,69,0.1); color: #28a745; border: 1px solid rgba(40,167,69,0.25); }
    .img-verdict.fake { background: rgba(220,53,69,0.1); color: #dc3545; border: 1px solid rgba(220,53,69,0.25); }

    .img-pbar { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
    .img-pbar .lbl { color: #7a8ba0; font-size: 11px; font-weight: 600; width: 36px; text-align: right; }
    .img-pbar .track { flex: 1; height: 6px; background: #1e2e3e; border-radius: 3px; overflow: hidden; }
    .img-pbar .fill { height: 100%; border-radius: 3px; }
    .img-pbar .fill.real { background: linear-gradient(90deg, #1a7a34, #28a745); }
    .img-pbar .fill.fake { background: linear-gradient(90deg, #c0392b, #dc3545); }
    .img-pbar .val { color: #e0e0e0; font-size: 12px; font-weight: 600; width: 44px; }

    .img-detail { display: flex; justify-content: space-between; padding: 5px 0; font-size: 12px; }
    .img-detail .dk { color: #5a6b7e; }
    .img-detail .dv { color: #c0c8d4; font-weight: 500; }

    .img-bar {
        display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 14px;
        background: #0f1620; border: 1px solid #1a2636; border-radius: 8px; margin-top: 12px;
    }
    .img-bar-chip {
        font-size: 10px; color: #5a6b7e; display: flex; align-items: center; gap: 4px;
    }
    .img-bar-chip strong { color: #7a8ba0; font-weight: 600; }

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

    /* ============ HOME — reference match ============ */
    .home-topbar { display: flex; justify-content: flex-end; align-items: center; gap: 10px; margin: 0 0 10px 0; }
    .home-topbar .deploy { color: #c4cfdf; font-size: 13px; font-weight: 600; }
    .home-topbar .dots { color: #8b9bb0; font-size: 16px; font-weight: 800; letter-spacing: 2px; }

    .home-hero {
        background:
            radial-gradient(800px 300px at 15% 0%, rgba(47,123,255,.18), transparent 60%),
            radial-gradient(700px 320px at 85% 30%, rgba(124,58,237,.16), transparent 60%),
            linear-gradient(120deg, #0b1230 0%, #0d1533 45%, #080d20 100%);
        border: 1px solid #1c3a5e; border-radius: 14px;
        padding: 0; margin-bottom: 16px; overflow: hidden; position: relative;
    }
    .home-hero-inner { display: grid; grid-template-columns: 1.15fr .85fr; gap: 0; min-height: 340px; }
    .home-hero-left { padding: 30px 28px 26px 30px; }
    .home-hero-badge {
        display: inline-block; border: 1px solid #2a4a7a; color: #9db8d9;
        background: rgba(47,123,255,.08); padding: 5px 14px; border-radius: 20px;
        font-size: 10.5px; font-weight: 700; letter-spacing: 1.6px; margin-bottom: 14px;
    }
    .home-hero h1 { color: #fff; margin: 0 0 12px 0; font-size: 40px; line-height: 1.05; font-weight: 800; letter-spacing: -.8px; }
    .home-hero h1 .grad {
        background: linear-gradient(90deg, #3b9dff 0%, #5b7cff 45%, #7c5cff 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .home-hero .sub { color: #9fb0c7; margin: 0 0 20px 0; font-size: 14.5px; line-height: 1.6; max-width: 560px; }
    .home-cta-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }
    .st-key-home_analyze button {
        background: linear-gradient(135deg, #1e90ff 0%, #1a5cff 100%) !important;
        color: #fff !important; border: 1px solid rgba(120,180,255,.6) !important;
        border-radius: 10px !important; font-weight: 800 !important; font-size: 14px !important;
        padding: 12px 22px !important; box-shadow: 0 0 22px rgba(30,144,255,.55), inset 0 0 12px rgba(255,255,255,.12) !important;
    }
    .st-key-home_live button {
        background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%) !important;
        color: #fff !important; border: 1px solid rgba(180,140,255,.5) !important;
        border-radius: 10px !important; font-weight: 800 !important; font-size: 14px !important;
        padding: 12px 22px !important; box-shadow: 0 0 22px rgba(124,58,237,.45) !important;
    }
    .home-trust { display: flex; gap: 18px; flex-wrap: wrap; }
    .home-trust .t { display: flex; align-items: center; gap: 8px; color: #a9bad1; font-size: 12.5px; font-weight: 600; }
    .home-trust .t .ic {
        width: 26px; height: 26px; border-radius: 7px; display: flex; align-items: center; justify-content: center;
        background: rgba(47,123,255,.12); border: 1px solid rgba(47,123,255,.25); font-size: 13px;
    }
    .home-hero-right { position: relative; min-height: 340px; overflow: hidden;
        background: #070c18 url("https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=800&q=80&auto=format&fit=crop") center/cover no-repeat; }
    .home-hero-right::before { content:""; position:absolute; inset:0;
        background: linear-gradient(90deg, #0b1230 0%, rgba(11,18,48,.2) 22%, rgba(5,8,15,.15) 55%, rgba(5,8,15,.75) 100%); }
    .home-hero-right::after { content:""; position:absolute; inset:0;
        background-image: linear-gradient(rgba(47,123,255,.09) 1px, transparent 1px), linear-gradient(90deg, rgba(47,123,255,.09) 1px, transparent 1px);
        background-size: 26px 26px; pointer-events:none; }
    .home-scanline { position:absolute; top:0; bottom:0; left:50%; width:2px;
        background: linear-gradient(180deg, transparent, #3b9dff 20%, #3b9dff 80%, transparent);
        box-shadow: 0 0 18px 2px rgba(59,157,255,.8); z-index:3; }
    .home-badge-real { position:absolute; left:12%; top:58%; z-index:4;
        background: rgba(6,40,20,.9); color:#22c55e; border:1px solid rgba(34,197,94,.7);
        font-size:12px; font-weight:800; letter-spacing:1px; padding:6px 14px; border-radius:7px;
        box-shadow:0 0 16px rgba(34,197,94,.5); }
    .home-badge-fake { position:absolute; right:32%; top:58%; z-index:4;
        background: rgba(45,8,12,.9); color:#ff6b6b; border:1px solid rgba(239,68,68,.7);
        font-size:12px; font-weight:800; letter-spacing:1px; padding:6px 14px; border-radius:7px;
        box-shadow:0 0 16px rgba(239,68,68,.55); }
    .home-side-pills { position:absolute; right:10px; top:16px; bottom:16px; z-index:4;
        display:flex; flex-direction:column; gap:10px; justify-content:center; }
    .home-side-pills .pill {
        display:flex; align-items:center; gap:8px; background:rgba(10,18,35,.85);
        border:1px solid #254a7a; color:#bcd0ea; font-size:11.5px; font-weight:800; letter-spacing:.6px;
        padding:10px 14px; border-radius:9px; min-width:118px; backdrop-filter: blur(6px);
    }
    .home-status-grid { display:grid; grid-template-columns: repeat(5, 1fr); gap:12px; margin: 0 0 18px 0; }
    .home-status-card { background:#0d1526; border:1px solid #1a2b45; border-radius:12px; padding:14px;
        display:flex; gap:12px; align-items:center; }
    .home-status-card .sicon { width:46px; height:46px; border-radius:10px; flex-shrink:0;
        display:flex; align-items:center; justify-content:center; font-size:20px; }
    .home-status-card .lbl { color:#8b9bb0; font-size:11.5px; font-weight:600; margin:0; }
    .home-status-card .val { color:#fff; font-size:17px; font-weight:800; margin:2px 0; }
    .home-status-card .val.green { color:#22c55e; }
    .home-status-card .hint { color:#5f7191; font-size:11px; margin:0; }
    .home-section-head { display:flex; justify-content:space-between; align-items:center; margin: 6px 0 12px 0; }
    .home-section-head h2 { color:#fff; font-size:20px; font-weight:800; margin:0; letter-spacing:-.3px; }
    .home-section-head a { color:#2f7bff; font-size:12.5px; font-weight:700; text-decoration:none; }
    .home-cap-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin-bottom:18px; }
    .home-cap { background:#0d1526; border:1px solid #1a2b45; border-radius:12px; padding:16px;
        display:flex; gap:12px; align-items:center; transition: all .18s ease; }
    .home-cap:hover { border-color:#2f7bff; transform: translateY(-1px); box-shadow:0 8px 24px rgba(0,0,0,.35); }
    .home-cap .cicon { width:48px; height:48px; border-radius:10px; flex-shrink:0;
        display:flex; align-items:center; justify-content:center; font-size:22px; }
    .home-cap h4 { color:#fff; margin:0 0 4px 0; font-size:14px; font-weight:700; }
    .home-cap p { color:#7c8da6; margin:0; font-size:12px; line-height:1.5; }
    .home-cap .arrow { margin-left:auto; width:32px; height:32px; border-radius:50%; flex-shrink:0;
        background:#182845; border:1px solid #24406a; color:#9db8d9; display:flex; align-items:center; justify-content:center;
        font-size:15px; font-weight:700; }
    .home-how { background:#0a1120; border:1px solid #16263f; border-radius:12px; padding:18px 18px 16px 18px; margin-bottom:10px; }
    .home-how h2 { color:#fff; font-size:20px; font-weight:800; margin:0 0 14px 0; }
    .home-steps { display:grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr; gap:8px; align-items:stretch; }
    .home-step { background:#0f1a2e; border:1px solid #1d3454; border-radius:10px; padding:12px;
        display:flex; gap:10px; align-items:flex-start; }
    .home-step.green { border-color: rgba(34,197,94,.4); }
    .home-step .num { width:34px; height:34px; border-radius:50%; flex-shrink:0;
        background:#2f7bff; color:#fff; font-size:12.5px; font-weight:800;
        display:flex; align-items:center; justify-content:center; }
    .home-step.green .num { background:#22c55e; }
    .home-step .sic { width:34px; height:34px; border-radius:8px; flex-shrink:0;
        background:rgba(47,123,255,.12); border:1px solid rgba(47,123,255,.25);
        display:flex; align-items:center; justify-content:center; font-size:15px; }
    .home-step h5 { color:#e6edf7; margin:0 0 3px 0; font-size:12.5px; font-weight:700; }
    .home-step p { color:#6f829e; margin:0; font-size:11.5px; line-height:1.45; }
    .home-steps .arr { align-self:center; color:#2f7bff; font-size:16px; font-weight:800; }
    @media (max-width: 1100px) {
        .home-hero-inner { grid-template-columns: 1fr; }
        .home-hero-right { min-height: 280px; }
        .home-hero h1 { font-size: 32px; }
        .home-status-grid { grid-template-columns: repeat(3, 1fr); }
        .home-cap-grid { grid-template-columns: repeat(2, 1fr); }
        .home-steps { grid-template-columns: 1fr 1fr; }
        .home-steps .arr { display:none; }
    }
    @media (max-width: 640px) {
        .home-hero-left { padding: 20px 18px; }
        .home-hero h1 { font-size: 26px; }
        .home-status-grid { grid-template-columns: 1fr 1fr; }
        .home-cap-grid { grid-template-columns: 1fr; }
        .home-steps { grid-template-columns: 1fr; }
    }

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
        <div class="sidebar-brand">
            <div class="shield">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#5ea0ff" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><circle cx="12" cy="10" r="2.2" fill="#5ea0ff" stroke="none"/><path d="M7.5 10h2M14.5 10h2" stroke="#5ea0ff" stroke-width="1.6"/></svg>
            </div>
            <div>
                <h2>Deepfake Detector</h2>
                <p>AI Security Suite</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation via radio — options unchanged, key added for Home CTA navigation
    NAV_OPTIONS = ["Home", "Image Detection", "Video Detection", "Audio Detection",
         "Webcam Detection", "Live Detection", "Grad-CAM", "Prediction History", "Dashboard", "About"]
    if "nav" not in st.session_state:
        st.session_state.nav = "Home"
    if st.session_state.get("_pending_nav"):
        st.session_state.nav = st.session_state.pop("_pending_nav")
    try:
        default_idx = NAV_OPTIONS.index(st.session_state.nav)
    except ValueError:
        default_idx = 0
    page = st.radio(
        "Navigation",
        NAV_OPTIONS,
        index=default_idx,
        label_visibility="collapsed",
        key="nav",
    )

    st.markdown('<p class="sidebar-threshold-label" style="margin-top:14px;">Decision Threshold</p>', unsafe_allow_html=True)

    # Decision threshold — same variable name/logic, red accent via CSS
    threshold = st.slider("Decision threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
                          label_visibility="collapsed")
    st.markdown(f'<p class="sidebar-threshold-val">{threshold:.2f}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sidebar-threshold-hint">'
        f'Verdict fires at <span style="color:#ef4444;font-weight:800;">{threshold:.2f}</span> '
        f'&mdash; lower values flag more samples as fake</p>',
        unsafe_allow_html=True,
    )

    # Device and model info — same data, reference styling
    st.markdown(
        f'<p class="sidebar-device">'
        f'<span class="dot"></span>'
        f'Device:&nbsp; <code>{device}</code> &nbsp;|&nbsp; Model:&nbsp; <code>{MODEL_NAME}</code></p>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Logout / Change password placeholders — logic unchanged
    if st.button("⎋ &nbsp; Log out", width="stretch"):
        st.info("Logout functionality requires authentication setup.")
    if st.button("⚿ &nbsp; Change my password", width="stretch"):
        st.info("Password change requires authentication setup.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

# --- HOME — redesigned to match reference, logic unchanged ---
if page == "Home":
    st.markdown("""
    <div class="home-topbar"><span class="deploy">Deploy</span><span class="dots">⋮</span></div>
    <div class="home-hero">
      <div class="home-hero-inner">
        <div class="home-hero-left">
          <div class="home-hero-badge">AI SECURITY &nbsp;•&nbsp; DIGITAL FORENSICS</div>
          <h1>AI-Powered <span class="grad">Deepfake Detection</span></h1>
          <p class="sub">Detect whether an image, video, audio, or live webcam stream is authentic or manipulated using deep learning and explainable AI.</p>
        </div>
        <div class="home-hero-right">
          <div class="home-scanline"></div>
          <div class="home-badge-real">REAL</div>
          <div class="home-badge-fake">FAKE</div>
          <div class="home-side-pills">
            <div class="pill"><span class="ic">🖼</span> IMAGE</div>
            <div class="pill"><span class="ic">▶</span> VIDEO</div>
            <div class="pill"><span class="ic">♫</span> AUDIO</div>
            <div class="pill"><span class="ic">◉</span> WEBCAM</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero CTAs — real Streamlit buttons so navigation keeps working (styled via .st-key-*)
    cta1, cta2, _sp = st.columns([1, 1, 1.4])
    with cta1:
        if st.button("🖼 &nbsp; Analyze Image &nbsp; →", key="home_analyze", width="stretch"):
            st.session_state["_pending_nav"] = "Image Detection"
            st.rerun()
    with cta2:
        if st.button("◉ &nbsp; Start Live Detection &nbsp; →", key="home_live", width="stretch"):
            st.session_state["_pending_nav"] = "Live Detection"
            st.rerun()

    st.markdown("""
    <div class="home-trust" style="margin: 4px 0 16px 0;">
      <div class="t"><span class="ic">🛡</span> Accurate Analysis</div>
      <div class="t"><span class="ic">⚡</span> Explainable AI (Grad-CAM)</div>
      <div class="t"><span class="ic">🔒</span> Privacy Focused</div>
      <div class="t"><span class="ic">📊</span> Multi-Modal Support</div>
    </div>
    """, unsafe_allow_html=True)

    # Status cards row — values wired to live device / threshold / model constants
    st.markdown(f"""
    <div class="home-status-grid">
      <div class="home-status-card">
        <div class="sicon" style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);color:#22c55e;">◉</div>
        <div><p class="lbl">System Status</p><p class="val green">Online</p><p class="hint">All services running</p></div>
      </div>
      <div class="home-status-card">
        <div class="sicon" style="background:rgba(47,123,255,.12);border:1px solid rgba(47,123,255,.3);color:#5ea0ff;">▣</div>
        <div><p class="lbl">Model</p><p class="val">Xception</p><p class="hint">Deep learning model</p></div>
      </div>
      <div class="home-status-card">
        <div class="sicon" style="background:rgba(124,58,237,.14);border:1px solid rgba(124,58,237,.35);color:#a78bfa;">🖼</div>
        <div><p class="lbl">Input Resolution</p><p class="val">299 × 299 px</p><p class="hint">Optimized for accuracy</p></div>
      </div>
      <div class="home-status-card">
        <div class="sicon" style="background:rgba(47,123,255,.12);border:1px solid rgba(47,123,255,.3);color:#5ea0ff;">💻</div>
        <div><p class="lbl">Inference Device</p><p class="val">{device.upper()}</p><p class="hint">Auto GPU/CPU support</p></div>
      </div>
      <div class="home-status-card">
        <div class="sicon" style="background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);color:#ff6b6b;">⇄</div>
        <div><p class="lbl">Decision Threshold</p><p class="val">{threshold:.2f}</p><p class="hint">Adjustable sensitivity</p></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-section-head"><h2>Core Detection Capabilities</h2><a href="#" onclick="return false;">Explore All Features →</a></div>
    <div class="home-cap-grid">
      <div class="home-cap">
        <div class="cicon" style="background:rgba(34,197,94,.14);border:1px solid rgba(34,197,94,.3);color:#22c55e;">🖼</div>
        <div><h4>Image Detection</h4><p>Analyze images for potential visual manipulation.</p></div>
        <div class="arrow">→</div>
      </div>
      <div class="home-cap">
        <div class="cicon" style="background:rgba(124,58,237,.16);border:1px solid rgba(124,58,237,.35);color:#a78bfa;">▶</div>
        <div><h4>Video Detection</h4><p>Perform frame-level and temporal deepfake analysis.</p></div>
        <div class="arrow">→</div>
      </div>
      <div class="home-cap">
        <div class="cicon" style="background:rgba(249,115,22,.14);border:1px solid rgba(249,115,22,.3);color:#fb923c;">♫</div>
        <div><h4>Audio Detection</h4><p>Analyze speech and audio for potential synthetic content.</p></div>
        <div class="arrow">→</div>
      </div>
      <div class="home-cap">
        <div class="cicon" style="background:rgba(47,123,255,.14);border:1px solid rgba(47,123,255,.3);color:#5ea0ff;">◉</div>
        <div><h4>Live Webcam</h4><p>Real-time camera-based deepfake analysis.</p></div>
        <div class="arrow">→</div>
      </div>
      <div class="home-cap">
        <div class="cicon" style="background:rgba(239,68,68,.13);border:1px solid rgba(239,68,68,.3);color:#ff6b6b;">⊕</div>
        <div><h4>Grad-CAM</h4><p>Visualize image regions influencing model predictions.</p></div>
        <div class="arrow">→</div>
      </div>
      <div class="home-cap">
        <div class="cicon" style="background:rgba(234,179,8,.14);border:1px solid rgba(234,179,8,.3);color:#facc15;">📊</div>
        <div><h4>Analytics</h4><p>View prediction history and detection statistics.</p></div>
        <div class="arrow">→</div>
      </div>
    </div>
    <div class="home-how">
      <h2>How It Works</h2>
      <div class="home-steps">
        <div class="home-step">
          <div class="num">01</div><div class="sic">⬆</div>
          <div><h5>Upload Media</h5><p>Provide an image, video, audio file, or live stream.</p></div>
        </div>
        <div class="arr">→</div>
        <div class="home-step">
          <div class="num">02</div><div class="sic">⚙</div>
          <div><h5>AI Analysis</h5><p>Deep learning model analyzes the content.</p></div>
        </div>
        <div class="arr">→</div>
        <div class="home-step">
          <div class="num">03</div><div class="sic">🔍</div>
          <div><h5>Evidence &amp; Explainability</h5><p>Get confidence scores and Grad-CAM visualization.</p></div>
        </div>
        <div class="arr">→</div>
        <div class="home-step green">
          <div class="num">04</div><div class="sic" style="background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.3);">✔</div>
          <div><h5>Authenticity Assessment</h5><p>Determine if the media is real or fake.</p></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# --- IMAGE DETECTION ---
elif page == "Image Detection":
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

    # ---- Page Header ----
    st.markdown("""
    <div class="img-hdr">
        <div class="img-badge">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            IMAGE FORENSICS
        </div>
        <h1 class="img-title">Image Deepfake Detection</h1>
        <p class="img-sub">Analyze images for patterns associated with AI-generated or manipulated visual content.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Upload Card ----
    st.markdown("""
    <div class="img-upload-card">
        <div class="img-upload-head">
            <div class="img-upload-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="1.5">
                    <rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/>
                </svg>
            </div>
            <div>
                <h3>Upload Image Evidence</h3>
                <p class="img-upload-sub">Upload a JPG, JPEG, or PNG image for forensic analysis.</p>
            </div>
        </div>
        <div class="img-formats">
            <span>JPG &bull; JPEG &bull; PNG</span>
            <span>Maximum: 200 MB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img",
                                   label_visibility="collapsed")

    # ---- Compact Pipeline ----
    step_upload = "done" if uploaded_img else "active"
    step_analyze = "active" if uploaded_img else ""
    st.markdown(f"""
    <div class="forensics-bar">
        <div class="fb-step">
            <div class="fb-icon {step_upload}">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            </div>
            <span class="fb-label {step_upload}">Upload</span>
        </div>
        <div class="fb-arrow">&#8594;</div>
        <div class="fb-step">
            <div class="fb-icon {step_analyze}">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            </div>
            <span class="fb-label {step_analyze}">Analyze</span>
        </div>
        <div class="fb-arrow">&#8594;</div>
        <div class="fb-step">
            <div class="fb-icon">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            </div>
            <span class="fb-label">Result</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Empty State ----
    if uploaded_img is None:
        st.markdown("""
        <div class="img-empty">
            <div class="img-empty-icon">&#9670;</div>
            <h3>Ready for Analysis</h3>
            <p>Upload an image to begin AI-powered deepfake detection.</p>
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
                col_img, col_result = st.columns([1, 1], gap="small")

                with col_img:
                    st.markdown("""<div class="img-card"><div class="img-card-title">Image Evidence</div></div>""", unsafe_allow_html=True)
                    st.image(pil_img, use_container_width=True)
                    size_kb = file_size / 1024
                    size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{file_size/1024/1024:.1f} MB"
                    st.markdown(f"""
                    <div class="img-bar">
                        <span class="img-bar-chip"><strong>Name:</strong> {uploaded_img.name}</span>
                        <span class="img-bar-chip"><strong>Format:</strong> {uploaded_img.type}</span>
                        <span class="img-bar-chip"><strong>Resolution:</strong> {img_w}x{img_h}</span>
                        <span class="img-bar-chip"><strong>Size:</strong> {size_str}</span>
                    </div>
                    """, unsafe_allow_html=True)

                with col_result:
                    verdict_class = "fake" if label == "FAKE" else "real"
                    verdict_icon = "&#10006;" if label == "FAKE" else "&#10004;"
                    st.markdown(f"""
                    <div class="img-card">
                        <div class="img-card-title">Detection Assessment</div>
                        <div class="img-verdict {verdict_class}">{verdict_icon}&nbsp; {label}</div>
                        <div style="margin-top:14px;">
                            <div class="img-pbar">
                                <span class="lbl">Fake</span>
                                <div class="track"><div class="fill fake" style="width:{prob * 100}%;"></div></div>
                                <span class="val">{prob:.1%}</span>
                            </div>
                            <div class="img-pbar">
                                <span class="lbl">Real</span>
                                <div class="track"><div class="fill real" style="width:{real_prob * 100}%;"></div></div>
                                <span class="val">{real_prob:.1%}</span>
                            </div>
                        </div>
                        <div style="margin-top:12px;padding-top:10px;border-top:1px solid #1e2e3e;">
                            <div class="img-detail"><span class="dk">Confidence</span><span class="dv">{confidence:.1%}</span></div>
                            <div class="img-detail"><span class="dk">Threshold</span><span class="dv">{threshold:.2f}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ---- Forensic Details Bar ----
                if label == "FAKE":
                    assessment = '<span style="color:#dc3545;font-weight:600;">Potentially manipulated</span>'
                else:
                    assessment = '<span style="color:#28a745;font-weight:600;">Likely authentic</span>'

                st.markdown(f"""
                <div class="img-card" style="margin-top:12px;">
                    <div class="img-card-title">AI Detection Assessment</div>
                    <div class="img-bar">
                        <span class="img-bar-chip"><strong>Model:</strong> {MODEL_NAME.upper()}</span>
                        <span class="img-bar-chip"><strong>Device:</strong> {device.upper()}</span>
                        <span class="img-bar-chip"><strong>Input:</strong> Image</span>
                        <span class="img-bar-chip"><strong>Threshold:</strong> {threshold:.2f}</span>
                        <span class="img-bar-chip"><strong>Result:</strong> {assessment}</span>
                        <span class="img-bar-chip"><strong>Time:</strong> {elapsed:.3f}s</span>
                    </div>
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
                _video_prob_str = f"{video_prob:.3f}" if video_prob is not None else "n/a"
                _audio_prob_str = f"{audio_prob:.3f}" if audio_prob is not None else "Unavailable"
                st.markdown(f"""
                <div class="analysis-card" style="margin-top:16px;">
                    <h4>Multimodal Evidence</h4>
                    <div class="forensic-detail-row"><span class="dk">Visual Evidence</span><span class="dv">{_video_prob_str}</span></div>
                    <div class="forensic-detail-row"><span class="dk">Audio Evidence</span><span class="dv">{_audio_prob_str}</span></div>
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
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
        .wc-hdr {
            background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0f1923 100%);
            border: 1px solid #2a3a4e; border-radius: 12px; padding: 18px 24px;
            margin-bottom: 20px; position: relative; overflow: hidden;
        }
        .wc-hdr::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
        }
        .wc-badge {
            display: inline-flex; align-items: center; gap: 5px;
            background: rgba(102,126,234,0.12); color: #667eea;
            padding: 3px 10px; border-radius: 12px; font-size: 9px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px;
            border: 1px solid rgba(102,126,234,0.2);
        }
        .wc-title { color: #e8edf3; margin: 0 0 4px 0; font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }
        .wc-sub { color: #7a8ba0; margin: 0; font-size: 12px; line-height: 1.4; }
        .wc-card {
            background: rgba(20,28,38,0.85); backdrop-filter: blur(12px);
            border: 1px solid #1e2e3e; border-radius: 12px; padding: 16px 18px;
        }
        .wc-card-title {
            color: #7a8ba0; font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px;
            margin: 0 0 12px 0; font-weight: 600; padding-bottom: 8px; border-bottom: 1px solid #1e2e3e;
        }
        .wc-live {
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(220,53,69,0.12); color: #dc3545;
            padding: 4px 12px; border-radius: 8px; font-size: 11px; font-weight: 700;
            letter-spacing: 1px; border: 1px solid rgba(220,53,69,0.25);
        }
        .wc-live .wc-dot {
            width: 7px; height: 7px; border-radius: 50%; background: #dc3545;
            animation: wc-pulse 1.4s ease-in-out infinite;
        }
        @keyframes wc-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        .wc-idle {
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(90,107,126,0.12); color: #5a6b7e;
            padding: 4px 12px; border-radius: 8px; font-size: 11px; font-weight: 700;
            letter-spacing: 1px; border: 1px solid rgba(90,107,126,0.2);
        }
        .wc-idle .wc-dot {
            width: 7px; height: 7px; border-radius: 50%; background: #5a6b7e;
        }
        .wc-metric {
            display: flex; justify-content: space-between; align-items: center;
            padding: 6px 0; font-size: 12px;
        }
        .wc-metric + .wc-metric { border-top: 1px solid rgba(30,46,62,0.5); }
        .wc-mk { color: #5a6b7e; }
        .wc-mv { color: #c0c8d4; font-weight: 600; font-family: 'SF Mono', 'Consolas', monospace; }
        .wc-legend-item {
            display: flex; align-items: center; gap: 8px; padding: 5px 0;
        }
        .wc-legend-dot {
            width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
        }
        .wc-legend-label { color: #7a8ba0; font-size: 12px; }
        .wc-ctrl-btn {
            display: block; width: 100%; padding: 10px 0; border: none; border-radius: 8px;
            font-size: 12px; font-weight: 700; letter-spacing: 0.8px; cursor: pointer;
            text-transform: uppercase; text-align: center; transition: all 0.2s;
        }
        .wc-ctrl-start { background: rgba(40,167,69,0.15); color: #28a745; border: 1px solid rgba(40,167,69,0.3); }
        .wc-ctrl-start:hover { background: rgba(40,167,69,0.25); }
        .wc-ctrl-stop { background: rgba(220,53,69,0.15); color: #dc3545; border: 1px solid rgba(220,53,69,0.3); }
        .wc-ctrl-stop:hover { background: rgba(220,53,69,0.25); }
        .wc-ctrl-device { background: rgba(102,126,234,0.1); color: #667eea; border: 1px solid rgba(102,126,234,0.25); }
        .wc-ctrl-device:hover { background: rgba(102,126,234,0.2); }
        .wc-preview {
            background: #0a0e14; border: 1px solid #1e2e3e; border-radius: 12px;
            overflow: hidden; position: relative;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---- Header ----
    st.markdown("""
    <div class="wc-hdr">
        <div class="wc-badge">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
            LIVE FORENSICS
        </div>
        <h1 class="wc-title">Real-Time Webcam Detection</h1>
        <p class="wc-sub">Live AI-powered deepfake analysis from your camera.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        from collections import deque
        import av
        from streamlit_webrtc import webrtc_streamer, WebRtcMode

        prob_history = deque(maxlen=8)

        # ---- Main Layout: Preview + Sidebar ----
        col_preview, col_sidebar = st.columns([3, 1], gap="medium")

        with col_preview:
            # Camera Preview tile
            st.markdown("""
            <div style="margin-bottom:8px;">
                <div class="wc-card-title" style="border:none;padding:0;margin:0 0 6px 0;">Camera Preview</div>
                <div class="wc-live"><span class="wc-dot"></span>LIVE</div>
            </div>
            """, unsafe_allow_html=True)

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

        with col_sidebar:
            # Detection Status tile
            st.markdown(f"""
            <div class="wc-card" style="margin-bottom:12px;">
                <div class="wc-card-title">Detection Status</div>
                <div class="wc-metric">
                    <span class="wc-mk">Status</span>
                    <span class="wc-mv">Awaiting feed</span>
                </div>
                <div class="wc-metric">
                    <span class="wc-mk">Verdict</span>
                    <span class="wc-mv">--</span>
                </div>
                <div class="wc-metric">
                    <span class="wc-mk">Confidence</span>
                    <span class="wc-mv">--</span>
                </div>
                <div style="margin-top:10px;padding-top:8px;border-top:1px solid #1e2e3e;">
                    <div style="display:flex;gap:16px;">
                        <div style="text-align:center;flex:1;">
                            <div style="color:#5a6b7e;font-size:9px;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:2px;">Fake</div>
                            <div style="color:#dc3545;font-size:16px;font-weight:700;font-family:monospace;">--</div>
                        </div>
                        <div style="text-align:center;flex:1;">
                            <div style="color:#5a6b7e;font-size:9px;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:2px;">Real</div>
                            <div style="color:#28a745;font-size:16px;font-weight:700;font-family:monospace;">--</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Performance tile
            st.markdown(f"""
            <div class="wc-card" style="margin-bottom:12px;">
                <div class="wc-card-title">Performance</div>
                <div class="wc-metric">
                    <span class="wc-mk">FPS</span>
                    <span class="wc-mv">--</span>
                </div>
                <div class="wc-metric">
                    <span class="wc-mk">Latency</span>
                    <span class="wc-mv">--</span>
                </div>
                <div class="wc-metric">
                    <span class="wc-mk">Device</span>
                    <span class="wc-mv">{device.upper()}</span>
                </div>
                <div class="wc-metric">
                    <span class="wc-mk">Model</span>
                    <span class="wc-mv">{MODEL_NAME.upper()}</span>
                </div>
                <div class="wc-metric">
                    <span class="wc-mk">Threshold</span>
                    <span class="wc-mv">{threshold:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Detection Legend tile
            st.markdown("""
            <div class="wc-card" style="margin-bottom:12px;">
                <div class="wc-card-title">Detection Legend</div>
                <div class="wc-legend-item">
                    <div class="wc-legend-dot" style="background:#28a745;"></div>
                    <span class="wc-legend-label"><strong style="color:#28a745;">REAL</strong> &mdash; Authentic content</span>
                </div>
                <div class="wc-legend-item">
                    <div class="wc-legend-dot" style="background:#dc3545;"></div>
                    <span class="wc-legend-label"><strong style="color:#dc3545;">FAKE</strong> &mdash; Manipulated content</span>
                </div>
                <div class="wc-legend-item">
                    <div class="wc-legend-dot" style="background:#ffc107;"></div>
                    <span class="wc-legend-label"><strong style="color:#ffc107;">UNCERTAIN</strong> &mdash; Inconclusive</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Camera Controls tile
            st.markdown("""
            <div class="wc-card">
                <div class="wc-card-title">Camera Controls</div>
                <p style="color:#5a6b7e;font-size:11px;margin:0 0 10px 0;line-height:1.5;">
                    Use the START/STOP buttons on the video player above to control the camera feed.
                </p>
                <div style="display:flex;gap:8px;">
                    <div class="wc-ctrl-btn wc-ctrl-start" style="flex:1;">START</div>
                    <div class="wc-ctrl-btn wc-ctrl-stop" style="flex:1;">STOP</div>
                </div>
                <p style="color:#4a5a6e;font-size:10px;margin:10px 0 0 0;line-height:1.4;">
                    Green box = REAL &bull; Red box = FAKE<br>
                    Prediction smoothed over recent frames.
                </p>
            </div>
            """, unsafe_allow_html=True)

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


# --- LIVE DETECTION (Webcam + Audio) ---
elif page == "Live Detection":
    # Professional live webcam + audio detection (see app/live_detection_page.py).
    # Engine: src/live_detection.py — reuses realtime_webcam / audio_infer / fusion.
    from app.live_detection_page import render_live_detection_page
    render_live_detection_page(
        model, device, transform, face_cascade, threshold,
        store_prediction, model_name=MODEL_NAME,
    )

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
    st.markdown("""
    <style>
        /* Prediction History — enterprise dashboard treatment (this page only) */
        .ph-eyebrow {
            display: inline-flex; align-items: center; gap: 7px;
            font-size: 10px; font-weight: 700; letter-spacing: 1.6px; color: #667eea;
            background: rgba(102, 126, 234, 0.08); border: 1px solid rgba(102, 126, 234, 0.22);
            padding: 4px 12px; border-radius: 20px; margin-bottom: 10px;
        }
        .ph-title { color: #e8edf3; margin: 0 0 4px 0; font-size: 24px; font-weight: 700; letter-spacing: -0.3px; }
        .ph-sub { color: #7a8ba0; margin: 0 0 4px 0; font-size: 13px; }
        .ph-grid {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 18px 0 6px 0;
        }
        @media (max-width: 1100px) {
            .ph-grid { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 640px) {
            .ph-grid { grid-template-columns: 1fr; }
        }
        .ph-card {
            background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
            padding: 18px 20px; display: flex; align-items: center; gap: 14px;
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        }
        .ph-card:hover {
            transform: translateY(-2px); border-color: #3a4a63;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.10);
        }
        .ph-icon {
            width: 40px; height: 40px; border-radius: 10px; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
        }
        .ph-meta { min-width: 0; }
        .ph-label {
            color: #7a8ba0; font-size: 10px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 1.2px; margin: 0 0 3px 0;
        }
        .ph-value { color: #e8edf3; font-size: 24px; font-weight: 700; margin: 0; line-height: 1.1; }
        .ph-section {
            background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
            padding: 18px 20px 20px 20px; margin-top: 16px;
        }
        .ph-section-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
        .ph-section-title { display: flex; align-items: center; gap: 10px; color: #e8edf3; font-size: 15px; font-weight: 700; margin: 0; }
        .ph-count {
            font-size: 10px; font-weight: 700; letter-spacing: 1px; color: #8a99ad;
            background: #0f1620; border: 1px solid #1a2636;
            padding: 3px 10px; border-radius: 20px; white-space: nowrap;
        }
        .ph-legend { display: flex; gap: 14px; margin-top: 12px; font-size: 11px; color: #5a6b7e; }
        .ph-legend .lg { display: flex; align-items: center; gap: 6px; }
        .ph-legend .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    </style>
    <div>
        <span class="ph-eyebrow">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            FORENSIC RECORDS
        </span>
        <p class="ph-title">Prediction History</p>
        <p class="ph-sub">Review all past analyses stored in the local database</p>
    </div>
    """, unsafe_allow_html=True)

    db = get_db()
    records = db.recent(limit=200)

    if not records:
        st.info("No predictions yet. Run a detection to see results here.")
    else:
        # Stats
        stats = db.stats()
        st.markdown(f"""
        <div class="ph-grid">
            <div class="ph-card">
                <div class="ph-icon" style="background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.2);">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
                </div>
                <div class="ph-meta"><p class="ph-label">Total Analyses</p><p class="ph-value">{stats["total"]}</p></div>
            </div>
            <div class="ph-card">
                <div class="ph-icon" style="background:rgba(220,53,69,0.1);border:1px solid rgba(220,53,69,0.2);">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#dc3545" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                </div>
                <div class="ph-meta"><p class="ph-label">Fake Detected</p><p class="ph-value">{stats["fake"]}</p></div>
            </div>
            <div class="ph-card">
                <div class="ph-icon" style="background:rgba(40,167,69,0.1);border:1px solid rgba(40,167,69,0.2);">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#28a745" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                </div>
                <div class="ph-meta"><p class="ph-label">Authentic</p><p class="ph-value">{stats["real"]}</p></div>
            </div>
            <div class="ph-card">
                <div class="ph-icon" style="background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.2);">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                </div>
                <div class="ph-meta"><p class="ph-label">Avg Confidence</p><p class="ph-value">{stats['avg_confidence']:.2%}</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Table
        import pandas as pd
        df = pd.DataFrame(records)
        df = df[["id", "file_name", "media_type", "prediction", "confidence",
                  "fake_probability", "inference_time", "created_at"]]
        df.columns = ["ID", "File", "Type", "Prediction", "Confidence",
                       "Fake Prob", "Time (s)", "Timestamp"]

        # Badge styling for the prediction column
        def badge_prediction(val):
            base = ("font-weight: 700; border-radius: 6px; "
                    "padding: 2px 10px; text-align: center;")
            if val == "FAKE":
                return base + "color: #ff6b7a; background-color: rgba(220,53,69,0.12);"
            if val == "UNCERTAIN":
                return base + "color: #ffc107; background-color: rgba(255,193,7,0.12);"
            return base + "color: #3ddc74; background-color: rgba(40,167,69,0.12);"

        st.markdown(f"""
        <div class="ph-section">
            <div class="ph-section-head">
                <p class="ph-section-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
                    Analysis Records
                </p>
                <span class="ph-count">{len(records)} SHOWN</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            df.style.map(badge_prediction, subset=["Prediction"]),
            use_container_width=True,
            height=400,
        )
        st.markdown("""
        <div class="ph-legend">
            <span class="lg"><span class="dot" style="background:#dc3545;"></span>FAKE — manipulated</span>
            <span class="lg"><span class="dot" style="background:#28a745;"></span>REAL — authentic</span>
            <span class="lg"><span class="dot" style="background:#ffc107;"></span>UNCERTAIN — needs review</span>
        </div>
        """, unsafe_allow_html=True)

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
        conf_hist = df["confidence"].value_counts(bins=20).sort_index()
        st.bar_chart(conf_hist, use_container_width=True)
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

        /* Core Capabilities — professional SaaS cards (About page only) */
        .cap-pro-head { margin-bottom: 18px; }
        .cap-pro-eyebrow {
            display: inline-flex; align-items: center; gap: 7px;
            font-size: 10px; font-weight: 700; letter-spacing: 1.6px; color: #667eea;
            background: rgba(102, 126, 234, 0.08); border: 1px solid rgba(102, 126, 234, 0.22);
            padding: 4px 12px; border-radius: 20px; margin-bottom: 10px;
        }
        .cap-pro-status {
            display: flex; align-items: center; gap: 8px; margin-top: 10px;
            font-size: 11px; color: #5a6b7e; font-weight: 500;
        }
        .cap-pro-status .s-dot { width: 6px; height: 6px; border-radius: 50%; background: #28a745; flex-shrink: 0; }
        .cap-pro-status strong { color: #8a99ad; font-weight: 600; }
        .cap-pro-status code {
            font-family: 'SF Mono', 'Consolas', monospace; font-size: 10px; color: #7a8ba0;
            background: #0f1620; border: 1px solid #1a2636; border-radius: 5px; padding: 1px 6px;
        }
        .cap-pro-grid {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
        }
        @media (max-width: 1100px) {
            .cap-pro-grid { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 640px) {
            .cap-pro-grid { grid-template-columns: 1fr; }
        }
        .cap-pro-card {
            background: #141c26; border: 1px solid #1e2e3e; border-radius: 14px;
            padding: 22px; min-height: 248px; display: flex; flex-direction: column;
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        }
        .cap-pro-card:hover {
            transform: translateY(-3px); border-color: #3a4a63;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.12);
        }
        .cap-pro-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
        .cap-pro-icon {
            width: 40px; height: 40px; border-radius: 10px; flex-shrink: 0;
            background: rgba(102, 126, 234, 0.1); border: 1px solid rgba(102, 126, 234, 0.2);
            display: flex; align-items: center; justify-content: center;
            transition: border-color 0.25s ease, background 0.25s ease;
        }
        .cap-pro-card:hover .cap-pro-icon {
            border-color: rgba(102, 126, 234, 0.45); background: rgba(102, 126, 234, 0.14);
        }
        .cap-pro-index {
            font-family: 'SF Mono', 'Consolas', monospace; font-size: 11px; font-weight: 600;
            color: #3a4a5e; letter-spacing: 1px;
        }
        .cap-pro-cat {
            color: #7a8ba0; font-size: 10px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 1.2px; margin: 0 0 4px 0;
        }
        .cap-pro-card h4 { color: #e8edf3; margin: 0 0 8px 0; font-size: 15px; font-weight: 700; letter-spacing: -0.1px; }
        .cap-pro-card p { color: #7a8ba0; font-size: 12px; line-height: 1.65; margin: 0; flex-grow: 1; }
        .cap-pro-meta {
            display: flex; align-items: center; justify-content: space-between;
            margin-top: 16px; padding-top: 13px; border-top: 1px solid #1e2e3e;
            font-size: 11px;
        }
        .cap-pro-meta .m-label { display: flex; align-items: center; gap: 6px; color: #8a99ad; font-weight: 600; }
        .cap-pro-meta .m-dot { width: 6px; height: 6px; border-radius: 50%; background: #28a745; flex-shrink: 0; }
        .cap-pro-meta .m-arrow { color: #667eea; font-size: 13px; transition: transform 0.25s ease; }
        .cap-pro-card:hover .m-arrow { transform: translateX(3px); }
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
    <div class="cap-pro-head">
        <span class="cap-pro-eyebrow">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            PLATFORM CAPABILITIES
        </span>
        <p class="about-section-title">Core Capabilities</p>
        <p class="about-section-sub">Multimodal detection across image, video, and audio media.</p>
        <div class="cap-pro-status">
            <span class="s-dot"></span>
            <span><strong>All modules operational</strong> &nbsp;·&nbsp; forensic engines <code>Xception</code> <code>EfficientNet</code></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    caps_primary = [
        ("AI ANALYSIS", "Image Detection", "Analyze images for patterns associated with manipulated or synthetic visual content.", "Deep Learning", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>'),
        ("AI ANALYSIS", "Video Detection", "Analyze sampled video frames and temporal evidence for potential manipulation.", "Frame Analysis", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>'),
        ("AI ANALYSIS", "Audio Detection", "Analyze speech recordings for patterns associated with synthetic or manipulated audio.", "Audio Forensics", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>'),
        ("REAL-TIME AI", "Real-Time Webcam", "Perform real-time visual analysis using the webcam detection pipeline.", "Live Analysis", '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>'),
    ]

    cards_html = "".join(
        f"""
        <div class="cap-pro-card">
            <div class="cap-pro-top">
                <div class="cap-pro-icon">{icon}</div>
                <span class="cap-pro-index">{idx}</span>
            </div>
            <p class="cap-pro-cat">{cat}</p>
            <h4>{title}</h4>
            <p>{desc}</p>
            <div class="cap-pro-meta">
                <span class="m-label"><span class="m-dot"></span>{meta}</span>
                <span class="m-arrow">&#8594;</span>
            </div>
        </div>
        """
        for idx, (cat, title, desc, meta, icon) in
        zip(["01", "02", "03", "04"], caps_primary)
    )
    st.markdown(f'<div class="cap-pro-grid">{cards_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    caps = [
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

