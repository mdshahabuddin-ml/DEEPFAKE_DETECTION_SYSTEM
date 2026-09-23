"""
app/streamlit_app.py
AI-Powered Explainable Deepfake Detection System — Streamlit front-end.

A multi-page dashboard (sidebar navigation) covering image, video, and webcam
detection, Grad-CAM explanations, prediction history, and analytics.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import base64
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Make the project root importable when Streamlit executes this as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (
    CHECKPOINT_PATH, DEFAULT_THRESHOLD, DEVICE, IMG_SIZE, MODEL_NAME,
    VIDEO_MAX_FRAMES, confidence_from_prob, label_from_prob,
)
from src.auth import authenticate, get_auth_db
from src.database import get_db
from src.inference import ModelNotFoundError, get_detector
from src.gradcam import GradCAMExplainer
from app.predict import run_image, run_video, store_result
from app.webcam import webcam_page

st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="wide")


# ---------------------------------------------------------------------------
# Authentication (login gate)
# ---------------------------------------------------------------------------
def _get_logo_base64() -> str:
    """Load and base64-encode the logo so it can be embedded in the styled HTML card."""
    logo_path = _ROOT / "assets" / "logo.png"
    if not logo_path.exists():
        return ""
    with open(logo_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def login_page() -> None:
    """Render a styled login/signup page. Sets session state and reruns on success."""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        }
        .login-card {
            background: rgba(255, 255, 255, 0.97);
            border-radius: 16px;
            padding: 40px 36px 28px 36px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
            margin-top: 40px;
        }
        .login-logo {
            display: block;
            margin: 0 auto 12px auto;
            max-width: 90px;
            max-height: 90px;
            border-radius: 12px;
        }
        .login-title {
            text-align: center;
            font-size: 30px;
            font-weight: 800;
            color: #1a1a2e;
            margin-bottom: 2px;
        }
        .login-subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 22px;
        }
        div[data-testid="stForm"] {
            border: none;
            padding: 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 18px;
            font-weight: 600;
        }
        div.stButton > button, button[kind="primary"], button[kind="secondary"] {
            border-radius: 8px;
            font-weight: 600;
            height: 42px;
        }
        .login-footer {
            text-align: center;
            color: #9ca3af;
            font-size: 12px;
            margin-top: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        logo_b64 = _get_logo_base64()
        if logo_b64:
            st.markdown(
                f'<img src="data:image/png;base64,{logo_b64}" class="login-logo">',
                unsafe_allow_html=True,
            )
        else:
            # Fallback emoji if no logo file is found, so the page never looks broken.
            st.markdown('<div style="text-align:center;font-size:46px;">🕵️</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="login-title">Deepfake Detection System</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">AI-powered, explainable, and secure</div>',
                    unsafe_allow_html=True)

        tab_signin, tab_signup = st.tabs(["🔐  Sign In", "📝  Sign Up"])

        # --- Sign In ---
        with tab_signin:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password",
                                         placeholder="Enter your password")
                st.write("")
                submitted = st.form_submit_button("Sign In", use_container_width=True,
                                                  type="primary")
            if submitted:
                if not username or not password:
                    st.warning("Enter both username and password.")
                else:
                    db = get_auth_db()  # ensure the store + default admin exist
                    if authenticate(username, password):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username.strip()
                        st.session_state["role"] = db.get_role(username.strip()) or "user"
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        # --- Sign Up ---
        with tab_signup:
            with st.form("signup_form", clear_on_submit=True):
                new_username = st.text_input("Choose a username", key="signup_user",
                                             placeholder="Pick a username")
                new_password = st.text_input("Choose a password", type="password",
                                             key="signup_pw",
                                             placeholder="At least 6 characters")
                confirm_password = st.text_input("Confirm password", type="password",
                                                  key="signup_pw_confirm",
                                                  placeholder="Re-enter password")
                st.write("")
                signup_submitted = st.form_submit_button("Create Account",
                                                         use_container_width=True,
                                                         type="primary")
            if signup_submitted:
                db = get_auth_db()
                if not new_username or not new_password:
                    st.warning("Username and password are required.")
                elif len(new_password) < 6:
                    st.warning("Password must be at least 6 characters.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif db.user_exists(new_username.strip()):
                    st.error(f"Username '{new_username}' is already taken.")
                else:
                    db.add_user(new_username.strip(), new_password, role="user")
                    st.success("Account created! Switch to 'Sign In' to log in.")

        st.markdown(
            '<div class="login-footer">First run? Default admin: '
            '<code>admin</code> / <code>admin123</code></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


def require_login() -> bool:
    """Return True if the user is authenticated, else render login and stop."""
    if st.session_state.get("authenticated"):
        return True
    login_page()
    return False


def logout() -> None:
    st.session_state.pop("authenticated", None)
    st.session_state.pop("username", None)
    st.session_state.pop("role", None)
    st.rerun()


def change_password_sidebar() -> None:
    """Self-service password change for the signed-in user (any role)."""
    username = st.session_state.get("username", "")
    with st.sidebar.expander("🔑 Change my password"):
        with st.form("change_my_password", clear_on_submit=True):
            current = st.text_input("Current password", type="password")
            new = st.text_input("New password", type="password")
            confirm = st.text_input("Confirm new password", type="password")
            submitted = st.form_submit_button("Update password",
                                              use_container_width=True)
        if submitted:
            db = get_auth_db()
            if not authenticate(username, current):
                st.error("Current password is incorrect.")
            elif not new:
                st.warning("Enter a new password.")
            elif new != confirm:
                st.error("New passwords do not match.")
            elif new == current:
                st.warning("New password must differ from the current one.")
            elif db.set_password(username, new):
                st.success("Password updated.")
            else:
                st.error("Could not update password.")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_detector():
    return get_detector()


@st.cache_resource(show_spinner="Preparing Grad-CAM...")
def load_explainer():
    return GradCAMExplainer(get_detector())


def get_detector_or_warn():
    """Return the detector, or render a friendly error and stop the page."""
    try:
        return load_detector()
    except ModelNotFoundError as exc:
        st.error(f"⚠️ {exc}")
        st.info(
            "Place a trained checkpoint at `models/best_xception_celebdf.pth`, "
            "or train one with `python src/train.py`."
        )
        st.stop()


def verdict_banner(prob: float, threshold: float) -> None:
    """Render a REAL/FAKE headline with confidence + probability bar."""
    label = label_from_prob(prob, threshold)
    conf = confidence_from_prob(prob, threshold)
    if label == "FAKE":
        st.error(f"### 🔴 DEEPFAKE  —  {conf * 100:.1f}% confidence")
    else:
        st.success(f"### 🟢 REAL  —  {conf * 100:.1f}% confidence")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Raw fake probability: {prob:.4f}  ·  threshold {threshold:.2f}")


# ---------------------------------------------------------------------------
# Pages
def inject_theme_css() -> None:
    """Global CSS for the professional AI-security dashboard theme."""
    st.markdown(
        """
        <style>
        /* ── Global overrides ── */
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

        /* ── Hero section ── */
        .hero-section {
            background: linear-gradient(135deg, rgba(15,25,40,0.95) 0%, rgba(25,40,60,0.95) 100%);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 38px 40px 32px 40px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
        }
        .hero-section::before {
            content: '';
            position: absolute;
            top: -40%;
            right: -10%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(56,152,236,0.08) 0%, transparent 70%);
            border-radius: 50%;
        }
        .hero-title {
            font-size: 32px;
            font-weight: 700;
            color: #e8ecf1;
            margin: 0 0 6px 0;
            letter-spacing: -0.3px;
            line-height: 1.25;
        }
        .hero-title span { color: #3898ec; }
        .hero-subtitle {
            font-size: 15px;
            color: #8a9bb5;
            margin: 0 0 22px 0;
            line-height: 1.55;
            max-width: 640px;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(56,152,236,0.12);
            border: 1px solid rgba(56,152,236,0.25);
            color: #6bb3f0;
            font-size: 11px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 20px;
            letter-spacing: 0.4px;
            text-transform: uppercase;
        }

        /* ── Info cards (Model / Input / Device) ── */
        .info-card {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 10px;
            padding: 16px 18px;
            text-align: center;
        }
        .info-card-label {
            font-size: 11px;
            font-weight: 600;
            color: #6b7a90;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 6px;
        }
        .info-card-value {
            font-size: 22px;
            font-weight: 700;
            color: #d0d8e4;
        }
        .info-card-value.accent { color: #3898ec; }

        /* ── Feature cards ── */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-top: 8px;
        }
        .feature-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 10px;
            padding: 20px 18px 18px 18px;
            transition: border-color 0.2s;
        }
        .feature-card:hover { border-color: rgba(56,152,236,0.35); }
        .feature-icon {
            font-size: 26px;
            margin-bottom: 10px;
            display: block;
        }
        .feature-title {
            font-size: 14px;
            font-weight: 700;
            color: #d0d8e4;
            margin-bottom: 5px;
        }
        .feature-desc {
            font-size: 12.5px;
            color: #7a8a9e;
            line-height: 1.5;
        }

        /* ── Section headers ── */
        .section-label {
            font-size: 11px;
            font-weight: 600;
            color: #506070;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }

        /* ── System status bar ── */
        .status-bar {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 7px;
            font-size: 12.5px;
            color: #7a8a9e;
        }
        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-dot.green { background: #2ecc71; box-shadow: 0 0 6px rgba(46,204,113,0.4); }
        .status-dot.orange { background: #e67e22; box-shadow: 0 0 6px rgba(230,126,34,0.4); }
        .status-dot.red { background: #e74c3c; box-shadow: 0 0 6px rgba(231,76,60,0.4); }

        /* ── Disclaimer ── */
        .disclaimer-box {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 8px;
            padding: 14px 18px;
            font-size: 12.5px;
            color: #6b7a90;
            line-height: 1.55;
        }

        /* ── Sidebar refinement ── */
        [data-testid="stSidebar"] .stRadio > div {
            gap: 2px;
        }
        [data-testid="stSidebar"] .stRadio > div > label {
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 13.5px;
        }
        [data-testid="stSidebar"] .stRadio > div > label:hover {
            background: rgba(255,255,255,0.04);
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.08);
            margin: 10px 0;
        }

        /* ── Image Detection page ── */
        .img-top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .img-top-badge {
            display: inline-block;
            background: rgba(46,204,113,0.10);
            border: 1px solid rgba(46,204,113,0.22);
            color: #6ddba4;
            font-size: 10px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 4px;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        .img-top-status {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11.5px;
            font-weight: 600;
            color: #6ddba4;
            letter-spacing: 0.3px;
        }
        .img-top-status .dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: #2ecc71;
            box-shadow: 0 0 6px rgba(46,204,113,0.5);
        }
        .img-page-title {
            font-size: 26px;
            font-weight: 700;
            color: #e8ecf1;
            margin: 0 0 4px 0;
            letter-spacing: -0.3px;
        }
        .img-page-title span { color: #2ecc71; }
        .img-page-sub {
            font-size: 13.5px;
            color: #7a8a9e;
            margin: 0 0 14px 0;
            line-height: 1.5;
        }
        .img-meta-row {
            display: flex;
            gap: 0;
            margin-bottom: 22px;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 8px;
            overflow: hidden;
        }
        .img-meta-cell {
            flex: 1;
            padding: 10px 16px;
            background: rgba(255,255,255,0.025);
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        .img-meta-cell:last-child { border-right: none; }
        .img-meta-cell .iml {
            font-size: 9.5px; font-weight: 700; color: #506060;
            text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 2px;
        }
        .img-meta-cell .imv {
            font-size: 14px; font-weight: 700; color: #d0d8e4;
        }

        .img-upload-zone {
            border: 1.5px dashed rgba(46,204,113,0.30);
            border-radius: 12px;
            padding: 32px 24px;
            text-align: center;
            background: rgba(46,204,113,0.02);
            margin-bottom: 12px;
        }
        .img-upload-zone .upload-icon {
            font-size: 32px;
            margin-bottom: 8px;
            opacity: 0.5;
        }
        .img-upload-zone .upload-title {
            font-size: 15px; font-weight: 700; color: #d0d8e4; margin-bottom: 4px;
        }
        .img-upload-zone .upload-desc {
            font-size: 12.5px; color: #6b7a90; line-height: 1.5;
        }
        .img-upload-meta {
            display: flex; justify-content: center; gap: 24px;
            font-size: 11.5px; color: #506060; margin-top: 8px;
        }
        .img-upload-meta span { display: flex; align-items: center; gap: 4px; }

        .img-empty-panel {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 28px 24px;
            text-align: center;
            margin-top: 16px;
        }
        .img-empty-panel .ept {
            font-size: 16px; font-weight: 700; color: #d0d8e4; margin-bottom: 4px;
        }
        .img-empty-panel .epd {
            font-size: 13px; color: #6b7a90; margin-bottom: 18px;
        }
        .img-cap-grid {
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
        }
        .img-cap-item {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 16px 12px;
        }
        .img-cap-item .ci-icon { font-size: 20px; margin-bottom: 6px; opacity: 0.6; }
        .img-cap-item .ci-title {
            font-size: 11px; font-weight: 700; color: #8a9bb5;
            text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 3px;
        }
        .img-cap-item .ci-desc {
            font-size: 11.5px; color: #5a6a7a; line-height: 1.4;
        }

        .img-section-label {
            font-size: 10px; font-weight: 700; color: #4a5a6a;
            text-transform: uppercase; letter-spacing: 0.8px;
            margin-bottom: 10px; padding-bottom: 6px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .img-evidence-card {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px;
            overflow: hidden;
        }
        .img-evidence-header {
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 11px; font-weight: 700; color: #6b7a90;
            text-transform: uppercase; letter-spacing: 0.6px;
        }
        .img-evidence-body { padding: 0; }

        .img-analysis-card {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px;
            overflow: hidden;
        }
        .img-analysis-header {
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 11px; font-weight: 700; color: #6b7a90;
            text-transform: uppercase; letter-spacing: 0.6px;
        }
        .img-analysis-body { padding: 20px; }

        .img-verdict-block {
            text-align: center;
            padding: 20px 16px 16px 16px;
            border-radius: 10px;
            margin-bottom: 16px;
        }
        .img-verdict-block.real {
            background: rgba(46,204,113,0.06);
            border: 1px solid rgba(46,204,113,0.18);
        }
        .img-verdict-block.fake {
            background: rgba(231,76,60,0.06);
            border: 1px solid rgba(231,76,60,0.18);
        }
        .img-verdict-block.uncertain {
            background: rgba(230,126,34,0.06);
            border: 1px solid rgba(230,126,34,0.18);
        }
        .img-vb-label {
            font-size: 10px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 1px; margin-bottom: 6px;
        }
        .img-vb-label.real { color: #2ecc71; }
        .img-vb-label.fake { color: #e74c3c; }
        .img-vb-label.uncertain { color: #e67e22; }
        .img-vb-title {
            font-size: 28px; font-weight: 800; color: #e8ecf1; margin-bottom: 6px;
        }
        .img-vb-desc {
            font-size: 12.5px; color: #7a8a9e; line-height: 1.5;
        }

        .img-metrics-tri {
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
            margin-bottom: 16px;
        }
        .img-metric-box {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 8px;
            padding: 12px 10px;
            text-align: center;
        }
        .img-metric-box .mb-label {
            font-size: 9.5px; font-weight: 700; color: #506060;
            text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
        }
        .img-metric-box .mb-value {
            font-size: 18px; font-weight: 700; color: #d0d8e4;
        }
        .img-metric-box .mb-value.green { color: #2ecc71; }
        .img-metric-box .mb-value.red { color: #e74c3c; }

        .img-prob-meter {
            margin: 12px 0 6px 0;
        }
        .img-prob-meter-label {
            display: flex; justify-content: space-between; align-items: baseline;
            margin-bottom: 6px;
        }
        .img-prob-meter-label .pml-text {
            font-size: 12px; font-weight: 600; color: #7a8a9e;
        }
        .img-prob-meter-label .pml-pct {
            font-size: 20px; font-weight: 800; color: #d0d8e4;
        }
        .img-meter-track {
            width: 100%; height: 8px; border-radius: 4px;
            background: rgba(255,255,255,0.06);
            position: relative; overflow: hidden;
        }
        .img-meter-fill {
            height: 100%; border-radius: 4px;
            transition: width 0.4s ease;
        }
        .img-meter-fill.low { background: linear-gradient(90deg, #2ecc71, #27ae60); }
        .img-meter-fill.mid { background: linear-gradient(90deg, #e67e22, #d35400); }
        .img-meter-fill.high { background: linear-gradient(90deg, #e74c3c, #c0392b); }
        .img-meter-threshold {
            position: absolute; top: -2px; width: 2px; height: 12px;
            background: rgba(255,255,255,0.35); border-radius: 1px;
        }
        .img-prob-hint {
            font-size: 11px; color: #506060; line-height: 1.5; margin-top: 8px;
        }

        .img-signals-grid {
            display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;
        }
        .img-signal-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 9px 12px;
            background: rgba(255,255,255,0.02);
            border-radius: 6px;
            font-size: 12px;
        }
        .img-signal-item .si-label { color: #6b7a90; }
        .img-signal-item .si-value { color: #d0d8e4; font-weight: 600; }
        .img-signal-item .si-value.green { color: #2ecc71; }

        .img-explain-card {
            background: rgba(46,204,113,0.03);
            border: 1px solid rgba(46,204,113,0.12);
            border-radius: 10px;
            padding: 18px 20px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .img-explain-text .iet {
            font-size: 14px; font-weight: 700; color: #d0d8e4; margin-bottom: 3px;
        }
        .img-explain-text .ied {
            font-size: 12.5px; color: #6b7a90; line-height: 1.5;
        }

        .img-summary-card {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px;
            overflow: hidden;
        }
        .img-summary-header {
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 11px; font-weight: 700; color: #6b7a90;
            text-transform: uppercase; letter-spacing: 0.6px;
        }
        .img-summary-grid {
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
            background: rgba(255,255,255,0.03);
        }
        .img-summary-cell {
            padding: 12px 16px;
            background: rgba(15,20,30,0.95);
        }
        .img-summary-cell .sc-label {
            font-size: 9.5px; font-weight: 700; color: #506060;
            text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 3px;
        }
        .img-summary-cell .sc-value {
            font-size: 13px; font-weight: 600; color: #d0d8e4;
        }
        .img-summary-cell .sc-value.green { color: #2ecc71; }
        .img-summary-cell .sc-value.red { color: #e74c3c; }

        .img-actions-row {
            display: flex; gap: 10px; margin-top: 16px;
        }

        .img-disclaimer {
            background: rgba(255,255,255,0.015);
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 11.5px;
            color: #4a5a6a;
            line-height: 1.55;
            margin-top: 18px;
        }

        div[data-testid="stFileUploader"] section {
            border: 2px dashed rgba(46,204,113,0.30) !important;
            background: rgba(46, 204, 113, 0.02) !important;
            border-radius: 10px !important;
        }

        /* ── Video page ── */
        .vid-hero {
            background: linear-gradient(135deg, rgba(15,25,40,0.95) 0%, rgba(22,36,52,0.95) 100%);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 32px 36px 28px 36px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }
        .vid-hero::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -8%;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(155,89,182,0.07) 0%, transparent 70%);
            border-radius: 50%;
        }
        .vid-hero-badge {
            display: inline-block;
            background: rgba(155,89,182,0.12);
            border: 1px solid rgba(155,89,182,0.25);
            color: #c39bd3;
            font-size: 11px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 20px;
            letter-spacing: 0.4px;
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        .vid-hero h1 {
            font-size: 28px;
            font-weight: 700;
            color: #e8ecf1;
            margin: 0 0 6px 0;
            letter-spacing: -0.3px;
            line-height: 1.25;
        }
        .vid-hero h1 span { color: #9b59b6; }
        .vid-hero p {
            font-size: 14.5px;
            color: #8a9bb5;
            margin: 0;
            line-height: 1.55;
            max-width: 620px;
        }

        .vid-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 10px;
            padding: 20px 22px;
        }
        .vid-card-label {
            font-size: 11px;
            font-weight: 600;
            color: #506070;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 10px;
        }
        .vid-card-desc {
            font-size: 12.5px;
            color: #6b7a90;
            line-height: 1.5;
            margin-top: 6px;
        }

        .vid-info-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin: 16px 0;
        }
        .vid-info-item {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 8px;
            padding: 14px 14px;
            text-align: center;
        }
        .vid-info-item .vil {
            font-size: 10.5px;
            font-weight: 600;
            color: #506070;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .vid-info-item .viv {
            font-size: 20px;
            font-weight: 700;
            color: #d0d8e4;
        }
        .vid-info-item .viv.green { color: #2ecc71; }
        .vid-info-item .viv.red { color: #e74c3c; }
        .vid-info-item .viv.amber { color: #e67e22; }
        .vid-info-item .viv.blue { color: #3898ec; }

        .vid-verdict-card {
            border-radius: 12px;
            padding: 24px 28px;
            margin: 20px 0;
            border: 1px solid;
        }
        .vid-verdict-card.real {
            background: rgba(46,204,113,0.06);
            border-color: rgba(46,204,113,0.25);
        }
        .vid-verdict-card.fake {
            background: rgba(231,76,60,0.06);
            border-color: rgba(231,76,60,0.25);
        }
        .vid-verdict-label {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .vid-verdict-label.real { color: #2ecc71; }
        .vid-verdict-label.fake { color: #e74c3c; }
        .vid-verdict-title {
            font-size: 26px;
            font-weight: 700;
            color: #e8ecf1;
            margin-bottom: 8px;
        }
        .vid-verdict-conf {
            font-size: 14px;
            color: #8a9bb5;
        }
        .vid-verdict-conf strong { color: #d0d8e4; }

        .vid-frame-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin-top: 14px;
        }
        .vid-frame-stat {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 12px 10px;
            text-align: center;
        }
        .vid-frame-stat .vfs-label {
            font-size: 10px;
            font-weight: 600;
            color: #506070;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 4px;
        }
        .vid-frame-stat .vfs-value {
            font-size: 18px;
            font-weight: 700;
            color: #d0d8e4;
        }
        .vid-frame-stat .vfs-value.green { color: #2ecc71; }
        .vid-frame-stat .vfs-value.red { color: #e74c3c; }

        .vid-disclaimer {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 12px;
            color: #5a6a7a;
            line-height: 1.55;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
# ---------------------------------------------------------------------------
def page_home(threshold: float) -> None:
    # ── Hero section ──
    st.markdown(
        '<div class="hero-section">'
        '<div class="hero-badge">AI Security</div>'
        '<h1 class="hero-title">AI-Powered <span>Explainable</span> Deepfake Detection</h1>'
        '<p class="hero-subtitle">'
        'Detect whether an image, video, or live webcam stream is authentic or manipulated. '
        'Confidence scores and Grad-CAM explainability provide transparent, '
        'auditable decision support for media verification.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Model info cards ──
    model_label = MODEL_NAME.replace("_", " ").title()
    device_label = "GPU (CUDA)" if DEVICE == "cuda" else "CPU"
    device_accent = "accent" if DEVICE == "cuda" else ""

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="info-card">'
            '<div class="info-card-label">Model Architecture</div>'
            f'<div class="info-card-value accent">{model_label}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="info-card">'
            '<div class="info-card-label">Input Resolution</div>'
            f'<div class="info-card-value">{IMG_SIZE} x {IMG_SIZE} px</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="info-card">'
            '<div class="info-card-label">Inference Device</div>'
            f'<div class="info-card-value {device_accent}">{device_label}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)

    # ── Capabilities ──
    st.markdown('<div class="section-label">Capabilities</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="feature-grid">'

        '<div class="feature-card">'
        '<span class="feature-icon">&#x1F4F7;</span>'
        '<div class="feature-title">Image Detection</div>'
        '<div class="feature-desc">Upload a photo to receive an instant REAL or DEEPFAKE verdict with confidence score and face localization.</div>'
        '</div>'

        '<div class="feature-card">'
        '<span class="feature-icon">&#x1F3AC;</span>'
        '<div class="feature-title">Video Detection</div>'
        '<div class="feature-desc">Frame-by-frame analysis with majority voting, per-frame probability tracking, and temporal statistics.</div>'
        '</div>'

        '<div class="feature-card">'
        '<span class="feature-icon">&#x1F4F9;</span>'
        '<div class="feature-title">Real-Time Webcam</div>'
        '<div class="feature-desc">Live detection directly from your camera with bounding boxes, smoothed predictions, and real-time FPS display.</div>'
        '</div>'

        '<div class="feature-card">'
        '<span class="feature-icon">&#x1F525;</span>'
        '<div class="feature-title">Grad-CAM Explainability</div>'
        '<div class="feature-desc">Visual heatmaps reveal which facial regions influenced the model\'s decision, enabling transparent audit trails.</div>'
        '</div>'

        '<div class="feature-card">'
        '<span class="feature-icon">&#x1F5C2;</span>'
        '<div class="feature-title">Prediction History</div>'
        '<div class="feature-desc">Every analysis is logged to SQLite with timestamps, confidence scores, and source metadata for review.</div>'
        '</div>'

        '<div class="feature-card">'
        '<span class="feature-icon">&#x1F4CA;</span>'
        '<div class="feature-title">Analytics Dashboard</div>'
        '<div class="feature-desc">Aggregate statistics including real vs. fake distribution, average confidence, and inference performance metrics.</div>'
        '</div>'

        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)

    # ── System status ──
    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)

    checkpoint_ok = CHECKPOINT_PATH.exists()
    model_status_class = "green" if checkpoint_ok else "red"
    model_status_text = "Checkpoint loaded" if checkpoint_ok else "Checkpoint missing"
    device_status_class = "green" if DEVICE == "cuda" else "orange"
    device_status_text = "GPU acceleration" if DEVICE == "cuda" else "CPU mode"

    st.markdown(
        '<div class="status-bar">'
        f'<div class="status-item"><span class="status-dot {model_status_class}"></span> Model: {model_status_text}</div>'
        f'<div class="status-item"><span class="status-dot {device_status_class}"></span> Device: {device_status_text}</div>'
        f'<div class="status-item"><span class="status-dot green"></span> API: Operational</div>'
        f'<div class="status-item"><span class="status-dot green"></span> Database: Connected</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not checkpoint_ok:
        st.warning(
            f"Model checkpoint not found at `{CHECKPOINT_PATH}`. "
            "Detection pages will prompt you to add one."
        )

    st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown(
        '<div class="disclaimer-box">'
        '<strong>Disclaimer:</strong> Predictions are probabilistic and may contain errors. '
        'This system is designed as a decision-support aid and should not be used as '
        'the sole basis for authenticity judgments. Always apply human oversight for '
        'critical verification decisions.'
        '</div>',
        unsafe_allow_html=True,
    )


def page_image(detector, threshold: float) -> None:
    from datetime import datetime

    # ── 1. TOP HEADER ──
    st.markdown(
        '<div class="img-top-bar">'
        '<div class="img-top-badge">AI VISUAL FORENSICS</div>'
        '<div class="img-top-status"><span class="dot"></span> SYSTEM READY</div>'
        '</div>'
        '<div class="img-page-title">Image <span>Deepfake</span> Detection</div>'
        '<div class="img-page-sub">Analyze digital images for signs of synthetic or manipulated content.</div>',
        unsafe_allow_html=True,
    )

    # Metadata row
    device_short = "GPU" if DEVICE == "cuda" else "CPU"
    st.markdown(
        '<div class="img-meta-row">'
        '<div class="img-meta-cell"><div class="iml">Model</div><div class="imv">'
        + MODEL_NAME.replace("_", " ").title()
        + '</div></div>'
        '<div class="img-meta-cell"><div class="iml">Input</div><div class="imv">'
        + str(IMG_SIZE) + " x " + str(IMG_SIZE)
        + '</div></div>'
        '<div class="img-meta-cell"><div class="iml">Engine</div><div class="imv">AI Vision</div></div>'
        '<div class="img-meta-cell"><div class="iml">Device</div><div class="imv">'
        + device_short
        + '</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── 2. UPLOAD WORKSPACE ──
    st.markdown(
        '<div class="img-section-label">Upload Evidence</div>'
        '<div class="img-upload-zone">'
        '<div class="upload-title">Upload Evidence</div>'
        '<div class="upload-desc">Upload an image containing a visible face for AI-assisted forensic analysis.</div>'
        '<div class="img-upload-meta">'
        '<span>JPG</span><span>JPEG</span><span>PNG</span>'
        '<span style="margin-left:12px;">Max 200 MB</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    # ── 10. EMPTY STATE ──
    if uploaded is None:
        st.markdown(
            '<div class="img-empty-panel">'
            '<div class="ept">Ready for Analysis</div>'
            '<div class="epd">Upload an image to begin AI-powered visual forensics.</div>'
            '<div class="img-cap-grid">'
            '<div class="img-cap-item">'
            '<div class="ci-title">Image Analysis</div>'
            '<div class="ci-desc">AI-powered classification</div>'
            '</div>'
            '<div class="img-cap-item">'
            '<div class="ci-title">Face Detection</div>'
            '<div class="ci-desc">Automatic face localization</div>'
            '</div>'
            '<div class="img-cap-item">'
            '<div class="ci-title">Model Explainability</div>'
            '<div class="ci-desc">Grad-CAM visualization</div>'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── 3. RUN INFERENCE ──
    with st.spinner("Analyzing image..."):
        result = run_image(detector, uploaded)

    # ── Extract values (no recalculation) ──
    prediction = result.get("prediction", "NO_FACE")
    probability_fake = float(
        result.get("probability_fake", result.get("fake_probability", 0.0))
    )
    inference_time = float(result.get("inference_time", 0.0))
    face_detected = result.get("face_detected", prediction != "NO_FACE")
    face_box = result.get("face_box")

    label = label_from_prob(probability_fake, threshold) if prediction != "NO_FACE" else "NO_FACE"
    conf = confidence_from_prob(probability_fake, threshold) if prediction != "NO_FACE" else 0.0

    file_size_mb = (uploaded.size or 0) / (1024 * 1024)

    # ── Two-column analysis workspace ──
    left_col, right_col = st.columns([1, 1])

    # ── LEFT: Evidence Preview ──
    with left_col:
        st.markdown(
            '<div class="img-evidence-card">'
            '<div class="img-evidence-header">Evidence Preview</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.image(uploaded, caption="Uploaded evidence", use_container_width=True)

        meta_items = [
            ("Filename", uploaded.name),
            ("File Size", f"{file_size_mb:.2f} MB"),
            ("Face", "Detected" if face_detected else "Not Detected"),
        ]
        for lbl, val in meta_items:
            face_cls = " green" if (lbl == "Face" and face_detected) else ""
            st.markdown(
                '<div class="img-signal-item">'
                '<span class="si-label">' + lbl + '</span>'
                '<span class="si-value' + face_cls + '">' + val + '</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── RIGHT: AI Analysis ──
    with right_col:
        st.markdown(
            '<div class="img-analysis-card">'
            '<div class="img-analysis-header">AI Analysis</div>'
            '<div class="img-analysis-body">',
            unsafe_allow_html=True,
        )

        if prediction == "NO_FACE":
            st.markdown(
                '<div class="img-verdict-block uncertain">'
                '<div class="img-vb-label uncertain">No Face Detected</div>'
                '<div class="img-vb-title">Unable to Analyze</div>'
                '<div class="img-vb-desc">No detectable face found. Please upload an image '
                'with a clear, visible face for reliable analysis.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            if label == "REAL":
                vc, vt = "real", "AUTHENTIC"
                vd = "Model indicates the image appears consistent with authentic content."
            elif label == "FAKE":
                vc, vt = "fake", "DEEPFAKE"
                vd = "Model indicates potential synthetic or manipulated content."
            else:
                vc, vt = "uncertain", "UNCERTAIN"
                vd = "Model confidence is insufficient for a reliable classification."

            st.markdown(
                '<div class="img-verdict-block ' + vc + '">'
                '<div class="img-vb-label ' + vc + '">Analysis Complete</div>'
                '<div class="img-vb-title">' + vt + '</div>'
                '<div class="img-vb-desc">' + vd + '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Three metrics
            prob_pct = probability_fake * 100
            conf_pct = conf * 100
            prob_color = "red" if probability_fake >= 0.5 else "green"
            conf_color = "green" if conf >= 0.7 else ("red" if conf < 0.5 else "")

            st.markdown(
                '<div class="img-metrics-tri">'
                '<div class="img-metric-box"><div class="mb-label">Fake Probability</div>'
                '<div class="mb-value ' + prob_color + '">' + f"{prob_pct:.1f}%" + '</div></div>'
                '<div class="img-metric-box"><div class="mb-label">Confidence</div>'
                '<div class="mb-value ' + conf_color + '">' + f"{conf_pct:.1f}%" + '</div></div>'
                '<div class="img-metric-box"><div class="mb-label">Threshold</div>'
                '<div class="mb-value">' + f"{threshold:.2f}" + '</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Confidence visualization
            fill_class = "low" if probability_fake < 0.35 else ("mid" if probability_fake < 0.65 else "high")
            threshold_pct = threshold * 100
            bar_width = min(prob_pct, 100)

            st.markdown(
                '<div class="img-prob-meter">'
                '<div class="img-prob-meter-label">'
                '<span class="pml-text">Fake Probability</span>'
                '<span class="pml-pct">' + f"{prob_pct:.1f}%" + '</span>'
                '</div>'
                '<div class="img-meter-track">'
                '<div class="img-meter-fill ' + fill_class + '" style="width:' + f"{bar_width:.1f}" + '%;"></div>'
                '<div class="img-meter-threshold" style="left:' + f"{threshold_pct:.0f}" + '%;"></div>'
                '</div>'
                '<div class="img-prob-hint">Probability represents the model\'s estimated likelihood that '
                'the analyzed image contains synthetic or manipulated content.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── 5. Detection Signals ──
    st.markdown('<div class="img-section-label" style="margin-top:18px;">Detection Signals</div>', unsafe_allow_html=True)
    sig_items = [
        ("Face", "Detected" if face_detected else "Not Detected", "green" if face_detected else ""),
        ("Model", MODEL_NAME.replace("_", " ").title(), ""),
        ("Analysis Mode", "Image", ""),
        ("Inference", f"{inference_time * 1000:.0f} ms" if inference_time > 0 else "N/A", ""),
    ]
    st.markdown('<div class="img-signals-grid">', unsafe_allow_html=True)
    for lbl, val, color in sig_items:
        st.markdown(
            '<div class="img-signal-item">'
            '<span class="si-label">' + lbl + '</span>'
            '<span class="si-value ' + color + '">' + val + '</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 6. Explainability ──
    if prediction != "NO_FACE":
        st.markdown(
            '<div class="img-explain-card" style="margin-top:16px;">'
            '<div class="img-explain-text">'
            '<div class="iet">Model Explainability</div>'
            '<div class="ied">Visualize the facial regions that contributed most to the model\'s prediction.</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── 7. Analysis Summary ──
    if prediction != "NO_FACE":
        st.markdown(
            '<div class="img-summary-card" style="margin-top:16px;">'
            '<div class="img-summary-header">Analysis Summary</div>'
            '<div class="img-summary-grid">',
            unsafe_allow_html=True,
        )
        verdict_color = "green" if label == "REAL" else ("red" if label == "FAKE" else "")
        face_color = "green" if face_detected else "red"
        summary_cells = [
            ("Verdict", label, verdict_color),
            ("Fake Probability", f"{probability_fake:.4f}", ""),
            ("Confidence", f"{conf:.4f}", ""),
            ("Face Detection", "Detected" if face_detected else "Not Detected", face_color),
            ("Model", MODEL_NAME.replace("_", " ").title(), ""),
            ("Threshold", f"{threshold:.2f}", ""),
        ]
        for lbl, val, color in summary_cells:
            st.markdown(
                '<div class="img-summary-cell">'
                '<div class="sc-label">' + lbl + '</div>'
                '<div class="sc-value ' + color + '">' + val + '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── 8. Actions ──
    act_c1, act_c2, act_c3 = st.columns(3)
    with act_c1:
        if st.button("Analyze Another Image", key="img_retry", use_container_width=True):
            st.rerun()
    with act_c2:
        if st.button("View Prediction History", key="img_history", use_container_width=True):
            st.session_state["_nav_target"] = "Prediction History"
            st.rerun()
    with act_c3:
        if prediction != "NO_FACE":
            if st.button("Explain with Grad-CAM", key="img_gradcam", use_container_width=True):
                st.session_state["_nav_target"] = "Grad-CAM"
                st.rerun()

    # ── 9. Disclaimer ──
    st.markdown(
        '<div class="img-disclaimer">'
        '<strong>AI-assisted analysis.</strong> Predictions are probabilistic and may contain '
        'false positives or false negatives. Results should be treated as decision-support '
        'information rather than definitive proof.'
        '</div>',
        unsafe_allow_html=True,
    )

def page_video(detector, threshold: float) -> None:
    # ── Hero ──
    st.markdown(
        '<div class="vid-hero">'
        '<div class="vid-hero-badge">Forensic Analysis</div>'
        '<h1>Video <span>Deepfake</span> Detection</h1>'
        '<p>Analyze uploaded video frames to identify potential synthetic or '
        'manipulated content with temporal consistency scoring.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Upload + Config side-by-side ──
    upload_col, config_col = st.columns([1.2, 1])

    with upload_col:
        st.markdown(
            '<div class="vid-card">'
            '<div class="vid-card-label">Upload Video</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Drag and drop or click to upload",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="vid-card-desc" style="margin-top:-4px;">'
            'Supported: MP4, AVI, MOV, MKV &nbsp;&middot;&nbsp; Max 200 MB'
            '</div>',
            unsafe_allow_html=True,
        )

    with config_col:
        st.markdown(
            '<div class="vid-card">'
            '<div class="vid-card-label">Analysis Configuration</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        max_frames = st.slider(
            "Frames to sample",
            4,
            32,
            VIDEO_MAX_FRAMES,
        )
        st.markdown(
            '<div class="vid-card-desc">'
            'More sampled frames provide broader temporal analysis but may '
            'increase processing time.</div>',
            unsafe_allow_html=True,
        )

    if uploaded is None:
        return

    # ── Video preview + file info ──
    st.video(uploaded)

    file_size_mb = (uploaded.size or 0) / (1024 * 1024)
    st.markdown(
        '<div class="vid-card">'
        '<div class="vid-card-label">Video Information</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    info_c1, info_c2, info_c3 = st.columns(3)
    with info_c1:
        st.markdown(
            '<div class="vid-info-item">'
            '<div class="vil">File Name</div>'
            f'<div class="viv blue" style="font-size:14px;">{uploaded.name}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with info_c2:
        st.markdown(
            '<div class="vid-info-item">'
            '<div class="vil">File Size</div>'
            f'<div class="viv">{file_size_mb:.1f} MB</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with info_c3:
        st.markdown(
            '<div class="vid-info-item">'
            '<div class="vil">Frames to Sample</div>'
            f'<div class="viv blue">{max_frames}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Processing pipeline ──
    st.markdown(
        '<div class="vid-card">'
        '<div class="vid-card-label">Processing Pipeline</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    steps_col1, steps_col2, steps_col3, steps_col4, steps_col5 = st.columns(5)
    step_labels = ["Upload", "Sample Frames", "Detect Faces", "Run Model", "Aggregate"]
    for col, label in zip([steps_col1, steps_col2, steps_col3, steps_col4, steps_col5], step_labels):
        with col:
            st.markdown(
                f'<div style="text-align:center; padding:10px 4px; background:rgba(155,89,182,0.08); '
                f'border-radius:6px; font-size:11.5px; color:#c39bd3; font-weight:600;">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )

    # ── Run inference ──
    with st.spinner("Analyzing video..."):
        result = run_video(
            detector,
            uploaded,
            max_frames=max_frames,
        )

    # ── No face handling ──
    if result.get("prediction") == "NO_FACE":
        st.markdown(
            '<div class="vid-verdict-card fake">'
            '<div class="vid-verdict-label fake">No Face Detected</div>'
            '<div class="vid-verdict-title">Unable to Analyze</div>'
            '<div class="vid-verdict-conf">No face was detected in any sampled frame. '
            'Try a video with clearer, front-facing subjects.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Extract values ──
    probability_fake = float(
        result.get("probability_fake", result.get("fake_probability", 0.0))
    )
    inference_time = float(result.get("inference_time", 0.0))
    frames_sampled = int(result.get("frames_sampled", 0))
    frames_with_face = int(result.get("frames_with_face", 0))
    frame_probabilities = result.get("frame_probabilities", []) or []

    fake_frames = result.get("fake_frames")
    real_frames = result.get("real_frames")
    if fake_frames is None:
        fake_frames = sum(1 for p in frame_probabilities if float(p) >= threshold)
    if real_frames is None:
        real_frames = sum(1 for p in frame_probabilities if float(p) < threshold)

    label = label_from_prob(probability_fake, threshold)
    conf = confidence_from_prob(probability_fake, threshold)

    # ── Verdict card ──
    verdict_class = "fake" if label == "FAKE" else "real"
    verdict_status = "DEEPFAKE" if label == "FAKE" else "AUTHENTIC"
    verdict_color_class = "fake" if label == "FAKE" else "real"

    st.markdown(
        f'<div class="vid-verdict-card {verdict_class}">'
        f'<div class="vid-verdict-label {verdict_color_class}">Analysis Complete</div>'
        f'<div class="vid-verdict-title">{verdict_status}</div>'
        f'<div class="vid-verdict-conf">'
        f'Confidence: <strong>{conf * 100:.1f}%</strong> &nbsp;&middot;&nbsp; '
        f'Fake probability: <strong>{probability_fake:.4f}</strong> &nbsp;&middot;&nbsp; '
        f'Threshold: <strong>{threshold:.2f}</strong> &nbsp;&middot;&nbsp; '
        f'Inference: <strong>{inference_time * 1000:.0f} ms</strong>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Frame analysis summary ──
    st.markdown(
        '<div class="vid-card">'
        '<div class="vid-card-label">Frame Analysis Summary</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    fs_c1, fs_c2, fs_c3, fs_c4, fs_c5 = st.columns(5)
    frame_stats = [
        ("Total Sampled", str(frames_sampled), "blue"),
        ("With Face", str(frames_with_face), "blue"),
        ("Real Frames", str(real_frames), "green"),
        ("Fake Frames", str(fake_frames), "red"),
        ("Verdict", label, "green" if label == "REAL" else "red"),
    ]
    for col, (lbl, val, color) in zip(
        [fs_c1, fs_c2, fs_c3, fs_c4, fs_c5], frame_stats
    ):
        with col:
            st.markdown(
                f'<div class="vid-frame-stat">'
                f'<div class="vfs-label">{lbl}</div>'
                f'<div class="vfs-value {color}">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Per-frame probability chart ──
    if frame_probabilities:
        st.markdown(
            '<div class="vid-card" style="margin-top:14px;">'
            '<div class="vid-card-label">Per-Frame Fake Probability</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.line_chart(frame_probabilities)

    # ── Disclaimer ──
    st.markdown(
        '<div class="vid-disclaimer">'
        'AI predictions are probabilistic and may contain false positives or false negatives. '
        'Results should be treated as decision-support information and verified with human oversight.'
        '</div>',
        unsafe_allow_html=True,
    )

def page_gradcam(detector, explainer, threshold: float) -> None:
    st.header("🔥 Grad-CAM Explainability")
    st.write("See which facial regions drove the model's decision.")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],
                                key="gradcam")
    if uploaded is None:
        return

    file_bytes = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not read the image.")
        return

    with st.spinner("Generating Grad-CAM..."):
        out = explainer.explain_image_array(bgr)

    if out is None:
        st.warning("No face detected — cannot generate a heatmap.")
        return

    verdict_banner(out["fake_probability"], threshold)
    c1, c2 = st.columns(2)
    c1.image(out["original"], caption="Detected face", use_container_width=True)
    c2.image(out["overlay"], caption="Grad-CAM overlay", use_container_width=True)
    st.caption("Warmer colors = more influence on the 'fake' decision.")


def page_history() -> None:
    st.header("🗂️ Prediction History")
    db = get_db()
    rows = db.recent(limit=200)
    if not rows:
        st.info("No predictions yet. Analyze an image or video to populate history.")
        return

    df = pd.DataFrame(rows)[
        ["created_at", "media_type", "prediction", "confidence",
         "fake_probability", "inference_time", "source", "file_name"]
    ]
    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download CSV", df.to_csv(index=False),
                       file_name="prediction_history.csv", mime="text/csv")
    if c2.button("🗑️ Clear history"):
        db.clear()
        st.rerun()


def page_dashboard() -> None:
    st.header("📊 Dashboard")
    db = get_db()
    stats = db.stats()
    if stats["total"] == 0:
        st.info("No data yet — run some predictions first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total predictions", stats["total"])
    c2.metric("Flagged FAKE", stats["fake"])
    c3.metric("Avg confidence", f"{stats['avg_confidence'] * 100:.1f}%")
    c4.metric("Avg inference", f"{stats['avg_inference_time'] * 1000:.0f} ms")

    rows = db.recent(limit=500)
    df = pd.DataFrame(rows)
    left, right = st.columns(2)
    with left:
        st.subheader("Real vs Fake")
        st.bar_chart(df["prediction"].value_counts())
    with right:
        st.subheader("By media type")
        st.bar_chart(df["media_type"].value_counts())


def page_about() -> None:
    st.header("ℹ️ About")
    st.markdown(
        f"""
        **AI-Powered Explainable Deepfake Detection System**

        - **Model:** {MODEL_NAME} (XceptionNet backbone)
        - **Dataset:** Celeb-DF v2,DFDC,faceForensics++,and Wild Deepfake datasets trained on a combine dataset
        - **Explainability:** Grad-CAM over the last convolutional layer
        - **Backend:** FastAPI REST API (`/docs` for Swagger)
        - **Storage:** SQLite prediction history
        - **Deployment:** Docker + Docker Compose

        Built with PyTorch, OpenCV, Streamlit, and FastAPI.

        ### Pipeline
        `Media → Face detection → Crop → Resize → Xception → Probability → Verdict`

        ### Disclaimer
        Predictions are probabilistic and can err. Use as a decision-support
        aid, not as sole ground truth.
        """
    )


def page_admin() -> None:
    """Admin-only user management: list, create, reset password, delete."""
    st.header("👥 User Management")
    if st.session_state.get("role") != "admin":
        st.error("You need the **admin** role to manage users.")
        return

    db = get_auth_db()
    current_user = st.session_state.get("username", "")

    # --- Existing users ---
    users = db.list_users()
    st.subheader("Existing users")
    st.dataframe(pd.DataFrame(users), use_container_width=True, hide_index=True)

    st.markdown("---")
    col_create, col_manage = st.columns(2)

    # --- Create user ---
    with col_create:
        st.subheader("➕ Create user")
        with st.form("create_user", clear_on_submit=True):
            new_user = st.text_input("Username")
            new_pw = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["user", "admin"])
            if st.form_submit_button("Create", type="primary", use_container_width=True):
                if not new_user or not new_pw:
                    st.warning("Username and password are required.")
                elif db.add_user(new_user, new_pw, role=new_role):
                    st.success(f"User '{new_user}' created.")
                    st.rerun()
                else:
                    st.error(f"Username '{new_user}' already exists.")

    # --- Reset password / delete ---
    with col_manage:
        st.subheader("🔑 Reset password")
        usernames = [u["username"] for u in users]
        with st.form("reset_pw", clear_on_submit=True):
            target = st.selectbox("User", usernames, key="reset_target")
            reset_pw = st.text_input("New password", type="password", key="reset_pw_val")
            if st.form_submit_button("Update password", use_container_width=True):
                if not reset_pw:
                    st.warning("Enter a new password.")
                elif db.set_password(target, reset_pw):
                    st.success(f"Password updated for '{target}'.")
                else:
                    st.error("User not found.")

        st.subheader("🗑️ Delete user")
        deletable = [u for u in usernames if u != current_user]
        if not deletable:
            st.caption("No other users to delete.")
        else:
            with st.form("delete_user", clear_on_submit=True):
                del_target = st.selectbox("User", deletable, key="del_target")
                confirm = st.checkbox(f"Confirm delete '{del_target}'")
                if st.form_submit_button("Delete", use_container_width=True):
                    if not confirm:
                        st.warning("Tick the confirmation box first.")
                    elif (db.get_role(del_target) == "admin"
                          and db.count_by_role("admin") <= 1):
                        st.error("Cannot delete the last remaining admin.")
                    elif db.delete_user(del_target):
                        st.success(f"User '{del_target}' deleted.")
                        st.rerun()
                    else:
                        st.error("User not found.")


# ---------------------------------------------------------------------------
# Main / routing
# ---------------------------------------------------------------------------
def main() -> None:
    # --- Auth gate: nothing renders until the user signs in ---
    if not require_login():
        return
    inject_theme_css()

    st.sidebar.title("Deepfake Detector")
    st.sidebar.markdown(
        f'<div style="font-size:12.5px; color:#8a9bb5; margin-bottom:10px;">'
        f'Signed in as <strong style="color:#d0d8e4;">{st.session_state.get("username", "user")}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div style="margin-bottom: 4px;"></div>', unsafe_allow_html=True)

    nav_pages = ["Home", "Image Detection", "Video Detection", "Webcam Detection",
                 "Grad-CAM", "Prediction History", "Dashboard", "About"]
    if st.session_state.get("role") == "admin":
        nav_pages.append("User Management")
    page = st.sidebar.radio("Navigate", nav_pages, label_visibility="collapsed")

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    # ── Threshold control ──
    st.sidebar.markdown(
        '<div style="font-size:12px; font-weight:600; color:#6b7a90; '
        'text-transform:uppercase; letter-spacing:0.6px; margin-bottom:6px;">'
        'Decision Threshold</div>',
        unsafe_allow_html=True,
    )
    threshold = st.sidebar.slider(
        "Decision threshold", 0.0, 1.0,
        DEFAULT_THRESHOLD, 0.01,
        label_visibility="collapsed",
        help="Fake probability >= threshold triggers a FAKE verdict. Lower = more sensitive.",
    )
    st.sidebar.markdown(
        f'<div style="font-size:12px; color:#6b7a90; margin-top:-8px; margin-bottom:12px;">'
        f'Verdict fires at <strong style="color:#d0d8e4;">{threshold:.2f}</strong> '
        f'&mdash; lower values flag more samples as fake</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    # ── System info ──
    device_color = "#2ecc71" if DEVICE == "cuda" else "#e67e22"
    st.sidebar.markdown(
        f'<div style="font-size:12px; color:#6b7a90; margin-bottom:6px;">'
        f'<span style="display:inline-block; width:6px; height:6px; border-radius:50%; '
        f'background:{device_color}; margin-right:6px;"></span>'
        f'Device: <code style="color:#d0d8e4;">{DEVICE}</code>'
        f' &nbsp;|&nbsp; Model: <code style="color:#d0d8e4;">{MODEL_NAME}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not CHECKPOINT_PATH.exists():
        st.sidebar.error("Model checkpoint missing")

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    if st.sidebar.button("Log out", use_container_width=True):
        logout()
    change_password_sidebar()

    # Pages that don't need the model can render without loading it.
    if page == "Home":
        page_home(threshold)
    elif page == "Prediction History":
        page_history()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "About":
        page_about()
    elif page == "User Management":
        page_admin()
    else:
        detector = get_detector_or_warn()
        if page == "Image Detection":
            page_image(detector, threshold)
        elif page == "Video Detection":
            page_video(detector, threshold)
        elif page == "Webcam Detection":
            webcam_page(detector, threshold)
        elif page == "Grad-CAM":
            explainer = load_explainer()
            page_gradcam(detector, explainer, threshold)


if __name__ == "__main__":
    main()