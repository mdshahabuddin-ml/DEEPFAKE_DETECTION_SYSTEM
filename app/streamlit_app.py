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
from app.predict import run_audio, run_image, run_video, store_result
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
            background: radial-gradient(900px 420px at 12% 20%, rgba(47,123,255,.12), transparent 60%),
                        radial-gradient(900px 500px at 88% 30%, rgba(124,58,237,.12), transparent 60%),
                        radial-gradient(700px 500px at 50% 110%, rgba(34,211,238,.07), transparent 60%),
                        linear-gradient(180deg, #04070f 0%, #060b18 55%, #04070f 100%);
        }
        .login-card {
            background: linear-gradient(165deg, rgba(17,28,55,.94) 0%, rgba(8,12,28,.96) 100%);
            border: 1px solid rgba(59,157,255,.38);
            border-radius: 20px;
            padding: 34px 40px 24px 40px;
            box-shadow: 0 0 42px rgba(59,157,255,.20), 0 0 100px rgba(124,58,237,.12),
                        inset 0 0 70px rgba(59,157,255,.05);
            margin-top: 8px;
        }
        .login-shield { display: flex; justify-content: center; margin-bottom: 12px;
            filter: drop-shadow(0 0 18px rgba(59,157,255,.55)); }
        .login-title { text-align: center; font-size: 29px; font-weight: 800;
            color: #fff; margin-bottom: 4px; letter-spacing: -.4px; }
        .login-title .grad { background: linear-gradient(90deg, #3b9dff, #7c5cff);
            -webkit-background-clip: text; background-clip: text; color: transparent; }
        .login-subtitle { text-align: center; color: #8b9bb0; font-size: 13.5px; margin-bottom: 18px; }
        div[data-testid="stForm"] { border: none; padding: 0; background: transparent; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; justify-content: center;
            border-bottom: 1px solid rgba(255,255,255,.08); }
        .stTabs [data-baseweb="tab"] { background: transparent; color: #7c8da6;
            padding: 10px 22px; font-weight: 700; border-bottom: 2px solid transparent; }
        .stTabs [aria-selected="true"] { color: #fff !important;
            border-bottom: 2px solid #22d3ee !important;
            text-shadow: 0 0 12px rgba(34,211,238,.6); }
        div[data-testid="stTextInput"] label { color: #dbe4f0 !important;
            font-size: 13.5px !important; font-weight: 600 !important; }
        div[data-testid="stTextInput"] div[data-baseweb="input"] {
            background: #0d1526 !important; border: 1px solid #1e2f4a !important;
            border-radius: 10px !important; color: #e6edf7 !important; }
        div[data-testid="stTextInput"] input { color: #e6edf7 !important; }
        div[data-testid="stTextInput"] input::placeholder { color: #5a6b82 !important; }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #1e90ff 0%, #7c3aed 100%) !important;
            border: none !important; border-radius: 10px !important;
            font-weight: 700 !important; height: 46px !important;
            box-shadow: 0 0 24px rgba(80,120,255,.5) !important; }
        div.stButton > button[kind="secondary"] { border-radius: 10px; font-weight: 600; height: 42px; }
        .login-or { display: flex; align-items: center; gap: 12px; color: #5a6b82;
            font-size: 11.5px; font-weight: 700; letter-spacing: 1px; margin: 16px 0 12px 0; }
        .login-or::before, .login-or::after { content: ""; flex: 1; height: 1px;
            background: rgba(255,255,255,.09); }
        .login-footer { text-align: center; color: #7c8da6; font-size: 12.5px; margin-top: 4px; }
        .login-footer code { background: rgba(34,197,94,.12); border: 1px solid rgba(34,197,94,.35);
            color: #4ade80; border-radius: 6px; padding: 2px 10px; font-size: 11.5px; font-weight: 700; }
        .login-side { padding-top: 30px; }
        .login-chips { display: flex; flex-direction: column; gap: 10px; margin-bottom: 26px; }
        .login-chip { display: flex; align-items: center; gap: 10px;
            background: rgba(15,26,46,.8); border: 1px solid #24406a; border-radius: 10px;
            padding: 11px 14px; color: #c4cfdf; font-size: 12.5px; font-weight: 600;
            box-shadow: 0 0 16px rgba(47,123,255,.10); }
        .login-chip .ci { width: 30px; height: 30px; border-radius: 8px; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center; font-size: 15px;
            background: rgba(47,123,255,.12); border: 1px solid rgba(47,123,255,.3); }
        .login-laptop { border: 1px solid #2a4a7a; border-radius: 12px 12px 0 0;
            background: linear-gradient(160deg, #0d1730, #070c18); padding: 14px;
            box-shadow: 0 0 30px rgba(59,157,255,.15); }
        .login-laptop .scr { border: 1px solid rgba(59,157,255,.3); border-radius: 8px;
            height: 190px; position: relative; overflow: hidden;
            background: radial-gradient(circle at 50% 55%, rgba(59,157,255,.14), transparent 70%); }
        .login-laptop .base { height: 10px; margin: 10px -22px -16px -22px;
            background: linear-gradient(180deg, #1b2c4e, #0a1226); border-radius: 0 0 10px 10px; }
        .login-radar { position: relative; height: 340px; }
        .login-radar .orb { position: absolute; border-radius: 50%; border: 1px solid rgba(59,157,255,.25); }
        .login-node { position: absolute; width: 52px; height: 52px; border-radius: 12px;
            background: rgba(15,26,46,.9); border: 1px solid #2a4a7a;
            display: flex; align-items: center; justify-content: center; font-size: 22px;
            box-shadow: 0 0 18px rgba(124,58,237,.3); }
        .login-badges { display: flex; justify-content: center; align-items: stretch;
            gap: 0; margin: 26px auto 10px auto; max-width: 900px; flex-wrap: wrap; }
        .login-badges .lb { display: flex; align-items: center; gap: 10px;
            padding: 4px 26px; color: #dbe4f0; font-size: 13px; font-weight: 700; line-height: 1.4; }
        .login-badges .lb + .lb { border-left: 1px solid rgba(255,255,255,.12); }
        .login-badges .lb .bi { font-size: 22px; filter: drop-shadow(0 0 8px rgba(59,157,255,.5)); }
        @media (max-width: 1000px) { .login-side { display: none; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([1, 1.3, 1], gap="large")
    with left:
        st.markdown(
            '<div class="login-side"><div class="login-chips">'
            '<div class="login-chip"><span class="ci">🖼</span> Image Detection</div>'
            '<div class="login-chip"><span class="ci">▶</span> Video Detection</div>'
            '<div class="login-chip"><span class="ci">♫</span> Audio Detection</div>'
            '<div class="login-chip"><span class="ci">◉</span> Webcam Detection</div>'
            '<div class="login-chip"><span class="ci">📊</span> Analysis &amp; Reports</div>'
            '</div>'
            '<div class="login-laptop"><div class="scr">'
            '<svg width="100%" height="100%" viewBox="0 0 220 190" preserveAspectRatio="xMidYMid meet">'
            '<g fill="none" stroke="#3b9dff" stroke-width="1.4" opacity="0.9">'
            '<path d="M40 30h24M156 30h24M40 160h24M156 160h24" stroke-width="3"/>'
            '<circle cx="110" cy="95" r="52"/><circle cx="110" cy="95" r="38" opacity=".55"/>'
            '<circle cx="110" cy="95" r="24" opacity=".35"/>'
            '<path d="M110 43v-8M110 147v-8M58 95h-8M162 95h-8M73 58l-6-6M147 58l6-6M73 132l-6 6M147 132l6 6"/>'
            '</g>'
            '<g fill="#5ea0ff" opacity="0.95">'
            '<circle cx="110" cy="95" r="3.5"/><circle cx="88" cy="78" r="2.5"/><circle cx="132" cy="78" r="2.5"/>'
            '<circle cx="88" cy="112" r="2.5"/><circle cx="132" cy="112" r="2.5"/><circle cx="110" cy="62" r="2.5"/>'
            '<circle cx="110" cy="128" r="2.5"/><circle cx="76" cy="95" r="2.5"/><circle cx="144" cy="95" r="2.5"/>'
            '</g>'
            '<g stroke="#3b9dff" stroke-width="1" opacity="0.5">'
            '<line x1="110" y1="95" x2="88" y2="78"/><line x1="110" y1="95" x2="132" y2="78"/>'
            '<line x1="110" y1="95" x2="88" y2="112"/><line x1="110" y1="95" x2="132" y2="112"/>'
            '<line x1="110" y1="95" x2="110" y2="62"/><line x1="110" y1="95" x2="110" y2="128"/>'
            '</g></svg>'
            '</div><div class="base"></div></div></div>',
            unsafe_allow_html=True,
        )
    with mid:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        st.markdown(
            '<div class="login-shield">'
            '<svg width="62" height="62" viewBox="0 0 64 64" fill="none">'
            '<defs><linearGradient id="lsh" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#7c5cff"/>'
            '</linearGradient></defs>'
            '<path d="M32 3l20 7v16c0 13-8.5 22.5-20 27C20.5 48.5 12 39 12 26V10l20-7z" '
            'stroke="url(#lsh)" stroke-width="3.5" stroke-linejoin="round"/>'
            '<g stroke="url(#lsh)" stroke-width="2.6" fill="none">'
            '<path d="M20 32c3.5-4.5 7.5-6.5 12-6.5s8.5 2 12 6.5c-3.5 4.5-7.5 6.5-12 6.5s-8.5-2-12-6.5z"/>'
            '<circle cx="32" cy="32" r="3.4" fill="url(#lsh)" stroke="none"/></g>'
            '</svg></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="login-title">Deepfake <span class="grad">Detection</span> System</div>',
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

        st.markdown('<div class="login-or">OR</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="login-footer">First run? Default admin: '
            '<code>admin</code> <code>admin123</code></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="login-side"><div class="login-radar">'
            '<div class="orb" style="inset:20px;"></div>'
            '<div class="orb" style="inset:55px;opacity:.7;"></div>'
            '<div class="orb" style="inset:90px;opacity:.5;"></div>'
            '<div class="orb" style="inset:125px;opacity:.35;"></div>'
            '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">'
            '<svg width="120" height="120" viewBox="0 0 64 64" fill="none" '
            'style="filter:drop-shadow(0 0 22px rgba(59,157,255,.6));">'
            '<defs><linearGradient id="rsh" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="#3b9dff"/><stop offset="1" stop-color="#22d3ee"/>'
            '</linearGradient></defs>'
            '<path d="M32 3l20 7v16c0 13-8.5 22.5-20 27C20.5 48.5 12 39 12 26V10l20-7z" '
            'stroke="url(#rsh)" stroke-width="2.5" stroke-linejoin="round" fill="rgba(47,123,255,.12)"/>'
            '<path d="M24 32l6 6 11-12" stroke="#fff" stroke-width="4" '
            'stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>'
            '</div>'
            '<div class="login-node" style="top:0;right:8px;">▶</div>'
            '<div class="login-node" style="top:34px;left:0;">🖼</div>'
            '<div class="login-node" style="bottom:52px;left:6px;">♫</div>'
            '<div class="login-node" style="bottom:16px;right:14px;">◉</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="login-badges">'
        '<div class="lb"><span class="bi">🛡</span><span>Accurate<br>Detection</span></div>'
        '<div class="lb"><span class="bi">🧬</span><span>Explainable<br>AI (Grad-CAM)</span></div>'
        '<div class="lb"><span class="bi">🔒</span><span>Secure<br>&amp; Private</span></div>'
        '<div class="lb"><span class="bi">⚡</span><span>Real-time<br>Analysis</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )


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


@st.cache_resource(show_spinner="Loading audio detector...")
def load_audio_detector():
    from src.audio_inference import get_audio_detector

    return get_audio_detector()


def get_audio_detector_or_warn():
    """Return the audio detector, or warn that audio results are unavailable."""
    from src.audio_inference import AudioModelNotFoundError

    try:
        return load_audio_detector()
    except AudioModelNotFoundError as exc:
        st.warning(
            "🔇 Audio detection is not available yet — no trained checkpoint "
            "was found, so audio results would not be valid. "
            "Train one with `python -m src.train_audio --data_root data/audio` "
            f"(details below). ({exc})"
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
            border: 1.5px dashed rgba(47,123,255,.35) !important;
            background: rgba(47,123,255,.03) !important;
            border-radius: 10px !important;
        }
        div[data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #1e90ff 0%, #1a5cff 100%) !important;
            color: #fff !important;
            border: 1px solid rgba(120,180,255,.6) !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            box-shadow: 0 0 18px rgba(30,144,255,.45) !important;
        }

        /* ── Image Detection page: reference match (male-only imagery) ── */
        .imx-hero { background: radial-gradient(700px 260px at 20% 0%, rgba(47,123,255,.16), transparent 60%), linear-gradient(120deg, #0b1230 0%, #0d1734 50%, #080d20 100%); border: 1px solid #1c3a5e; border-radius: 14px; margin-bottom: 14px; overflow: hidden; }
        .imx-hero-inner { display: grid; grid-template-columns: 1.25fr .75fr; min-height: 300px; }
        .imx-hero-left { padding: 26px 26px 22px 28px; }
        .imx-badge { display: inline-flex; align-items: center; gap: 7px; border: 1px solid #2a4a7a; color: #9db8d9; background: rgba(47,123,255,.08); padding: 5px 14px; border-radius: 8px; font-size: 10.5px; font-weight: 800; letter-spacing: 1.6px; margin-bottom: 12px; }
        .imx-hero h1 { color: #fff; margin: 0 0 10px 0; font-size: 38px; line-height: 1.05; font-weight: 800; letter-spacing: -.8px; }
        .imx-hero h1 .blue { color: #3b9dff; }
        .imx-hero .sub { color: #9fb0c7; margin: 0 0 18px 0; font-size: 14px; line-height: 1.6; max-width: 520px; }
        .imx-trust { display: flex; gap: 16px; flex-wrap: wrap; }
        .imx-trust .t { display: flex; align-items: center; gap: 8px; color: #a9bad1; font-size: 12px; font-weight: 600; }
        .imx-trust .t .ic { width: 26px; height: 26px; border-radius: 7px; display: flex; align-items: center; justify-content: center; background: rgba(47,123,255,.12); border: 1px solid rgba(47,123,255,.25); font-size: 13px; }
        .imx-hero-right { position: relative; min-height: 300px; overflow: hidden; background: linear-gradient(90deg, #0b1230 0%, rgba(11,18,48,.15) 30%, rgba(5,8,15,.05) 60%, rgba(5,8,15,.55) 100%), url("https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=700&q=80&auto=format&fit=crop") center 15%/cover no-repeat, radial-gradient(circle at 55% 35%, #16244d 0%, #070c18 70%); }
        .imx-hero-right::after { content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(47,123,255,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(47,123,255,.08) 1px, transparent 1px); background-size: 24px 24px; pointer-events: none; }
        .imx-scan { position: absolute; left: 8%; top: 12%; width: 52%; bottom: 12%; z-index: 3; pointer-events: none; }
        .imx-scan i { position: absolute; width: 22px; height: 22px; border: 2.5px solid #3b9dff; filter: drop-shadow(0 0 6px rgba(59,157,255,.9)); }
        .imx-scan i:nth-child(1) { left: 0; top: 0; border-right: none; border-bottom: none; border-radius: 6px 0 0 0; }
        .imx-scan i:nth-child(2) { right: 0; top: 0; border-left: none; border-bottom: none; border-radius: 0 6px 0 0; }
        .imx-scan i:nth-child(3) { left: 0; bottom: 0; border-right: none; border-top: none; border-radius: 0 0 0 6px; }
        .imx-scan i:nth-child(4) { right: 0; bottom: 0; border-left: none; border-top: none; border-radius: 0 0 6px 0; }
        .imx-scores { position: absolute; right: 34%; top: 50%; transform: translateY(-50%); z-index: 4; display: flex; flex-direction: column; gap: 8px; }
        .imx-score { padding: 7px 16px; border-radius: 8px; font-size: 13px; font-weight: 800; letter-spacing: .5px; text-align: center; line-height: 1.5; }
        .imx-score.real { background: rgba(6,40,20,.92); color: #22c55e; border: 1px solid rgba(34,197,94,.7); box-shadow: 0 0 14px rgba(34,197,94,.45); }
        .imx-score.fake { background: rgba(45,8,12,.92); color: #ff6b6b; border: 1px solid rgba(239,68,68,.7); box-shadow: 0 0 14px rgba(239,68,68,.5); }
        .imx-thumbs { position: absolute; right: 10px; top: 14px; bottom: 14px; z-index: 4; display: flex; flex-direction: column; gap: 10px; justify-content: center; }
        .imx-thumbs .th { width: 76px; height: 76px; border-radius: 10px; overflow: hidden; border: 1px solid #2a4a7a; position: relative; background: #0a1322; }
        .imx-thumbs .th img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .imx-thumbs .th.gray img { filter: grayscale(1) contrast(1.05); }
        .imx-thumbs .th.heat::after { content: ""; position: absolute; inset: 0; background: radial-gradient(circle at 35% 40%, rgba(255,235,59,.75) 0%, rgba(255,152,0,.55) 22%, rgba(244,67,54,.5) 45%, rgba(33,150,243,.45) 70%, rgba(13,71,161,.55) 100%); mix-blend-mode: screen; }
        .imx-models { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px; }
        .imx-model { background: #0d1526; border: 1px solid #1a2b45; border-radius: 12px; padding: 14px 16px; display: flex; gap: 12px; align-items: center; }
        .imx-model .mic { width: 44px; height: 44px; border-radius: 10px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 20px; background: rgba(47,123,255,.12); border: 1px solid rgba(47,123,255,.3); color: #7db4ff; }
        .imx-model .k { color: #8b9bb0; font-size: 10.5px; font-weight: 700; letter-spacing: 1.2px; margin: 0; text-transform: uppercase; }
        .imx-model .v { color: #fff; font-size: 16.5px; font-weight: 800; margin: 2px 0 0 0; }
        .imx-work { display: grid; grid-template-columns: 1.7fr 1fr; gap: 12px; margin-bottom: 14px; }
        .imx-upload { background: #0b1322; border: 1.5px dashed rgba(47,123,255,.4); border-radius: 12px; padding: 30px 24px 22px 24px; text-align: center; }
        .imx-upload .cloud { width: 56px; height: 56px; border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; font-size: 26px; background: rgba(47,123,255,.1); border: 1px solid rgba(47,123,255,.3); box-shadow: 0 0 24px rgba(47,123,255,.35); }
        .imx-upload h3 { color: #fff; margin: 0 0 6px 0; font-size: 19px; font-weight: 800; }
        .imx-upload .d { color: #7c8da6; margin: 0 0 6px 0; font-size: 13px; }
        .imx-upload .drag { color: #5f7191; font-size: 12.5px; margin: 10px 0 0 0; }
        .imx-formats { display: flex; align-items: center; gap: 10px; margin-top: 16px; padding: 10px 14px; background: #0d1526; border: 1px solid #1a2b45; border-radius: 10px; font-size: 12px; color: #8b9bb0; }
        .imx-formats .chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 7px; font-weight: 800; font-size: 11.5px; }
        .imx-formats .sep { width: 1px; align-self: stretch; background: #1a2b45; }
        .imx-examples { background: #0b1322; border: 1px solid #1a2b45; border-radius: 12px; padding: 18px; }
        .imx-examples .eh { display: flex; align-items: center; justify-content: center; gap: 8px; color: #c4cfdf; font-size: 13.5px; font-weight: 700; margin-bottom: 14px; }
        .imx-ex-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        a.imx-ex { display: block; border-radius: 10px; overflow: hidden; border: 1px solid #24406a; transition: all .15s ease; background: #0a1322; }
        a.imx-ex:hover { border-color: #3b9dff; box-shadow: 0 0 14px rgba(59,157,255,.4); transform: translateY(-1px); }
        a.imx-ex img { width: 100%; height: 96px; object-fit: cover; display: block; }
        a.imx-ex.gray img { filter: grayscale(1) contrast(1.05); }
        .imx-ex-hint { display: flex; gap: 8px; align-items: flex-start; color: #7c8da6; font-size: 12px; margin-top: 14px; line-height: 1.5; }
        .imx-caps { background: #0a1120; border: 1px solid #1c3a5e; border-radius: 12px; padding: 20px 20px 18px 20px; margin-bottom: 10px; }
        .imx-caps h2 { color: #fff; font-size: 20px; font-weight: 800; margin: 0 0 4px 0; display: flex; align-items: center; gap: 10px; }
        .imx-caps .sub { color: #7c8da6; font-size: 13px; margin: 0 0 16px 0; }
        .imx-caps-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        .imx-cap { background: #0f1a2e; border: 1px solid #1d3454; border-radius: 12px; padding: 16px; display: flex; gap: 12px; align-items: flex-start; position: relative; overflow: hidden; }
        .imx-cap::before { content: ""; position: absolute; left: 0; top: 14px; bottom: 14px; width: 3px; border-radius: 3px; }
        .imx-cap.purple::before { background: #a855f7; box-shadow: 0 0 10px rgba(168,85,247,.8); }
        .imx-cap.green::before { background: #22c55e; box-shadow: 0 0 10px rgba(34,197,94,.8); }
        .imx-cap.orange::before { background: #f97316; box-shadow: 0 0 10px rgba(249,115,22,.8); }
        .imx-cap .cc { width: 52px; height: 52px; border-radius: 12px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 24px; }
        .imx-cap h4 { color: #fff; margin: 0 0 4px 0; font-size: 14.5px; font-weight: 700; }
        .imx-cap p { color: #7c8da6; margin: 0; font-size: 12px; line-height: 1.55; }
        .imx-cap .arrow { margin-left: auto; width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0; align-self: center; background: #182845; border: 1px solid #24406a; color: #9db8d9; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700; }
        @media (max-width: 1100px) {
            .imx-hero-inner { grid-template-columns: 1fr; }
            .imx-hero-right { min-height: 260px; }
            .imx-hero h1 { font-size: 30px; }
            .imx-models { grid-template-columns: repeat(2, 1fr); }
            .imx-work { grid-template-columns: 1fr; }
            .imx-caps-grid { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 640px) {
            .imx-hero-left { padding: 20px 18px; }
            .imx-hero h1 { font-size: 26px; }
            .imx-models { grid-template-columns: 1fr 1fr; }
            .imx-caps-grid { grid-template-columns: 1fr; }
            .imx-thumbs .th { width: 60px; height: 60px; }
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

        /* ── Home page: dark futuristic AI-security theme (reference match) ── */
        .stApp { background: radial-gradient(1200px 500px at 20% -5%, rgba(47,123,255,.06), transparent 60%), #05080f; }
        [data-testid="stSidebar"] { background: #0a0f1d; border-right: 1px solid #141f33; }
        .side-brand { display: flex; align-items: center; gap: 10px; padding: 6px 2px 12px 2px; border-bottom: 1px solid #16223a; margin-bottom: 12px; }
        .side-brand .shield { width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0; background: linear-gradient(135deg, #12305e, #0a1a35); border: 1.5px solid #2f7bff; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 14px rgba(47,123,255,.4); }
        .side-brand h2 { color: #fff; margin: 0; font-size: 16px; font-weight: 800; letter-spacing: -.2px; }
        .side-brand p { color: #5a6b82; margin: 2px 0 0 0; font-size: 11px; }
        [data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label, [data-testid="stSidebar"] div[role="radiogroup"] > div { border: 1px solid transparent; border-radius: 10px; padding: 9px 12px; margin: 1px 0; transition: background .18s ease, border-color .18s ease; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover, [data-testid="stSidebar"] div[role="radiogroup"] > div:hover { background: rgba(47,123,255,.08); border-color: rgba(47,123,255,.18); }
        [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"], [data-testid="stSidebar"] div[role="radiogroup"] > div[data-checked="true"] { background: linear-gradient(135deg, #1d5cff 0%, #1a3a9e 100%) !important; border-color: rgba(80,140,255,.5) !important; box-shadow: 0 4px 16px rgba(29,92,255,.35); }
        [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p, [data-testid="stSidebar"] div[role="radiogroup"] div[data-checked="true"] p { color: #fff !important; font-weight: 700 !important; }
        [data-testid="stSidebar"] div[role="radiogroup"] p { color: #c4cfdf !important; font-size: 13.5px !important; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1) p::before, [data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(1) p::before { content: "⌂  "; font-weight: 700; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(2) p::before, [data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(2) p::before { content: "◫  "; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(3) p::before, [data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(3) p::before { content: "▶  "; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(4) p::before, [data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(4) p::before { content: "♫  "; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(5) p::before, [data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(5) p::before { content: "◉  "; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(6) p::before, [data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(6) p::before { content: "⊕  "; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(7) p::before, [data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(7) p::before { content: "◷  "; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(8) p::before, [data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(8) p::before { content: "▂▄  "; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(9) p::before, [data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(9) p::before { content: "ⓘ  "; }
        .side-thresh-label { color: #8b9bb0; font-size: 10.5px; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin: 0 0 2px 0; }
        .side-thresh-val { color: #ef4444; font-size: 12px; font-weight: 800; text-align: center; margin: 0 0 4px 0; }
        .side-thresh-hint { color: #6b7d95; font-size: 11.5px; line-height: 1.5; margin: 6px 0 0 0; }
        .side-device { color: #6b7d95; font-size: 12px; margin: 10px 0 0 0; }
        .side-device code { color: #22c55e; background: transparent; padding: 0; font-size: 12px; font-weight: 700; }
        .side-device .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #22c55e; margin-right: 6px; box-shadow: 0 0 8px rgba(34,197,94,.8); }
        [data-testid="stSidebar"] .stButton > button { background: #0f1729 !important; color: #c4cfdf !important; border: 1px solid #1e2f4a !important; border-radius: 10px !important; font-size: 12.5px !important; font-weight: 600 !important; }
        [data-testid="stSidebar"] .stButton > button:hover { border-color: #2f7bff !important; color: #fff !important; }
        [data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] { background-color: #ef4444 !important; border-color: #ef4444 !important; }

        .home-topbar { display: flex; justify-content: flex-end; align-items: center; gap: 10px; margin: 0 0 10px 0; }
        .home-topbar .deploy { color: #c4cfdf; font-size: 13px; font-weight: 600; }
        .home-topbar .dots { color: #8b9bb0; font-size: 16px; font-weight: 800; letter-spacing: 2px; }
        .home-hero { background: radial-gradient(800px 300px at 15% 0%, rgba(47,123,255,.18), transparent 60%), radial-gradient(700px 320px at 85% 30%, rgba(124,58,237,.16), transparent 60%), linear-gradient(120deg, #0b1230 0%, #0d1533 45%, #080d20 100%); border: 1px solid #1c3a5e; border-radius: 14px; margin-bottom: 16px; overflow: hidden; }
        .home-hero-inner { display: grid; grid-template-columns: 1.15fr .85fr; min-height: 340px; }
        .home-hero-left { padding: 30px 28px 26px 30px; }
        .home-hero-badge { display: inline-block; border: 1px solid #2a4a7a; color: #9db8d9; background: rgba(47,123,255,.08); padding: 5px 14px; border-radius: 20px; font-size: 10.5px; font-weight: 700; letter-spacing: 1.6px; margin-bottom: 14px; }
        .home-hero h1 { color: #fff; margin: 0 0 12px 0; font-size: 40px; line-height: 1.05; font-weight: 800; letter-spacing: -.8px; }
        .home-hero h1 .grad { background: linear-gradient(90deg, #3b9dff 0%, #5b7cff 45%, #7c5cff 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .home-hero .sub { color: #9fb0c7; margin: 0 0 20px 0; font-size: 14.5px; line-height: 1.6; max-width: 560px; }
        .home-cta-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }
        .home-cta { display: inline-flex; align-items: center; gap: 10px; padding: 12px 22px; border-radius: 10px; font-weight: 800; font-size: 14px; text-decoration: none; color: #fff !important; }
        .home-cta.primary { background: linear-gradient(135deg, #1e90ff 0%, #1a5cff 100%); border: 1px solid rgba(120,180,255,.6); box-shadow: 0 0 22px rgba(30,144,255,.55), inset 0 0 12px rgba(255,255,255,.12); }
        .home-cta.live { background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%); border: 1px solid rgba(180,140,255,.5); box-shadow: 0 0 22px rgba(124,58,237,.45); }
        .home-cta:hover { filter: brightness(1.1); text-decoration: none; color: #fff !important; }
        .home-trust { display: flex; gap: 18px; flex-wrap: wrap; }
        .home-trust .t { display: flex; align-items: center; gap: 8px; color: #a9bad1; font-size: 12.5px; font-weight: 600; }
        .home-trust .t .ic { width: 26px; height: 26px; border-radius: 7px; display: flex; align-items: center; justify-content: center; background: rgba(47,123,255,.12); border: 1px solid rgba(47,123,255,.25); font-size: 13px; }
        .home-hero-right { position: relative; min-height: 340px; overflow: hidden; background: linear-gradient(90deg, #0b1230 0%, rgba(11,18,48,.2) 22%, rgba(5,8,15,.15) 55%, rgba(5,8,15,.75) 100%), url("https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=800&q=80&auto=format&fit=crop") center 20%/cover no-repeat, radial-gradient(circle at 60% 40%, #16244d 0%, #070c18 70%); }
        .home-hero-right::after { content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(47,123,255,.09) 1px, transparent 1px), linear-gradient(90deg, rgba(47,123,255,.09) 1px, transparent 1px); background-size: 26px 26px; pointer-events: none; }
        .home-scanline { position: absolute; top: 0; bottom: 0; left: 50%; width: 2px; background: linear-gradient(180deg, transparent, #3b9dff 20%, #3b9dff 80%, transparent); box-shadow: 0 0 18px 2px rgba(59,157,255,.8); z-index: 3; }
        .home-badge-real { position: absolute; left: 12%; top: 58%; z-index: 4; background: rgba(6,40,20,.9); color: #22c55e; border: 1px solid rgba(34,197,94,.7); font-size: 12px; font-weight: 800; letter-spacing: 1px; padding: 6px 14px; border-radius: 7px; box-shadow: 0 0 16px rgba(34,197,94,.5); }
        .home-badge-fake { position: absolute; right: 32%; top: 58%; z-index: 4; background: rgba(45,8,12,.9); color: #ff6b6b; border: 1px solid rgba(239,68,68,.7); font-size: 12px; font-weight: 800; letter-spacing: 1px; padding: 6px 14px; border-radius: 7px; box-shadow: 0 0 16px rgba(239,68,68,.55); }
        .home-side-pills { position: absolute; right: 10px; top: 16px; bottom: 16px; z-index: 4; display: flex; flex-direction: column; gap: 10px; justify-content: center; }
        .home-side-pills .pill { display: flex; align-items: center; gap: 8px; background: rgba(10,18,35,.85); border: 1px solid #254a7a; color: #bcd0ea; font-size: 11.5px; font-weight: 800; letter-spacing: .6px; padding: 10px 14px; border-radius: 9px; min-width: 118px; }
        .home-status-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 0 0 18px 0; }
        .home-status-card { background: #0d1526; border: 1px solid #1a2b45; border-radius: 12px; padding: 14px; display: flex; gap: 12px; align-items: center; }
        .home-status-card .sicon { width: 46px; height: 46px; border-radius: 10px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 20px; }
        .home-status-card .lbl { color: #8b9bb0; font-size: 11.5px; font-weight: 600; margin: 0; }
        .home-status-card .val { color: #fff; font-size: 17px; font-weight: 800; margin: 2px 0; }
        .home-status-card .val.green { color: #22c55e; }
        .home-status-card .hint { color: #5f7191; font-size: 11px; margin: 0; }
        .home-section-head { display: flex; justify-content: space-between; align-items: center; margin: 6px 0 12px 0; }
        .home-section-head h2 { color: #fff; font-size: 20px; font-weight: 800; margin: 0; letter-spacing: -.3px; }
        .home-section-head span { color: #2f7bff; font-size: 12.5px; font-weight: 700; }
        .home-cap-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }
        a.home-cap { background: #0d1526; border: 1px solid #1a2b45; border-radius: 12px; padding: 16px; display: flex; gap: 12px; align-items: center; transition: all .18s ease; text-decoration: none !important; }
        a.home-cap:hover { border-color: #2f7bff; transform: translateY(-1px); box-shadow: 0 8px 24px rgba(0,0,0,.35); }
        .home-cap .cicon { width: 48px; height: 48px; border-radius: 10px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 22px; }
        .home-cap h4 { color: #fff; margin: 0 0 4px 0; font-size: 14px; font-weight: 700; }
        .home-cap p { color: #7c8da6; margin: 0; font-size: 12px; line-height: 1.5; }
        .home-cap .arrow { margin-left: auto; width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0; background: #182845; border: 1px solid #24406a; color: #9db8d9; display: flex; align-items: center; justify-content: center; font-size: 15px; font-weight: 700; }
        .home-how { background: #0a1120; border: 1px solid #16263f; border-radius: 12px; padding: 18px 18px 16px 18px; margin-bottom: 10px; }
        .home-how h2 { color: #fff; font-size: 20px; font-weight: 800; margin: 0 0 14px 0; }
        .home-steps { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr; gap: 8px; align-items: stretch; }
        .home-step { background: #0f1a2e; border: 1px solid #1d3454; border-radius: 10px; padding: 12px; display: flex; gap: 10px; align-items: flex-start; }
        .home-step.green { border-color: rgba(34,197,94,.4); }
        .home-step .num { width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0; background: #2f7bff; color: #fff; font-size: 12.5px; font-weight: 800; display: flex; align-items: center; justify-content: center; }
        .home-step.green .num { background: #22c55e; }
        .home-step .sic { width: 34px; height: 34px; border-radius: 8px; flex-shrink: 0; background: rgba(47,123,255,.12); border: 1px solid rgba(47,123,255,.25); display: flex; align-items: center; justify-content: center; font-size: 15px; }
        .home-step h5 { color: #e6edf7; margin: 0 0 3px 0; font-size: 12.5px; font-weight: 700; }
        .home-step p { color: #6f829e; margin: 0; font-size: 11.5px; line-height: 1.45; }
        .home-steps .arr { align-self: center; color: #2f7bff; font-size: 16px; font-weight: 800; }
        @media (max-width: 1100px) {
            .home-hero-inner { grid-template-columns: 1fr; }
            .home-hero-right { min-height: 280px; }
            .home-hero h1 { font-size: 32px; }
            .home-status-grid { grid-template-columns: repeat(3, 1fr); }
            .home-cap-grid { grid-template-columns: repeat(2, 1fr); }
            .home-steps { grid-template-columns: 1fr 1fr; }
            .home-steps .arr { display: none; }
        }
        @media (max-width: 640px) {
            .imx-hero-left { padding: 20px 18px; }
            .imx-hero h1 { font-size: 26px; }
            .imx-models { grid-template-columns: 1fr 1fr; }
            .imx-caps-grid { grid-template-columns: 1fr; }
            .imx-thumbs .th { width: 60px; height: 60px; }
        }

        /* ── Video Detection page: cyber-forensics (no human imagery) ── */
        .vdx-hero { background: radial-gradient(700px 260px at 20% 0%, rgba(124,58,237,.14), transparent 60%), radial-gradient(600px 280px at 85% 40%, rgba(47,123,255,.12), transparent 60%), linear-gradient(120deg, #0b1230 0%, #0d1734 50%, #080d20 100%); border: 1px solid #1c3a5e; border-radius: 14px; margin-bottom: 14px; overflow: hidden; }
        .vdx-hero-inner { display: grid; grid-template-columns: 1.05fr .95fr; min-height: 300px; }
        .vdx-hero-left { padding: 26px 26px 22px 28px; }
        .vdx-badge { display: inline-block; background: linear-gradient(135deg, #7c3aed, #4c1d95); color: #e9d5ff; padding: 5px 16px; border-radius: 8px; font-size: 10.5px; font-weight: 800; letter-spacing: 1.8px; margin-bottom: 12px; border: 1px solid rgba(168,85,247,.5); }
        .vdx-hero h1 { color: #fff; margin: 0 0 10px 0; font-size: 38px; line-height: 1.05; font-weight: 800; letter-spacing: -.8px; }
        .vdx-hero h1 .grad { background: linear-gradient(90deg, #a855f7 0%, #6366f1 50%, #3b9dff 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .vdx-hero .sub { color: #9fb0c7; margin: 0 0 18px 0; font-size: 14px; line-height: 1.6; max-width: 520px; }
        .vdx-trust { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; }
        .vdx-trust .t { display: flex; align-items: center; gap: 10px; }
        .vdx-trust .t .ic { width: 34px; height: 34px; border-radius: 9px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 16px; background: rgba(47,123,255,.1); border: 1px solid rgba(47,123,255,.25); }
        .vdx-trust .t b { display: block; color: #e6edf7; font-size: 12.5px; font-weight: 700; }
        .vdx-trust .t span { display: block; color: #6f829e; font-size: 11px; }
        .vdx-visual { position: relative; min-height: 300px; overflow: hidden; background: radial-gradient(circle at 60% 40%, #101c3a 0%, #070c18 70%); display: flex; align-items: center; gap: 10px; padding: 22px 16px; }
        .vdx-visual::after { content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(124,58,237,.07) 1px, transparent 1px), linear-gradient(90deg, rgba(124,58,237,.07) 1px, transparent 1px); background-size: 24px 24px; pointer-events: none; }
        .vdx-mini-player { width: 92px; height: 120px; border-radius: 10px; flex-shrink: 0; background: linear-gradient(160deg, #131f3d, #0a1122); border: 1px solid #2a4a7a; position: relative; z-index: 2; display: flex; align-items: center; justify-content: center; }
        .vdx-mini-player::before { content: "▶"; color: #7db4ff; font-size: 20px; background: rgba(47,123,255,.15); border: 1px solid rgba(47,123,255,.4); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }
        .vdx-player { flex: 1; border-radius: 12px; background: linear-gradient(160deg, #101b38, #080e1e); border: 1px solid #2a4a7a; box-shadow: 0 0 30px rgba(59,157,255,.18); position: relative; z-index: 2; overflow: hidden; }
        .vdx-player .chrome { display: flex; align-items: center; gap: 5px; padding: 8px 10px; border-bottom: 1px solid #1a2b45; }
        .vdx-player .chrome i { width: 7px; height: 7px; border-radius: 50%; background: #2a4a7a; }
        .vdx-player .screen { height: 110px; display: flex; align-items: center; justify-content: center; position: relative; background: radial-gradient(circle at 50% 60%, rgba(59,157,255,.18), transparent 70%); }
        .vdx-play { width: 56px; height: 56px; border-radius: 50%; background: radial-gradient(circle, #fff 0%, #cfe4ff 100%); color: #1a3a6e; font-size: 20px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 28px rgba(160,200,255,.7); z-index: 2; }
        .vdx-film { display: flex; gap: 5px; padding: 8px 10px; border-top: 1px solid #1a2b45; }
        .vdx-film span { flex: 1; height: 34px; border-radius: 5px; border: 1px solid #24406a; background: linear-gradient(160deg, #16244d 0%, #0b1428 100%); position: relative; overflow: hidden; }
        .vdx-film span::after { content: ""; position: absolute; left: 20%; right: 20%; top: 8px; height: 8px; border-radius: 4px; background: rgba(93,135,255,.35); box-shadow: 0 12px 0 -2px rgba(93,135,255,.2); }
        .vdx-player .controls { display: flex; align-items: center; gap: 8px; padding: 7px 10px; color: #7db4ff; font-size: 11px; }
        .vdx-player .controls .bar { flex: 1; height: 4px; border-radius: 2px; background: #1a2b45; position: relative; overflow: hidden; }
        .vdx-player .controls .bar::after { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 38%; border-radius: 2px; background: linear-gradient(90deg, #3b9dff, #a855f7); }
        .vdx-corners i { position: absolute; width: 20px; height: 20px; border: 2.5px solid #22d3ee; filter: drop-shadow(0 0 6px rgba(34,211,238,.9)); z-index: 3; }
        .vdx-corners i:nth-child(1) { left: 6px; top: 6px; border-right: none; border-bottom: none; border-radius: 6px 0 0 0; }
        .vdx-corners i:nth-child(2) { right: 6px; top: 6px; border-left: none; border-bottom: none; border-radius: 0 6px 0 0; }
        .vdx-corners i:nth-child(3) { left: 6px; bottom: 6px; border-right: none; border-top: none; border-radius: 0 0 0 6px; }
        .vdx-corners i:nth-child(4) { right: 6px; bottom: 6px; border-left: none; border-top: none; border-radius: 0 0 6px 0; }
        .vdx-side { width: 128px; flex-shrink: 0; z-index: 2; display: flex; flex-direction: column; gap: 8px; }
        .vdx-score { padding: 8px 6px; border-radius: 9px; text-align: center; font-size: 13px; font-weight: 800; line-height: 1.6; letter-spacing: .5px; }
        .vdx-score.real { background: rgba(6,40,20,.92); color: #22c55e; border: 1px solid rgba(34,197,94,.7); box-shadow: 0 0 14px rgba(34,197,94,.4); }
        .vdx-score.fake { background: rgba(45,8,12,.92); color: #ff6b6b; border: 1px solid rgba(239,68,68,.7); box-shadow: 0 0 14px rgba(239,68,68,.45); }
        .vdx-wave { background: rgba(10,18,35,.85); border: 1px solid #24406a; border-radius: 8px; padding: 8px 10px; }
        .vdx-chart { background: rgba(10,18,35,.85); border: 1px solid #24406a; border-radius: 8px; padding: 6px; }
        .vdx-panels { display: grid; grid-template-columns: 1.6fr 1fr; gap: 12px; margin-bottom: 14px; }
        .vdx-upload { background: #0d1526; border: 1px solid #1a2b45; border-radius: 12px; padding: 20px 20px 16px 20px; }
        .vdx-upload-head { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 14px; }
        .vdx-upload-head .uic { width: 44px; height: 44px; border-radius: 11px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 20px; background: rgba(47,123,255,.14); border: 1px solid rgba(47,123,255,.35); color: #7db4ff; }
        .vdx-upload-head h3 { color: #fff; margin: 0 0 4px 0; font-size: 16.5px; font-weight: 800; }
        .vdx-upload-head p { color: #7c8da6; margin: 0; font-size: 12.5px; }
        .vdx-drop { border: 1.5px dashed rgba(124,58,237,.4); border-radius: 12px; padding: 26px 18px 18px 18px; text-align: center; background: rgba(124,58,237,.03); }
        .vdx-drop .cloud { font-size: 30px; color: #5ea0ff; filter: drop-shadow(0 0 12px rgba(59,157,255,.6)); }
        .vdx-drop h4 { color: #fff; margin: 8px 0 4px 0; font-size: 15px; font-weight: 700; }
        .vdx-drop p { color: #5f7191; margin: 0 0 4px 0; font-size: 12.5px; }
        .vdx-formats { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
        .vdx-formats .fb { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 7px; font-weight: 800; font-size: 11.5px; }
        .vdx-formats .sep { width: 1px; align-self: stretch; background: #1a2b45; }
        .vdx-formats .max { color: #8b9bb0; font-size: 12px; }
        .vdx-foot { color: #5f7191; font-size: 12px; margin: 12px 0 0 0; }
        .vdx-config { background: #0d1526; border: 1px solid #1a2b45; border-radius: 12px; padding: 20px 18px 16px 18px; }
        .vdx-config-head { display: flex; gap: 10px; align-items: center; margin-bottom: 16px; }
        .vdx-config-head .gic { width: 36px; height: 36px; border-radius: 9px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 17px; background: rgba(47,123,255,.14); border: 1px solid rgba(47,123,255,.35); color: #7db4ff; }
        .vdx-config-head h3 { color: #fff; margin: 0; font-size: 15px; font-weight: 800; }
        .vdx-field { color: #e6edf7; font-size: 13px; font-weight: 700; margin: 0 0 2px 0; }
        .vdx-field-sub { color: #5f7191; font-size: 11.5px; margin: 0 0 6px 0; }
        .vdx-info { display: flex; gap: 10px; margin-top: 14px; padding: 12px; background: rgba(47,123,255,.06); border: 1px solid rgba(47,123,255,.22); border-radius: 10px; color: #7db4ff; font-size: 12px; line-height: 1.55; }
        .vdx-info b { color: #a9c8ff; }
        .vdx-caps { background: #0a1120; border: 1px solid #1c3a5e; border-radius: 12px; padding: 20px 20px 18px 20px; margin-bottom: 10px; }
        .vdx-caps h2 { color: #fff; font-size: 19px; font-weight: 800; margin: 0 0 4px 0; display: flex; align-items: center; gap: 10px; }
        .vdx-caps .sub { color: #7c8da6; font-size: 13px; margin: 0 0 16px 0; }
        .vdx-caps-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
        .vdx-cap { background: #0f1a2e; border: 1px solid #1d3454; border-radius: 12px; padding: 16px; display: flex; gap: 12px; align-items: flex-start; }
        .vdx-cap .cc { width: 50px; height: 50px; border-radius: 12px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 23px; }
        .vdx-cap h4 { color: #fff; margin: 0 0 4px 0; font-size: 13.5px; font-weight: 700; }
        .vdx-cap p { color: #7c8da6; margin: 0; font-size: 11.5px; line-height: 1.55; }
        @media (max-width: 1100px) {
            .vdx-hero-inner { grid-template-columns: 1fr; }
            .vdx-visual { min-height: 260px; }
            .vdx-hero h1 { font-size: 30px; }
            .vdx-panels { grid-template-columns: 1fr; }
            .vdx-caps-grid { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 640px) {
            .vdx-hero-left { padding: 20px 18px; }
            .vdx-hero h1 { font-size: 26px; }
            .vdx-trust { grid-template-columns: 1fr; }
            .vdx-caps-grid { grid-template-columns: 1fr; }
            .vdx-side { width: 108px; }
        }

        /* ── Audio Detection page: voice-forensics (no human imagery) ── */
        .adx-hero { background: radial-gradient(700px 260px at 20% 0%, rgba(47,123,255,.15), transparent 60%), radial-gradient(600px 280px at 85% 40%, rgba(124,58,237,.14), transparent 60%), linear-gradient(120deg, #0b1230 0%, #0d1734 50%, #080d20 100%); border: 1px solid #1c3a5e; border-radius: 14px; margin-bottom: 14px; overflow: hidden; }
        .adx-hero-inner { display: grid; grid-template-columns: 1fr 1fr; min-height: 300px; }
        .adx-hero-left { padding: 26px 26px 22px 28px; }
        .adx-badge { display: inline-flex; align-items: center; gap: 7px; background: linear-gradient(135deg, #7c3aed, #4c1d95); color: #e9d5ff; padding: 5px 16px; border-radius: 8px; font-size: 10.5px; font-weight: 800; letter-spacing: 1.8px; margin-bottom: 12px; border: 1px solid rgba(168,85,247,.5); }
        .adx-hero h1 { color: #fff; margin: 0 0 10px 0; font-size: 40px; line-height: 1.05; font-weight: 800; letter-spacing: -.8px; }
        .adx-hero h1 .grad { background: linear-gradient(90deg, #a855f7 0%, #6366f1 55%, #3b9dff 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .adx-hero .sub { color: #9fb0c7; margin: 0 0 18px 0; font-size: 13.5px; line-height: 1.6; max-width: 520px; }
        .adx-trust { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; }
        .adx-trust .t { display: flex; align-items: center; gap: 10px; }
        .adx-trust .t .ic { width: 34px; height: 34px; border-radius: 9px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 16px; background: rgba(47,123,255,.1); border: 1px solid rgba(47,123,255,.25); }
        .adx-trust .t b { display: block; color: #e6edf7; font-size: 12.5px; font-weight: 700; }
        .adx-trust .t span { display: block; color: #6f829e; font-size: 11px; }
        .adx-visual { position: relative; min-height: 300px; overflow: hidden; background: radial-gradient(circle at 50% 45%, #101c3a 0%, #070c18 72%); display: flex; align-items: center; justify-content: center; gap: 0; padding: 20px 12px; }
        .adx-visual::after { content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(124,58,237,.06) 1px, transparent 1px), linear-gradient(90deg, rgba(124,58,237,.06) 1px, transparent 1px); background-size: 24px 24px; pointer-events: none; }
        .adx-mic-wrap { position: relative; width: 190px; height: 190px; flex-shrink: 0; z-index: 2; display: flex; align-items: center; justify-content: center; }
        .adx-mic-wrap .ring { position: absolute; border-radius: 50%; }
        .adx-mic-wrap .r1 { inset: 0; border: 2px solid rgba(34,211,238,.55); box-shadow: 0 0 26px rgba(34,211,238,.35), inset 0 0 26px rgba(34,211,238,.12); }
        .adx-mic-wrap .r2 { inset: 14px; border: 1.5px solid rgba(59,157,255,.5); box-shadow: inset 0 0 20px rgba(59,157,255,.15); }
        .adx-mic-wrap .r3 { inset: 28px; border: 1.5px solid rgba(168,85,247,.5); background: radial-gradient(circle, rgba(30,20,60,.9) 0%, rgba(10,15,35,.9) 100%); box-shadow: 0 0 34px rgba(124,58,237,.45); }
        .adx-mic-wrap svg { position: relative; z-index: 2; }
        .adx-wave-side { z-index: 2; display: flex; align-items: center; }
        .adx-ai-chip { position: absolute; top: 16px; right: 148px; z-index: 3; background: rgba(10,18,35,.9); border: 1px solid #2a4a7a; color: #7db4ff; font-weight: 800; font-size: 17px; padding: 6px 12px; border-radius: 9px; box-shadow: 0 0 16px rgba(59,157,255,.3); }
        .adx-playerbar { position: absolute; left: 50%; transform: translateX(-50%); bottom: 14px; z-index: 3; display: flex; align-items: center; gap: 10px; background: rgba(10,18,35,.92); border: 1px solid #2a4a7a; border-radius: 11px; padding: 8px 14px; width: 62%; }
        .adx-playerbar .pp { width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0; background: radial-gradient(circle, #3b9dff, #1a5cff); color: #fff; font-size: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 14px rgba(59,157,255,.6); }
        .adx-playerbar .tm { color: #8b9bb0; font-size: 11px; font-weight: 600; white-space: nowrap; }
        .adx-scores { position: absolute; right: 10px; top: 14px; bottom: 14px; z-index: 3; display: flex; flex-direction: column; gap: 8px; justify-content: center; width: 128px; }
        .adx-score { padding: 10px 8px; border-radius: 10px; text-align: center; font-size: 13px; font-weight: 800; line-height: 1.6; letter-spacing: .5px; }
        .adx-score small { display: block; font-size: 9px; font-weight: 600; letter-spacing: 1px; opacity: .8; }
        .adx-score.human { background: rgba(6,40,20,.92); color: #22c55e; border: 1px solid rgba(34,197,94,.7); box-shadow: 0 0 14px rgba(34,197,94,.4); }
        .adx-score.fake { background: rgba(45,8,12,.92); color: #ff6b6b; border: 1px solid rgba(239,68,68,.7); box-shadow: 0 0 14px rgba(239,68,68,.45); }
        .adx-panels { display: grid; grid-template-columns: 1.6fr 1fr; gap: 12px; margin-bottom: 14px; }
        .adx-upload { background: #0d1526; border: 1px solid #1a2b45; border-radius: 12px; padding: 20px 20px 16px 20px; }
        .adx-upload-head { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 14px; }
        .adx-upload-head .uic { width: 44px; height: 44px; border-radius: 11px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 20px; background: rgba(47,123,255,.14); border: 1px solid rgba(47,123,255,.35); color: #7db4ff; }
        .adx-upload-head h3 { color: #fff; margin: 0 0 4px 0; font-size: 16.5px; font-weight: 800; }
        .adx-upload-head p { color: #7c8da6; margin: 0; font-size: 12.5px; }
        .adx-drop { border: 1.5px dashed rgba(47,123,255,.35); border-radius: 12px; padding: 26px 18px 18px 18px; text-align: center; background: rgba(47,123,255,.02); }
        .adx-drop .cloud { font-size: 30px; color: #5ea0ff; filter: drop-shadow(0 0 12px rgba(59,157,255,.6)); }
        .adx-drop h4 { color: #fff; margin: 8px 0 4px 0; font-size: 15px; font-weight: 700; }
        .adx-drop p { color: #5f7191; margin: 0 0 4px 0; font-size: 12.5px; }
        .adx-formats { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
        .adx-formats .fb { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 7px; font-weight: 800; font-size: 11.5px; }
        .adx-formats .sep { width: 1px; align-self: stretch; background: #1a2b45; }
        .adx-formats .max { color: #8b9bb0; font-size: 12px; }
        .adx-foot { color: #5f7191; font-size: 12px; margin: 12px 0 0 0; }
        .adx-detect-btn button { background: linear-gradient(135deg, #1e90ff 0%, #7c3aed 100%) !important; color: #fff !important; border: 1px solid rgba(140,170,255,.55) !important; border-radius: 10px !important; font-weight: 800 !important; font-size: 14px !important; padding: 10px 18px !important; box-shadow: 0 0 22px rgba(80,120,255,.45) !important; width: 100%; }
        .adx-config { background: #0d1526; border: 1px solid #1a2b45; border-radius: 12px; padding: 20px 18px 16px 18px; }
        .adx-config-head { display: flex; gap: 10px; align-items: center; margin-bottom: 14px; }
        .adx-config-head .gic { width: 36px; height: 36px; border-radius: 9px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 17px; background: rgba(47,123,255,.14); border: 1px solid rgba(47,123,255,.35); color: #7db4ff; }
        .adx-config-head h3 { color: #fff; margin: 0; font-size: 15px; font-weight: 800; }
        .adx-dur-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
        .adx-field { color: #e6edf7; font-size: 13px; font-weight: 700; margin: 0 0 2px 0; }
        .adx-field-sub { color: #5f7191; font-size: 11.5px; margin: 0 0 6px 0; }
        .adx-info { display: flex; gap: 10px; margin-top: 14px; padding: 12px; background: rgba(47,123,255,.06); border: 1px solid rgba(47,123,255,.22); border-radius: 10px; color: #7db4ff; font-size: 12px; line-height: 1.55; }
        .adx-info b { color: #a9c8ff; }
        .adx-caps { background: #0a1120; border: 1px solid #1c3a5e; border-radius: 12px; padding: 20px 20px 18px 20px; margin-bottom: 10px; }
        .adx-caps h2 { color: #fff; font-size: 19px; font-weight: 800; margin: 0 0 4px 0; display: flex; align-items: center; gap: 10px; }
        .adx-caps .sub { color: #7c8da6; font-size: 13px; margin: 0 0 16px 0; }
        .adx-caps-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
        .adx-cap { background: #0f1a2e; border: 1px solid #1d3454; border-radius: 12px; padding: 16px; display: flex; gap: 12px; align-items: flex-start; }
        .adx-cap .cc { width: 50px; height: 50px; border-radius: 12px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 23px; }
        .adx-cap h4 { color: #fff; margin: 0 0 4px 0; font-size: 13.5px; font-weight: 700; }
        .adx-cap p { color: #7c8da6; margin: 0; font-size: 11.5px; line-height: 1.55; }
        @media (max-width: 1100px) {
            .adx-hero-inner { grid-template-columns: 1fr; }
            .adx-visual { min-height: 260px; }
            .adx-hero h1 { font-size: 32px; }
            .adx-panels { grid-template-columns: 1fr; }
            .adx-caps-grid { grid-template-columns: 1fr 1fr; }
            .adx-ai-chip { right: 150px; }
        }
        @media (max-width: 640px) {
            .adx-hero-left { padding: 20px 18px; }
            .adx-hero h1 { font-size: 27px; }
            .adx-trust { grid-template-columns: 1fr; }
            .adx-caps-grid { grid-template-columns: 1fr; }
            .adx-scores { width: 104px; }
            .adx-playerbar { width: 78%; }
        }

        /* ── Webcam Detection page: live-security (no human imagery) ── */
        .wcx-hero { background: radial-gradient(700px 260px at 20% 0%, rgba(47,123,255,.15), transparent 60%), radial-gradient(600px 280px at 85% 40%, rgba(124,58,237,.14), transparent 60%), linear-gradient(120deg, #0b1230 0%, #0d1734 50%, #080d20 100%); border: 1px solid #1c3a5e; border-radius: 14px; margin-bottom: 14px; overflow: hidden; }
        .wcx-hero-inner { display: grid; grid-template-columns: 1fr 1fr; min-height: 300px; }
        .wcx-hero-left { padding: 26px 26px 22px 28px; }
        .wcx-badge { display: inline-flex; align-items: center; gap: 8px; background: linear-gradient(135deg, #7c3aed, #4c1d95); color: #e9d5ff; padding: 5px 16px; border-radius: 8px; font-size: 10.5px; font-weight: 800; letter-spacing: 1.8px; margin-bottom: 12px; border: 1px solid rgba(168,85,247,.5); }
        .wcx-badge .rd { width: 8px; height: 8px; border-radius: 50%; background: #ff5252; box-shadow: 0 0 10px rgba(255,82,82,.9); }
        .wcx-hero h1 { color: #fff; margin: 0 0 10px 0; font-size: 38px; line-height: 1.05; font-weight: 800; letter-spacing: -.8px; }
        .wcx-hero h1 .grad { background: linear-gradient(90deg, #3b9dff 0%, #7c5cff 60%, #3b9dff 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .wcx-hero .sub { color: #9fb0c7; margin: 0 0 18px 0; font-size: 13.5px; line-height: 1.6; max-width: 520px; }
        .wcx-trust { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; }
        .wcx-trust .t { display: flex; align-items: center; gap: 10px; }
        .wcx-trust .t .ic { width: 34px; height: 34px; border-radius: 9px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 16px; background: rgba(47,123,255,.1); border: 1px solid rgba(47,123,255,.25); }
        .wcx-trust .t b { display: block; color: #5ea0ff; font-size: 12.5px; font-weight: 700; }
        .wcx-trust .t span { display: block; color: #6f829e; font-size: 11px; }
        .wcx-visual { position: relative; min-height: 300px; overflow: hidden; background: radial-gradient(circle at 50% 45%, #101c3a 0%, #070c18 72%); display: flex; align-items: center; justify-content: center; padding: 18px 150px 46px 12px; }
        .wcx-visual::after { content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(59,157,255,.06) 1px, transparent 1px), linear-gradient(90deg, rgba(59,157,255,.06) 1px, transparent 1px); background-size: 24px 24px; pointer-events: none; }
        .wcx-scores { position: absolute; right: 10px; top: 14px; z-index: 3; display: flex; flex-direction: column; gap: 8px; width: 92px; }
        .wcx-score { padding: 8px 4px; border-radius: 9px; text-align: center; font-size: 12px; font-weight: 800; line-height: 1.6; }
        .wcx-score.real { background: rgba(6,40,20,.92); color: #22c55e; border: 1px solid rgba(34,197,94,.7); box-shadow: 0 0 14px rgba(34,197,94,.4); }
        .wcx-score.fake { background: rgba(45,8,12,.92); color: #ff6b6b; border: 1px solid rgba(239,68,68,.7); box-shadow: 0 0 14px rgba(239,68,68,.45); }
        .wcx-minis { position: absolute; right: 10px; top: 50%; z-index: 3; width: 92px; display: flex; flex-direction: column; gap: 8px; margin-top: 44px; }
        .wcx-mini { background: rgba(10,18,35,.9); border: 1px solid #2a4a7a; border-radius: 9px; padding: 6px; }
        .wcx-playbar { position: absolute; left: 12px; right: 112px; bottom: 12px; z-index: 3; display: flex; align-items: center; gap: 10px; background: rgba(10,18,35,.92); border: 1px solid #2a4a7a; border-radius: 11px; padding: 8px 14px; }
        .wcx-playbar .pp { width: 28px; height: 28px; border-radius: 8px; flex-shrink: 0; background: rgba(47,123,255,.15); border: 1px solid rgba(47,123,255,.4); color: #7db4ff; font-size: 12px; display: flex; align-items: center; justify-content: center; }
        .wcx-panels { display: grid; grid-template-columns: 1.55fr 1fr; gap: 12px; margin-bottom: 14px; align-items: start; }
        .wcx-card { background: #0d1526; border: 1px solid #1a2b45; border-radius: 12px; padding: 20px 20px 18px 20px; }
        .wcx-head { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 6px; }
        .wcx-head .hic { width: 44px; height: 44px; border-radius: 11px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 20px; background: rgba(47,123,255,.14); border: 1px solid rgba(47,123,255,.35); color: #7db4ff; }
        .wcx-head h3 { color: #fff; margin: 0 0 4px 0; font-size: 16.5px; font-weight: 800; }
        .wcx-head p { color: #7c8da6; margin: 0; font-size: 12.5px; line-height: 1.55; }
        .wcx-zone { border: 1px solid #1e2f4a; border-radius: 10px; padding: 20px 16px 14px 16px; text-align: center; background: #0b1322; margin-top: 12px; }
        .wcx-zone .pic { width: 52px; height: 52px; border-radius: 50%; margin: 0 auto 4px; display: flex; align-items: center; justify-content: center; }
        .wcx-zone .cap { color: #5f7191; font-size: 12px; margin: 8px 0 0 0; }
        .wcx-enrolled { background: #0b1322; border: 1px solid #1e2f4a; border-radius: 10px; padding: 14px; margin-top: 12px; text-align: center; }
        .wcx-ok { display: flex; align-items: center; justify-content: center; gap: 8px; background: rgba(34,197,94,.1); border: 1px solid rgba(34,197,94,.35); color: #4ade80; font-size: 12.5px; font-weight: 700; border-radius: 8px; padding: 9px 10px; margin-bottom: 12px; }
        .wcx-enrolled .lbl { color: #8b9bb0; font-size: 12px; margin: 10px 0 0 0; }
        .wcx-preview { position: relative; border: 1px solid #1e2f4a; border-radius: 10px; background: radial-gradient(circle at 50% 55%, #0e1830 0%, #070c18 75%); min-height: 190px; display: flex; align-items: center; justify-content: center; margin: 12px 0; overflow: hidden; }
        .wcx-preview i { position: absolute; width: 20px; height: 20px; border: 2px solid rgba(94,160,255,.7); }
        .wcx-preview i:nth-child(1) { left: 12px; top: 12px; border-right: none; border-bottom: none; border-radius: 5px 0 0 0; }
        .wcx-preview i:nth-child(2) { right: 12px; top: 12px; border-left: none; border-bottom: none; border-radius: 0 5px 0 0; }
        .wcx-preview i:nth-child(3) { left: 12px; bottom: 12px; border-right: none; border-top: none; border-radius: 0 0 0 5px; }
        .wcx-preview i:nth-child(4) { right: 12px; bottom: 12px; border-left: none; border-top: none; border-radius: 0 0 5px 0; }
        .wcx-preview .off { display: inline-flex; align-items: center; gap: 7px; background: rgba(0,0,0,.55); border: 1px solid rgba(255,82,82,.4); color: #ff7b7b; font-size: 11.5px; font-weight: 700; border-radius: 16px; padding: 5px 14px; }
        .wcx-preview .off .rd { width: 8px; height: 8px; border-radius: 50%; background: #ff5252; box-shadow: 0 0 8px rgba(255,82,82,.9); }
        .st-key-wc_start button { background: linear-gradient(90deg, #1e90ff 0%, #7c3aed 100%) !important; color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 800 !important; font-size: 14px !important; padding: 11px 18px !important; box-shadow: 0 0 22px rgba(80,120,255,.45) !important; }
        .st-key-wc_replace button { background: transparent !important; color: #c4cfdf !important; border: 1px solid #2a4a7a !important; border-radius: 9px !important; font-weight: 600 !important; }
        .st-key-profile_enrollment button { background: linear-gradient(135deg, #1e90ff 0%, #7c3aed 100%) !important; color: #fff !important; border: 1px solid rgba(140,170,255,.5) !important; border-radius: 10px !important; font-weight: 700 !important; box-shadow: 0 0 18px rgba(80,120,255,.4) !important; }
        .wcx-how { background: #0a1120; border: 1px solid #1c3a5e; border-radius: 12px; padding: 20px 20px 18px 20px; margin-bottom: 10px; }
        .wcx-how-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 14px; }
        .wcx-step { background: #0f1a2e; border: 1px solid #1d3454; border-radius: 12px; padding: 16px; display: flex; gap: 12px; align-items: flex-start; }
        .wcx-step .cc { width: 50px; height: 50px; border-radius: 12px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 23px; }
        .wcx-step h4 { color: #fff; margin: 0 0 4px 0; font-size: 13.5px; font-weight: 700; }
        .wcx-step p { color: #7c8da6; margin: 0; font-size: 11.5px; line-height: 1.55; }
        @media (max-width: 1100px) {
            .wcx-hero-inner { grid-template-columns: 1fr; }
            .wcx-visual { min-height: 260px; padding-right: 12px; }
            .wcx-hero h1 { font-size: 30px; }
            .wcx-panels { grid-template-columns: 1fr; }
            .wcx-how-grid { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 640px) {
            .wcx-hero-left { padding: 20px 18px; }
            .wcx-hero h1 { font-size: 26px; }
            .wcx-trust { grid-template-columns: 1fr; }
            .wcx-how-grid { grid-template-columns: 1fr; }
            .wcx-scores, .wcx-minis { display: none; }
        }

        /* ── Grad-CAM page: heat-visual explainability (no human imagery) ── */
        .gcam-hero { background: radial-gradient(700px 260px at 20% 0%, rgba(124,58,237,.16), transparent 60%), radial-gradient(600px 280px at 85% 40%, rgba(59,157,255,.12), transparent 60%), linear-gradient(120deg, #0b1230 0%, #0d1734 50%, #080d20 100%); border: 1px solid #1c3a5e; border-radius: 14px; margin-bottom: 14px; overflow: hidden; }
        .gcam-hero-inner { display: grid; grid-template-columns: 1.1fr .9fr; min-height: 300px; }
        .gcam-hero-left { padding: 26px 26px 22px 28px; }
        .gcam-badge { display: inline-flex; align-items: center; gap: 8px; background: linear-gradient(135deg, #7c3aed, #4c1d95); color: #e9d5ff; padding: 5px 16px; border-radius: 8px; font-size: 10.5px; font-weight: 800; letter-spacing: 1.8px; margin-bottom: 12px; border: 1px solid rgba(168,85,247,.5); }
        .gcam-hero h1 { color: #fff; margin: 0 0 10px 0; font-size: 40px; line-height: 1.05; font-weight: 800; letter-spacing: -.8px; }
        .gcam-hero h1 .grad { background: linear-gradient(90deg, #3b9dff 0%, #7c5cff 60%, #3b9dff 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .gcam-hero .sub { color: #fff; margin: 0 0 8px 0; font-size: 14.5px; font-weight: 700; }
        .gcam-hero .desc { color: #9fb0c7; margin: 0 0 18px 0; font-size: 13.5px; line-height: 1.6; max-width: 520px; }
        .gcam-trust { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; }
        .gcam-trust .t { display: flex; align-items: center; gap: 10px; }
        .gcam-trust .t .ic { width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 17px; background: rgba(47,123,255,.1); border: 1px solid rgba(47,123,255,.25); }
        .gcam-trust .t b { display: block; color: #5ea0ff; font-size: 12.5px; font-weight: 700; }
        .gcam-trust .t span { display: block; color: #6f829e; font-size: 11px; }
        .gcam-visual { position: relative; min-height: 300px; overflow: hidden; background: radial-gradient(circle at 50% 45%, #101c3a 0%, #070c18 72%); display: flex; align-items: center; justify-content: center; padding: 18px 150px 18px 12px; }
        .gcam-visual::after { content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(124,58,237,.07) 1px, transparent 1px), linear-gradient(90deg, rgba(124,58,237,.07) 1px, transparent 1px); background-size: 24px 24px; pointer-events: none; }
        .gcam-orb-wrap { position: relative; width: 230px; height: 230px; flex-shrink: 0; z-index: 2; }
        .gcam-orb { position: absolute; inset: 18px; border-radius: 50%; background: radial-gradient(circle at 42% 38%, #ffeb3b 0%, #ff9800 18%, #f44336 38%, #9c27b0 58%, #2196f3 76%, #0d47a1 100%); box-shadow: 0 0 50px rgba(244,67,54,.35), 0 0 100px rgba(124,58,237,.25); overflow: hidden; }
        .gcam-orb::after { content: ""; position: absolute; inset: 0; border-radius: 50%; background-image: linear-gradient(rgba(4,7,15,.55) 1px, transparent 1px), linear-gradient(90deg, rgba(4,7,15,.55) 1px, transparent 1px); background-size: 14px 14px; }
        .gcam-ring { position: absolute; border-radius: 50%; border: 1.5px solid rgba(59,157,255,.5); }
        .gcam-scan i { position: absolute; width: 22px; height: 22px; border: 2.5px solid #3b9dff; filter: drop-shadow(0 0 6px rgba(59,157,255,.9)); z-index: 3; }
        .gcam-scan i:nth-child(1) { left: 6px; top: 6px; border-right: none; border-bottom: none; border-radius: 6px 0 0 0; }
        .gcam-scan i:nth-child(2) { right: 6px; top: 6px; border-left: none; border-bottom: none; border-radius: 0 6px 0 0; }
        .gcam-scan i:nth-child(3) { left: 6px; bottom: 6px; border-right: none; border-top: none; border-radius: 0 0 0 6px; }
        .gcam-scan i:nth-child(4) { right: 6px; bottom: 6px; border-left: none; border-top: none; border-radius: 0 0 6px 0; }
        .gcam-ped { position: absolute; left: 50%; bottom: 8px; transform: translateX(-50%); width: 200px; height: 26px; z-index: 1; border-radius: 50%; border: 2px solid rgba(59,157,255,.5); box-shadow: 0 0 24px rgba(59,157,255,.3), inset 0 0 18px rgba(59,157,255,.15); }
        .gcam-side { position: absolute; right: 10px; top: 14px; bottom: 14px; width: 128px; z-index: 3; display: flex; flex-direction: column; gap: 8px; justify-content: center; }
        .gcam-panel { background: rgba(10,18,35,.9); border: 1px solid #2a4a7a; border-radius: 10px; padding: 10px 12px; color: #dbe4f0; font-size: 12px; font-weight: 700; }
        .gcam-panel .ln { height: 5px; border-radius: 3px; background: #1a2b45; margin-top: 7px; }
        .gcam-thumb { background: rgba(10,18,35,.9); border: 1px solid rgba(168,85,247,.5); border-radius: 10px; height: 110px; background-image: radial-gradient(circle at 50% 40%, #ffeb3b 0%, #ff9800 20%, #f44336 40%, #9c27b0 62%, #0d47a1 100%); box-shadow: 0 0 16px rgba(168,85,247,.35); }
        .gcam-panels { display: grid; grid-template-columns: 1.65fr 1fr; gap: 12px; margin-bottom: 14px; align-items: start; }
        .gcam-card { background: #0d1526; border: 1px solid #1a2b45; border-radius: 12px; padding: 20px 20px 18px 20px; }
        .gcam-head { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 6px; }
        .gcam-head .hic { width: 44px; height: 44px; border-radius: 11px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 20px; background: rgba(47,123,255,.14); border: 1px solid rgba(47,123,255,.35); color: #7db4ff; }
        .gcam-head h3 { color: #fff; margin: 0 0 4px 0; font-size: 16.5px; font-weight: 800; }
        .gcam-head p { color: #7c8da6; margin: 0; font-size: 12.5px; }
        .gcam-drop { border: 1.5px dashed rgba(47,123,255,.35); border-radius: 12px; padding: 26px 18px 16px 18px; text-align: center; background: rgba(47,123,255,.02); margin-top: 12px; }
        .gcam-drop .cloud { font-size: 30px; color: #5ea0ff; filter: drop-shadow(0 0 12px rgba(59,157,255,.6)); }
        .gcam-drop h4 { color: #fff; margin: 8px 0 4px 0; font-size: 15px; font-weight: 700; }
        .gcam-drop p { color: #5ea0ff; margin: 0 0 4px 0; font-size: 12.5px; }
        .gcam-meta { color: #5f7191; font-size: 12px; margin: 12px 0 0 0; text-align: center; }
        .gcam-meta .sep { margin: 0 8px; color: #24406a; }
        .gcam-ex-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }
        .gcam-ex { border: 1px solid #24406a; border-radius: 10px; overflow: hidden; background: #0a1322; }
        .gcam-ex .tile { height: 110px; }
        .gcam-ex .cap { color: #8b9bb0; font-size: 11px; font-weight: 600; text-align: center; padding: 7px 4px; }
        .gcam-results { background: #0a1120; border: 1px solid #1c3a5e; border-radius: 12px; padding: 20px 20px 18px 20px; margin-bottom: 10px; }
        .gcam-res-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }
        .gcam-slot { border: 1.5px dashed rgba(47,123,255,.3); border-radius: 10px; min-height: 150px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; padding: 22px 12px; text-align: center; }
        .gcam-slot .sic { font-size: 24px; color: #5ea0ff; opacity: .8; }
        .gcam-slot b { color: #dbe4f0; font-size: 13px; }
        .gcam-slot span { color: #5f7191; font-size: 12px; }
        .gcam-slot.filled { border-style: solid; padding: 10px; }
        .gcam-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 14px; }
        .gcam-stat { background: #0f1a2e; border: 1px solid #1d3454; border-radius: 12px; padding: 14px 16px; display: flex; gap: 12px; align-items: center; }
        .gcam-stat .cc { width: 48px; height: 48px; border-radius: 12px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 22px; }
        .gcam-stat .k { color: #e6edf7; font-size: 13px; font-weight: 700; margin: 0; }
        .gcam-stat .v { font-size: 17px; font-weight: 800; margin: 2px 0; }
        .gcam-stat .h { color: #5f7191; font-size: 11px; margin: 0; }
        .st-key-gradcam button { background: linear-gradient(90deg, #7c3aed 0%, #1e90ff 100%) !important; color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 800 !important; font-size: 14px !important; padding: 11px 34px !important; box-shadow: 0 0 22px rgba(124,58,237,.5) !important; }
        @media (max-width: 1100px) {
            .gcam-hero-inner { grid-template-columns: 1fr; }
            .gcam-visual { min-height: 260px; padding-right: 12px; }
            .gcam-hero h1 { font-size: 32px; }
            .gcam-panels { grid-template-columns: 1fr; }
            .gcam-res-grid { grid-template-columns: 1fr 1fr; }
            .gcam-stats { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 640px) {
            .gcam-hero-left { padding: 20px 18px; }
            .gcam-hero h1 { font-size: 27px; }
            .gcam-trust { grid-template-columns: 1fr; }
            .gcam-res-grid { grid-template-columns: 1fr; }
            .gcam-stats { grid-template-columns: 1fr; }
            .gcam-side { display: none; }
            .gcam-ex-grid { grid-template-columns: 1fr 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
# ---------------------------------------------------------------------------
def page_home(threshold: float) -> None:
    # ── Top bar ──
    st.markdown(
        '<div class="home-topbar"><span class="deploy">Deploy</span>'
        '<span class="dots">⋮</span></div>',
        unsafe_allow_html=True,
    )

    # ── Hero banner ──
    st.markdown(
        '<div class="home-hero"><div class="home-hero-inner">'
        '<div class="home-hero-left">'
        '<div class="home-hero-badge">AI SECURITY &nbsp;•&nbsp; DIGITAL FORENSICS</div>'
        '<h1>AI-Powered <span class="grad">Deepfake Detection</span></h1>'
        '<p class="sub">Detect whether an image, video, audio, or live webcam '
        'stream is authentic or manipulated using deep learning and '
        'explainable AI.</p>'
        '<div class="home-cta-row">'
        '<a class="home-cta primary" href="?nav=Image+Detection" target="_self">'
        '&#x1F5BC; &nbsp; Analyze Image &nbsp; →</a>'
        '<a class="home-cta live" href="?nav=Webcam+Detection" target="_self">'
        '&#x1F4F7; &nbsp; Start Live Detection &nbsp; →</a>'
        '</div>'
        '<div class="home-trust">'
        '<div class="t"><span class="ic">🛡</span> Accurate Analysis</div>'
        '<div class="t"><span class="ic">⚡</span> Explainable AI (Grad-CAM)</div>'
        '<div class="t"><span class="ic">🔒</span> Privacy Focused</div>'
        '<div class="t"><span class="ic">📊</span> Multi-Modal Support</div>'
        '</div>'
        '</div>'
        '<div class="home-hero-right">'
        '<div class="home-scanline"></div>'
        '<div class="home-badge-real">REAL</div>'
        '<div class="home-badge-fake">FAKE</div>'
        '<div class="home-side-pills">'
        '<div class="pill"><span>🖼</span> IMAGE</div>'
        '<div class="pill"><span>▶</span> VIDEO</div>'
        '<div class="pill"><span>♫</span> AUDIO</div>'
        '<div class="pill"><span>◉</span> WEBCAM</div>'
        '</div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Status cards (wired to live values) ──
    model_label = MODEL_NAME.replace("_", " ").title()
    device_label = DEVICE.upper()
    st.markdown(
        '<div class="home-status-grid">'
        '<div class="home-status-card">'
        '<div class="sicon" style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);color:#22c55e;">◉</div>'
        '<div><p class="lbl">System Status</p><p class="val green">Online</p>'
        '<p class="hint">All services running</p></div>'
        '</div>'
        '<div class="home-status-card">'
        '<div class="sicon" style="background:rgba(47,123,255,.12);border:1px solid rgba(47,123,255,.3);color:#5ea0ff;">▣</div>'
        f'<div><p class="lbl">Model</p><p class="val">{model_label}</p>'
        '<p class="hint">Deep learning model</p></div>'
        '</div>'
        '<div class="home-status-card">'
        '<div class="sicon" style="background:rgba(124,58,237,.14);border:1px solid rgba(124,58,237,.35);color:#a78bfa;">🖼</div>'
        f'<div><p class="lbl">Input Resolution</p><p class="val">{IMG_SIZE} × {IMG_SIZE} px</p>'
        '<p class="hint">Optimized for accuracy</p></div>'
        '</div>'
        '<div class="home-status-card">'
        '<div class="sicon" style="background:rgba(47,123,255,.12);border:1px solid rgba(47,123,255,.3);color:#5ea0ff;">💻</div>'
        f'<div><p class="lbl">Inference Device</p><p class="val">{device_label}</p>'
        '<p class="hint">Auto GPU/CPU support</p></div>'
        '</div>'
        '<div class="home-status-card">'
        '<div class="sicon" style="background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);color:#ff6b6b;">⇄</div>'
        f'<div><p class="lbl">Decision Threshold</p><p class="val">{threshold:.2f}</p>'
        '<p class="hint">Adjustable sensitivity</p></div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Core detection capabilities ──
    st.markdown(
        '<div class="home-section-head"><h2>Core Detection Capabilities</h2>'
        '<span>Explore All Features →</span></div>'
        '<div class="home-cap-grid">'
        '<a class="home-cap" href="?nav=Image+Detection" target="_self">'
        '<div class="cicon" style="background:rgba(34,197,94,.14);border:1px solid rgba(34,197,94,.3);color:#22c55e;">🖼</div>'
        '<div><h4>Image Detection</h4><p>Analyze images for potential visual manipulation.</p></div>'
        '<div class="arrow">→</div></a>'
        '<a class="home-cap" href="?nav=Video+Detection" target="_self">'
        '<div class="cicon" style="background:rgba(124,58,237,.16);border:1px solid rgba(124,58,237,.35);color:#a78bfa;">▶</div>'
        '<div><h4>Video Detection</h4><p>Perform frame-level and temporal deepfake analysis.</p></div>'
        '<div class="arrow">→</div></a>'
        '<a class="home-cap" href="?nav=Audio+Detection" target="_self">'
        '<div class="cicon" style="background:rgba(249,115,22,.14);border:1px solid rgba(249,115,22,.3);color:#fb923c;">♫</div>'
        '<div><h4>Audio Detection</h4><p>Analyze speech and audio for potential synthetic content.</p></div>'
        '<div class="arrow">→</div></a>'
        '<a class="home-cap" href="?nav=Webcam+Detection" target="_self">'
        '<div class="cicon" style="background:rgba(47,123,255,.14);border:1px solid rgba(47,123,255,.3);color:#5ea0ff;">◉</div>'
        '<div><h4>Live Webcam</h4><p>Real-time camera-based deepfake analysis.</p></div>'
        '<div class="arrow">→</div></a>'
        '<a class="home-cap" href="?nav=Grad-CAM" target="_self">'
        '<div class="cicon" style="background:rgba(239,68,68,.13);border:1px solid rgba(239,68,68,.3);color:#ff6b6b;">⊕</div>'
        '<div><h4>Grad-CAM</h4><p>Visualize image regions influencing model predictions.</p></div>'
        '<div class="arrow">→</div></a>'
        '<a class="home-cap" href="?nav=Dashboard" target="_self">'
        '<div class="cicon" style="background:rgba(234,179,8,.14);border:1px solid rgba(234,179,8,.3);color:#facc15;">📊</div>'
        '<div><h4>Analytics</h4><p>View prediction history and detection statistics.</p></div>'
        '<div class="arrow">→</div></a>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── How it works ──
    st.markdown(
        '<div class="home-how"><h2>How It Works</h2><div class="home-steps">'
        '<div class="home-step"><div class="num">01</div><div class="sic">⬆</div>'
        '<div><h5>Upload Media</h5><p>Provide an image, video, audio file, or live stream.</p></div></div>'
        '<div class="arr">→</div>'
        '<div class="home-step"><div class="num">02</div><div class="sic">⚙</div>'
        '<div><h5>AI Analysis</h5><p>Deep learning model analyzes the content.</p></div></div>'
        '<div class="arr">→</div>'
        '<div class="home-step"><div class="num">03</div><div class="sic">🔍</div>'
        '<div><h5>Evidence &amp; Explainability</h5><p>Get confidence scores and Grad-CAM visualization.</p></div></div>'
        '<div class="arr">→</div>'
        '<div class="home-step green"><div class="num">04</div>'
        '<div class="sic" style="background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.3);">✔</div>'
        '<div><h5>Authenticity Assessment</h5><p>Determine if the media is real or fake.</p></div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if not CHECKPOINT_PATH.exists():
        st.warning(
            f"Model checkpoint not found at `{CHECKPOINT_PATH}`. "
            "Detection pages will prompt you to add one."
        )


# ---------------------------------------------------------------------------
# Try-an-example male portraits for the Image Detection page (Unsplash CDN).
# ---------------------------------------------------------------------------
IMAGE_EXAMPLES = [
    {"file": "example-man-1.jpg",
     "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600&q=80&auto=format&fit=crop"},
    {"file": "example-man-2.jpg",
     "url": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=600&q=80&auto=format&fit=crop"},
    {"file": "example-man-3.jpg",
     "url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=600&q=80&auto=format&fit=crop"},
    {"file": "example-man-4.jpg",
     "url": "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=600&q=80&auto=format&fit=crop"},
]


@st.cache_data(show_spinner=False)
def _fetch_example_image(url: str) -> bytes:
    """Download an example portrait (cached). Raises on network failure."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


class _ExampleUpload:
    """Minimal UploadedFile stand-in so examples flow through run_image()."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self.size = len(data)
        self.type = "image/jpeg"

    def getvalue(self) -> bytes:
        return self._data

    def getbuffer(self) -> bytes:
        return self._data


def page_image(detector, threshold: float) -> None:
    from datetime import datetime

    # ── 1. HERO BANNER ──
    st.markdown(
        '<div class="home-topbar"><span class="deploy">Deploy</span>'
        '<span class="dots">⋮</span></div>'
        '<div class="imx-hero"><div class="imx-hero-inner">'
        '<div class="imx-hero-left">'
        '<div class="imx-badge">🖼 IMAGE ANALYSIS</div>'
        '<h1>Image <span class="blue">Deepfake</span> Detection</h1>'
        '<p class="sub">Analyze digital images for signs of synthetic or '
        'manipulated content using deep learning and explainable AI.</p>'
        '<div class="imx-trust">'
        '<div class="t"><span class="ic">🛡</span> Accurate Analysis</div>'
        '<div class="t"><span class="ic">⚡</span> Explainable AI (Grad-CAM)</div>'
        '<div class="t"><span class="ic">🔒</span> Privacy Focused</div>'
        '<div class="t"><span class="ic">📊</span> Fast &amp; Reliable</div>'
        '</div>'
        '</div>'
        '<div class="imx-hero-right">'
        '<div class="imx-scan"><i></i><i></i><i></i><i></i></div>'
        '<div class="imx-scores">'
        '<div class="imx-score real">REAL<br>0.12</div>'
        '<div class="imx-score fake">FAKE<br>0.88</div>'
        '</div>'
        '<div class="imx-thumbs">'
        '<div class="th gray"><img src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200&q=70&auto=format&fit=crop" alt="reference face"></div>'
        '<div class="th heat"><img src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=200&q=70&auto=format&fit=crop" alt="grad-cam preview"></div>'
        '</div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Model info cards ──
    device_short = "GPU" if DEVICE == "cuda" else "CPU"
    st.markdown(
        '<div class="imx-models">'
        '<div class="imx-model"><div class="mic">▣</div>'
        '<div><p class="k">Model</p><p class="v">'
        + MODEL_NAME.replace("_", " ").title()
        + '</p></div></div>'
        '<div class="imx-model"><div class="mic">🖼</div>'
        '<div><p class="k">Input Resolution</p><p class="v">'
        + str(IMG_SIZE) + " × " + str(IMG_SIZE)
        + '</p></div></div>'
        '<div class="imx-model"><div class="mic">⚙</div>'
        '<div><p class="k">Engine</p><p class="v">AI Vision</p></div></div>'
        '<div class="imx-model"><div class="mic">▣</div>'
        '<div><p class="k">Inference Device</p><p class="v">'
        + device_short
        + '</p></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── 2. UPLOAD + EXAMPLES ──
    # Example clicks arrive via ?example=N; consume once into session state.
    _qx = st.query_params.get("example", None)
    if isinstance(_qx, str) and _qx.isdigit() and int(_qx) in range(len(IMAGE_EXAMPLES)):
        st.session_state["img_example"] = int(_qx)
        st.query_params.clear()

    up_col, ex_col = st.columns([1.7, 1], gap="medium")
    with up_col:
        st.markdown(
            '<div class="imx-upload"><div class="cloud">☁⬆</div>'
            '<h3>Upload an Image</h3>'
            '<p class="d">Upload an image containing a visible face for '
            'AI-assisted forensic analysis.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        st.markdown(
            '<p class="imx-upload drag" style="text-align:center;">'
            'or drag and drop an image here</p>'
            '<div class="imx-formats">'
            '<span class="chip" style="background:rgba(168,85,247,.14);border:1px solid rgba(168,85,247,.35);color:#c084fc;">🖼 JPG</span>'
            '<span class="chip" style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);color:#4ade80;">🖼 JPEG</span>'
            '<span class="chip" style="background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.3);color:#fb923c;">🖼 PNG</span>'
            '<span class="sep"></span>'
            '<span>🗄 &nbsp; Max file size: 200 MB</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with ex_col:
        st.markdown(
            '<div class="imx-examples">'
            '<div class="eh">🖼 &nbsp; Try an Example</div>'
            '<div class="imx-ex-grid">'
            '<a class="imx-ex" href="?example=0" target="_self"><img src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&q=70&auto=format&fit=crop" alt="example male portrait 1"></a>'
            '<a class="imx-ex" href="?example=1" target="_self"><img src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400&q=70&auto=format&fit=crop" alt="example male portrait 2"></a>'
            '<a class="imx-ex" href="?example=2" target="_self"><img src="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&q=70&auto=format&fit=crop" alt="example male portrait 3"></a>'
            '<a class="imx-ex gray" href="?example=3" target="_self"><img src="https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=400&q=70&auto=format&fit=crop" alt="example male portrait 4"></a>'
            '</div>'
            '<div class="imx-ex-hint">👆 &nbsp; Click an example image to try</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Resolve the active file: fresh upload wins, else the chosen example.
    if uploaded is not None and "img_example" in st.session_state:
        st.session_state.pop("img_example", None)
    example_idx = st.session_state.get("img_example", None)
    if uploaded is None and isinstance(example_idx, int) and 0 <= example_idx < len(IMAGE_EXAMPLES):
        try:
            example_bytes = _fetch_example_image(IMAGE_EXAMPLES[example_idx]["url"])
            uploaded = _ExampleUpload(IMAGE_EXAMPLES[example_idx]["file"], example_bytes)
            st.caption(f"Using {IMAGE_EXAMPLES[example_idx]['file']} — analysis runs automatically.")
        except Exception as exc:
            st.error(f"Could not load the example image ({exc}). Try uploading a file instead.")
            st.session_state.pop("img_example", None)

    # ── 10. EMPTY STATE ──
    if uploaded is None:
        st.markdown(
            '<div class="imx-caps"><h2>✦ Analysis Capabilities</h2>'
            '<p class="sub">Our AI model analyzes multiple visual cues to detect '
            'manipulated or synthetic images.</p>'
            '<div class="imx-caps-grid">'
            '<div class="imx-cap purple"><div class="cc" style="background:rgba(168,85,247,.14);border:1px solid rgba(168,85,247,.35);color:#c084fc;">🖼</div>'
            '<div><h4>Image Analysis</h4><p>Detects visual inconsistencies and deepfake artifacts using Xception.</p></div>'
            '<div class="arrow">→</div></div>'
            '<div class="imx-cap green"><div class="cc" style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);color:#4ade80;">◉</div>'
            '<div><h4>Face Detection</h4><p>Automatic face localization and alignment for accurate analysis.</p></div>'
            '<div class="arrow">→</div></div>'
            '<div class="imx-cap orange"><div class="cc" style="background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.3);color:#fb923c;">◎</div>'
            '<div><h4>Model Explainability</h4><p>Grad-CAM visualization to highlight image regions influencing the prediction.</p></div>'
            '<div class="arrow">→</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return
    # Bytes for preview (example adapter is not a real UploadedFile).
    display_src = uploaded.getvalue() if isinstance(uploaded, _ExampleUpload) else uploaded
    if isinstance(uploaded, _ExampleUpload):
        if st.button("✕ Clear example image", key="img_example_clear"):
            st.session_state.pop("img_example", None)
            st.rerun()

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
        st.image(display_src, caption="Uploaded evidence", use_container_width=True)

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
            st.session_state.pop("img_example", None)
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
    # Page-scoped purple accents (uploader button, slider, dropdown).
    st.markdown(
        '<style>'
        'main div[data-testid="stFileUploader"] button {'
        'background: linear-gradient(135deg, #7c3aed 0%, #4c1d95 100%) !important;'
        'border: 1px solid rgba(180,140,255,.55) !important;'
        'box-shadow: 0 0 18px rgba(124,58,237,.5) !important;}'
        'main [data-baseweb="slider"] div[role="slider"] {'
        'background-color: #a855f7 !important; border-color: #a855f7 !important;}'
        'main div[data-baseweb="select"] > div {'
        'border-color: #24406a !important; background: #0f1a2e !important;'
        'border-radius: 10px !important;}'
        '</style>',
        unsafe_allow_html=True,
    )

    # ── Hero ──
    st.markdown(
        '<div class="home-topbar"><span class="deploy">Deploy</span>'
        '<span class="dots">⋮</span></div>'
        '<div class="vdx-hero"><div class="vdx-hero-inner">'
        '<div class="vdx-hero-left">'
        '<div class="vdx-badge">FORENSIC ANALYSIS</div>'
        '<h1>Video <span class="grad">Deepfake</span> Detection</h1>'
        '<p class="sub">Analyze uploaded video frames to identify potential '
        'synthetic or manipulated content with temporal consistency scoring.</p>'
        '<div class="vdx-trust">'
        '<div class="t"><span class="ic">⚡</span><div><b>Frame Analysis</b><span>AI-powered detection</span></div></div>'
        '<div class="t"><span class="ic">▤</span><div><b>Temporal Consistency</b><span>Sequence-level analysis</span></div></div>'
        '<div class="t"><span class="ic">🛡</span><div><b>High Accuracy</b><span>Reliable predictions</span></div></div>'
        '<div class="t"><span class="ic">📊</span><div><b>Detailed Insights</b><span>Frame-wise results</span></div></div>'
        '</div>'
        '</div>'
        '<div class="vdx-visual">'
        '<div class="vdx-mini-player"></div>'
        '<div class="vdx-player">'
        '<div class="vdx-corners"><i></i><i></i><i></i><i></i></div>'
        '<div class="chrome"><i></i><i></i><i></i></div>'
        '<div class="screen"><div class="vdx-play">▶</div></div>'
        '<div class="vdx-film"><span></span><span></span><span></span><span></span><span></span></div>'
        '<div class="controls"><span>▶</span><span class="bar"></span><span>00:16</span></div>'
        '</div>'
        '<div class="vdx-side">'
        '<div class="vdx-score real">REAL<br>0.12</div>'
        '<div class="vdx-score fake">FAKE<br>0.88</div>'
        '<div class="vdx-wave">'
        '<svg width="100%" height="44" viewBox="0 0 100 44" preserveAspectRatio="none">'
        '<g fill="#a855f7">'
        '<rect x="2" y="18" width="3" height="8" rx="1"/><rect x="8" y="14" width="3" height="16" rx="1"/>'
        '<rect x="14" y="10" width="3" height="24" rx="1"/><rect x="20" y="16" width="3" height="12" rx="1"/>'
        '<rect x="26" y="8" width="3" height="28" rx="1"/><rect x="32" y="12" width="3" height="20" rx="1"/>'
        '<rect x="38" y="6" width="3" height="32" rx="1"/><rect x="44" y="14" width="3" height="16" rx="1"/>'
        '<rect x="50" y="4" width="3" height="36" rx="1"/><rect x="56" y="12" width="3" height="20" rx="1"/>'
        '<rect x="62" y="8" width="3" height="28" rx="1"/><rect x="68" y="16" width="3" height="12" rx="1"/>'
        '<rect x="74" y="10" width="3" height="24" rx="1"/><rect x="80" y="18" width="3" height="8" rx="1"/>'
        '<rect x="86" y="14" width="3" height="16" rx="1"/><rect x="92" y="20" width="3" height="6" rx="1"/>'
        '</g></svg></div>'
        '<div class="vdx-chart">'
        '<svg width="100%" height="44" viewBox="0 0 100 44" preserveAspectRatio="none">'
        '<polyline points="0,36 12,30 24,32 36,20 48,24 60,12 72,16 84,8 100,10" fill="none" stroke="#3b9dff" stroke-width="2"/>'
        '<polygon points="0,36 12,30 24,32 36,20 48,24 60,12 72,16 84,8 100,10 100,44 0,44" fill="rgba(59,157,255,.18)"/>'
        '</svg></div>'
        '</div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Upload + Config side-by-side ──
    upload_col, config_col = st.columns([1.6, 1], gap="medium")

    with upload_col:
        st.markdown(
            '<div class="vdx-upload">'
            '<div class="vdx-upload-head"><div class="uic">⬆</div>'
            '<div><h3>Upload Video for Analysis</h3>'
            '<p>Upload a video file to detect potential deepfake or manipulated content.</p></div></div>'
            '<div class="vdx-drop"><div class="cloud">☁⬆</div>'
            '<h4>Drag and drop a video file here</h4>'
            '<p>or click to browse files</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Drag and drop or click to upload",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="vdx-formats">'
            '<span class="fb" style="background:rgba(47,123,255,.14);border:1px solid rgba(47,123,255,.35);color:#7db4ff;">🎞 MP4</span>'
            '<span class="fb" style="background:rgba(168,85,247,.14);border:1px solid rgba(168,85,247,.35);color:#c084fc;">🎞 AVI</span>'
            '<span class="fb" style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);color:#4ade80;">🎞 MOV</span>'
            '<span class="fb" style="background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.3);color:#fb923c;">🎞 MKV</span>'
            '<span class="sep"></span>'
            '<span class="max">🗄 &nbsp; Max file size: 200 MB</span>'
            '</div>'
            '<p class="vdx-foot">Supported formats: MP4, AVI, MOV, MKV &nbsp;•&nbsp; Maximum file size: 200 MB</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with config_col:
        st.markdown(
            '<div class="vdx-config">'
            '<div class="vdx-config-head"><div class="gic">⚙</div>'
            '<h3>Analysis Configuration</h3></div>'
            '<p class="vdx-field">Frames to Sample</p>'
            '<p class="vdx-field-sub">Number of frames to analyze from the video</p>',
            unsafe_allow_html=True,
        )
        max_frames = st.slider(
            "Frames to sample",
            4,
            32,
            VIDEO_MAX_FRAMES,
            label_visibility="collapsed",
        )
        st.markdown(
            '<p class="vdx-field" style="margin-top:12px;">Analysis Mode</p>'
            '<p class="vdx-field-sub">Choose the detection sensitivity level</p>',
            unsafe_allow_html=True,
        )
        mode = st.selectbox(
            "Analysis Mode",
            ["Strict", "Balanced (Recommended)", "Lenient"],
            index=1,
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="vdx-info">ⓘ &nbsp; <span><b>More sampled frames</b> provide '
            'broader temporal analysis but may increase processing time.</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Sensitivity preset offsets the sidebar threshold; Balanced keeps it exact.
    mode_offset = {"Strict": -0.05, "Balanced (Recommended)": 0.0, "Lenient": 0.05}[mode]
    eff_threshold = min(0.95, max(0.05, threshold + mode_offset))

    if uploaded is None:
        st.markdown(
            '<div class="vdx-caps"><h2>✦ Video Analysis Capabilities</h2>'
            '<p class="sub">Our AI model analyzes both spatial and temporal '
            'features to detect manipulated content in videos.</p>'
            '<div class="vdx-caps-grid">'
            '<div class="vdx-cap"><div class="cc" style="background:rgba(168,85,247,.14);border:1px solid rgba(168,85,247,.35);color:#c084fc;">▣</div>'
            '<div><h4>Frame Analysis</h4><p>Analyzes individual frames for visual inconsistencies.</p></div></div>'
            '<div class="vdx-cap"><div class="cc" style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);color:#4ade80;">◷</div>'
            '<div><h4>Temporal Consistency</h4><p>Detects unnatural temporal patterns across frames.</p></div></div>'
            '<div class="vdx-cap"><div class="cc" style="background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.3);color:#fb923c;">〜</div>'
            '<div><h4>Feature Extraction</h4><p>Extracts deep visual features using Xception model.</p></div></div>'
            '<div class="vdx-cap"><div class="cc" style="background:rgba(47,123,255,.14);border:1px solid rgba(47,123,255,.35);color:#7db4ff;">▂▄</div>'
            '<div><h4>Confidence Scoring</h4><p>Provides frame-wise and overall prediction scores.</p></div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
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
        fake_frames = sum(1 for p in frame_probabilities if float(p) >= eff_threshold)
    if real_frames is None:
        real_frames = sum(1 for p in frame_probabilities if float(p) < eff_threshold)

    label = label_from_prob(probability_fake, eff_threshold)
    conf = confidence_from_prob(probability_fake, eff_threshold)

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
        f'Threshold: <strong>{eff_threshold:.2f}</strong> &nbsp;&middot;&nbsp; '
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


def page_audio(detector, threshold: float) -> None:
    """AI voice detector: REAL (human) vs FAKE (AI-generated / cloned)."""
    import tempfile
    from pathlib import Path

    import numpy as np

    from src.audio_inference import (
        ALLOWED_AUDIO_EXT,
        AUDIO_SAMPLE_RATE,
        AudioInferenceError,
    )
    from src.config import confidence_from_prob, label_from_prob

    # Page-scoped purple accents (uploader button, slider, dropdown).
    st.markdown(
        '<style>'
        'main div[data-testid="stFileUploader"] button {'
        'background: linear-gradient(135deg, #1e90ff 0%, #7c3aed 100%) !important;'
        'border: 1px solid rgba(140,170,255,.55) !important;'
        'box-shadow: 0 0 18px rgba(80,120,255,.5) !important;}'
        'main [data-baseweb="slider"] div[role="slider"] {'
        'background-color: #a855f7 !important; border-color: #a855f7 !important;}'
        'main div[data-baseweb="select"] > div {'
        'border-color: #24406a !important; background: #0f1a2e !important;'
        'border-radius: 10px !important;}'
        '</style>',
        unsafe_allow_html=True,
    )

    # ── Hero ──
    st.markdown(
        '<div class="home-topbar"><span class="deploy">Deploy</span>'
        '<span class="dots">⋮</span></div>'
        '<div class="adx-hero"><div class="adx-hero-inner">'
        '<div class="adx-hero-left">'
        '<div class="adx-badge">◉ FORENSIC ANALYSIS</div>'
        '<h1>Audio <span class="grad">Detection</span></h1>'
        '<p class="sub">Upload a voice clip to check whether it sounds human '
        'or AI-generated (text-to-speech / voice cloning). Short clips work — '
        'clips under 4 seconds are loop-extended and longer clips are '
        'averaged over overlapping 4-second windows.</p>'
        '<div class="adx-trust">'
        '<div class="t"><span class="ic">🎙</span><div><b>AI-Powered</b><span>Analysis</span></div></div>'
        '<div class="t"><span class="ic">🎵</span><div><b>Detects</b><span>Synthetic Audio</span></div></div>'
        '<div class="t"><span class="ic">🛡</span><div><b>High</b><span>Accuracy</span></div></div>'
        '<div class="t"><span class="ic">📊</span><div><b>Detailed</b><span>Insights</span></div></div>'
        '</div>'
        '</div>'
        '<div class="adx-visual">'
        '<div class="adx-wave-side">'
        '<svg width="120" height="90" viewBox="0 0 120 90">'
        '<g fill="#3b9dff">'
        '<rect x="4" y="36" width="4" height="18" rx="2"/><rect x="12" y="28" width="4" height="34" rx="2"/>'
        '<rect x="20" y="20" width="4" height="50" rx="2"/><rect x="28" y="32" width="4" height="26" rx="2"/>'
        '<rect x="36" y="14" width="4" height="62" rx="2"/><rect x="44" y="26" width="4" height="38" rx="2"/>'
        '<rect x="52" y="8" width="4" height="74" rx="2"/><rect x="60" y="24" width="4" height="42" rx="2"/>'
        '<rect x="68" y="16" width="4" height="58" rx="2"/><rect x="76" y="30" width="4" height="30" rx="2"/>'
        '<rect x="84" y="22" width="4" height="46" rx="2"/><rect x="92" y="34" width="4" height="22" rx="2"/>'
        '<rect x="100" y="28" width="4" height="34" rx="2"/><rect x="108" y="38" width="4" height="14" rx="2"/>'
        '</g></svg></div>'
        '<div class="adx-mic-wrap">'
        '<div class="ring r1"></div><div class="ring r2"></div><div class="ring r3"></div>'
        '<svg width="64" height="64" viewBox="0 0 64 64" fill="none">'
        '<defs><linearGradient id="adxmic" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#3b9dff"/><stop offset="1" stop-color="#a855f7"/>'
        '</linearGradient></defs>'
        '<rect x="24" y="4" width="16" height="26" rx="8" fill="url(#adxmic)"/>'
        '<path d="M16 28a16 16 0 0 0 32 0" stroke="url(#adxmic)" stroke-width="4" stroke-linecap="round"/>'
        '<line x1="32" y1="44" x2="32" y2="54" stroke="url(#adxmic)" stroke-width="4" stroke-linecap="round"/>'
        '<line x1="22" y1="54" x2="42" y2="54" stroke="url(#adxmic)" stroke-width="4" stroke-linecap="round"/>'
        '</svg></div>'
        '<div class="adx-wave-side">'
        '<svg width="120" height="90" viewBox="0 0 120 90">'
        '<g fill="#a855f7">'
        '<rect x="4" y="38" width="4" height="14" rx="2"/><rect x="12" y="28" width="4" height="34" rx="2"/>'
        '<rect x="20" y="22" width="4" height="46" rx="2"/><rect x="28" y="30" width="4" height="30" rx="2"/>'
        '<rect x="36" y="16" width="4" height="58" rx="2"/><rect x="44" y="24" width="4" height="42" rx="2"/>'
        '<rect x="52" y="8" width="4" height="74" rx="2"/><rect x="60" y="14" width="4" height="62" rx="2"/>'
        '<rect x="68" y="26" width="4" height="38" rx="2"/><rect x="76" y="20" width="4" height="50" rx="2"/>'
        '<rect x="84" y="32" width="4" height="26" rx="2"/><rect x="92" y="22" width="4" height="46" rx="2"/>'
        '<rect x="100" y="34" width="4" height="22" rx="2"/><rect x="108" y="28" width="4" height="34" rx="2"/>'
        '</g></svg></div>'
        '<div class="adx-ai-chip">AI</div>'
        '<div class="adx-playerbar"><span class="pp">▶</span>'
        '<svg width="100%" height="30" viewBox="0 0 200 30" preserveAspectRatio="none">'
        '<g fill="#3b9dff">'
        '<rect x="2" y="11" width="2" height="8" rx="1"/><rect x="8" y="8" width="2" height="14" rx="1"/>'
        '<rect x="14" y="5" width="2" height="20" rx="1"/><rect x="20" y="9" width="2" height="12" rx="1"/>'
        '<rect x="26" y="3" width="2" height="24" rx="1"/><rect x="32" y="7" width="2" height="16" rx="1"/>'
        '<rect x="38" y="10" width="2" height="10" rx="1"/><rect x="44" y="6" width="2" height="18" rx="1"/>'
        '<rect x="50" y="11" width="2" height="8" rx="1"/><rect x="56" y="8" width="2" height="14" rx="1"/>'
        '<rect x="62" y="4" width="2" height="22" rx="1"/><rect x="68" y="9" width="2" height="12" rx="1"/>'
        '<rect x="74" y="6" width="2" height="18" rx="1"/><rect x="80" y="11" width="2" height="8" rx="1"/>'
        '<rect x="86" y="8" width="2" height="14" rx="1"/><rect x="92" y="5" width="2" height="20" rx="1"/>'
        '<rect x="98" y="10" width="2" height="10" rx="1"/><rect x="104" y="7" width="2" height="16" rx="1"/>'
        '<rect x="110" y="11" width="2" height="8" rx="1"/><rect x="116" y="8" width="2" height="14" rx="1"/>'
        '<rect x="122" y="5" width="2" height="20" rx="1"/><rect x="128" y="9" width="2" height="12" rx="1"/>'
        '<rect x="134" y="6" width="2" height="18" rx="1"/><rect x="140" y="11" width="2" height="8" rx="1"/>'
        '<rect x="146" y="8" width="2" height="14" rx="1"/><rect x="152" y="4" width="2" height="22" rx="1"/>'
        '<rect x="158" y="9" width="2" height="12" rx="1"/><rect x="164" y="6" width="2" height="18" rx="1"/>'
        '<rect x="170" y="11" width="2" height="8" rx="1"/><rect x="176" y="8" width="2" height="14" rx="1"/>'
        '<rect x="182" y="10" width="2" height="10" rx="1"/><rect x="188" y="7" width="2" height="16" rx="1"/>'
        '<rect x="194" y="11" width="2" height="8" rx="1"/>'
        '</g></svg><span class="tm">00:08 / 00:15</span></div>'
        '<div class="adx-scores">'
        '<div class="adx-score human"><small>HUMAN</small>0.12<br>'
        '<svg width="60" height="20" viewBox="0 0 60 20"><g fill="#22c55e">'
        '<rect x="2" y="8" width="3" height="5" rx="1"/><rect x="9" y="5" width="3" height="10" rx="1"/>'
        '<rect x="16" y="2" width="3" height="16" rx="1"/><rect x="23" y="6" width="3" height="8" rx="1"/>'
        '<rect x="30" y="3" width="3" height="14" rx="1"/><rect x="37" y="7" width="3" height="6" rx="1"/>'
        '<rect x="44" y="4" width="3" height="12" rx="1"/><rect x="51" y="8" width="3" height="5" rx="1"/>'
        '</g></svg></div>'
        '<div class="adx-score fake"><small>FAKE</small>0.88<br>'
        '<svg width="60" height="20" viewBox="0 0 60 20"><g fill="#ff6b6b">'
        '<rect x="2" y="7" width="3" height="6" rx="1"/><rect x="9" y="4" width="3" height="12" rx="1"/>'
        '<rect x="16" y="6" width="3" height="8" rx="1"/><rect x="23" y="2" width="3" height="16" rx="1"/>'
        '<rect x="30" y="5" width="3" height="10" rx="1"/><rect x="37" y="3" width="3" height="14" rx="1"/>'
        '<rect x="44" y="7" width="3" height="6" rx="1"/><rect x="51" y="4" width="3" height="12" rx="1"/>'
        '</g></svg></div>'
        '</div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Upload + Config side-by-side ──
    up_col, cfg_col = st.columns([1.6, 1], gap="medium")
    with up_col:
        st.markdown(
            '<div class="adx-upload">'
            '<div class="adx-upload-head"><div class="uic">⬆</div>'
            '<div><h3>Upload Audio for Analysis</h3>'
            '<p>Upload a voice clip to check whether it sounds human or AI-generated.</p></div></div>'
            '<div class="adx-drop"><div class="cloud">☁⬆</div>'
            '<h4>Drag and drop an audio file here</h4>'
            '<p>or click to browse files</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload audio",
            type=[e.lstrip(".") for e in sorted(ALLOWED_AUDIO_EXT)],
            label_visibility="collapsed",
            help="Supported: wav, mp3, m4a, flac, ogg",
        )
        st.markdown(
            '<div class="adx-formats">'
            '<span class="fb" style="background:rgba(168,85,247,.14);border:1px solid rgba(168,85,247,.35);color:#c084fc;">🎵 FLAC</span>'
            '<span class="fb" style="background:rgba(47,123,255,.14);border:1px solid rgba(47,123,255,.35);color:#7db4ff;">🎵 M4A</span>'
            '<span class="fb" style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);color:#4ade80;">🎵 MP3</span>'
            '<span class="fb" style="background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.3);color:#fb923c;">🎵 OGG</span>'
            '<span class="fb" style="background:rgba(47,123,255,.14);border:1px solid rgba(47,123,255,.35);color:#7db4ff;">🎵 WAV</span>'
            '<span class="sep"></span>'
            '<span class="max">🗄 &nbsp; Max file size: 200 MB</span>'
            '</div>'
            '<p class="adx-foot">Supported formats: FLAC, M4A, MP3, OGG, WAV &nbsp;•&nbsp; Maximum file size: 200 MB</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with cfg_col:
        st.markdown(
            '<div class="adx-config">'
            '<div class="adx-config-head"><div class="gic">⚙</div>'
            '<h3>Analysis Configuration</h3></div>'
            '<div class="adx-dur-row"><div>'
            '<p class="adx-field">Audio Duration</p>'
            '<p class="adx-field-sub">Short clips (&lt; 4s) are loop-extended.</p>'
            '</div>'
            '<svg width="90" height="34" viewBox="0 0 90 34">'
            '<g fill="#a855f7">'
            '<rect x="2" y="13" width="3" height="8" rx="1"/><rect x="9" y="9" width="3" height="16" rx="1"/>'
            '<rect x="16" y="5" width="3" height="24" rx="1"/><rect x="23" y="11" width="3" height="12" rx="1"/>'
            '<rect x="30" y="3" width="3" height="28" rx="1"/><rect x="37" y="9" width="3" height="16" rx="1"/>'
            '<rect x="44" y="6" width="3" height="22" rx="1"/><rect x="51" y="12" width="3" height="10" rx="1"/>'
            '<rect x="58" y="8" width="3" height="18" rx="1"/><rect x="65" y="13" width="3" height="8" rx="1"/>'
            '<rect x="72" y="9" width="3" height="16" rx="1"/><rect x="79" y="14" width="3" height="6" rx="1"/>'
            '</g></svg></div>'
            '<p class="adx-field">Clipping Window (seconds)</p>',
            unsafe_allow_html=True,
        )
        window_sec = st.slider(
            "Clipping window in seconds", 2, 8, 4,
            label_visibility="collapsed",
            help="Length of each overlapping analysis window. Shorter windows "
            "localize synthetic segments better; longer windows add context.",
        )
        st.markdown(
            '<p class="adx-field" style="margin-top:12px;">Analysis Mode</p>'
            '<p class="adx-field-sub">Choose the detection sensitivity level.</p>',
            unsafe_allow_html=True,
        )
        mode = st.selectbox(
            "Analysis Mode",
            ["Strict", "Balanced (Recommended)", "Lenient"],
            index=1,
            label_visibility="collapsed",
        )
        denoise = st.checkbox(
            "Remove background noise/music before analysis",
            value=False,
            help="Spectral-gating denoise (needs the noisereduce package). "
            "Music separation (demucs) is skipped when not installed.",
        )
        st.markdown(
            '<div class="adx-info">ⓘ &nbsp; <span><b>Longer audio clips</b> provide '
            'more reliable predictions but may increase processing time.</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Sensitivity preset offsets the sidebar threshold; Balanced keeps it exact.
    mode_offset = {"Strict": -0.05, "Balanced (Recommended)": 0.0, "Lenient": 0.05}[mode]
    eff_threshold = min(0.95, max(0.05, threshold + mode_offset))

    if uploaded is None:
        st.markdown(
            '<div class="adx-caps"><h2>✦ Audio Analysis Capabilities</h2>'
            '<p class="sub">Our AI model analyzes multiple audio cues to detect '
            'synthetic or manipulated speech.</p>'
            '<div class="adx-caps-grid">'
            '<div class="adx-cap"><div class="cc" style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);color:#4ade80;">🎙</div>'
            '<div><h4>Spectral Analysis</h4><p>Analyzes frequency patterns and spectral features.</p></div></div>'
            '<div class="adx-cap"><div class="cc" style="background:rgba(168,85,247,.14);border:1px solid rgba(168,85,247,.35);color:#c084fc;">⁂</div>'
            '<div><h4>Voice Pattern Detection</h4><p>Detects unnatural voice cloning or synthesis artifacts.</p></div></div>'
            '<div class="adx-cap"><div class="cc" style="background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.3);color:#fb923c;">〜</div>'
            '<div><h4>Temporal Consistency</h4><p>Analyzes timing and sequence-level features.</p></div></div>'
            '<div class="adx-cap"><div class="cc" style="background:rgba(47,123,255,.14);border:1px solid rgba(47,123,255,.35);color:#7db4ff;">▂▄</div>'
            '<div><h4>Confidence Scoring</h4><p>Provides detailed analysis with probability scores.</p></div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    st.audio(uploaded)
    if not st.button("▶ &nbsp; Detect Now", type="primary", key="audio_detect"):
        return

    # Respect the sidebar threshold (plus the sensitivity preset above).
    detector.threshold = float(eff_threshold)
    try:
        with st.spinner("Analyzing audio..."):
            result = run_audio(detector, uploaded, denoise=denoise,
                               window_sec=float(window_sec))
    except AudioInferenceError as exc:
        st.error(f"⚠️ {exc}")
        return

    probability_fake = float(
        result.get("probability_fake", result.get("fake_probability", 0.0))
    )
    inference_time = float(result.get("inference_time", 0.0))
    label = label_from_prob(probability_fake, eff_threshold)
    conf = confidence_from_prob(probability_fake, eff_threshold)

    verdict_banner(probability_fake, eff_threshold)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{conf * 100:.1f}%")
    m2.metric("Fake probability", f"{probability_fake:.4f}")
    m3.metric("Inference time", f"{inference_time * 1000:.0f} ms")
    m4.metric("Duration", f"{float(result.get('duration_sec', 0.0)):.1f} s")
    st.caption(
        f"Verdict: {label} · {int(result.get('n_windows', 1))} × "
        f"{float(result.get('window_sec', 4.0)):.0f}s window(s) averaged · "
        f"{result.get('denoise_note', '')}"
    )

    # Waveform + mel-spectrogram from the uploaded bytes (temp file so that
    # mp3/m4a decode the same way as inference).
    suffix = Path(uploaded.name).suffix or ".wav"
    tmp_path = None
    try:
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        y, _ = librosa.load(tmp_path, sr=AUDIO_SAMPLE_RATE, mono=True)

        fig_w, ax_w = plt.subplots(figsize=(8, 2.2))
        ax_w.plot(y, linewidth=0.6)
        ax_w.set_title("Waveform (16 kHz mono)")
        ax_w.set_xlabel("Sample")
        ax_w.set_ylabel("Amplitude")
        fig_w.tight_layout()
        st.pyplot(fig_w)
        plt.close(fig_w)

        mel = librosa.feature.melspectrogram(y=y, sr=AUDIO_SAMPLE_RATE)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        fig_m, ax_m = plt.subplots(figsize=(8, 2.6))
        img = librosa.display.specshow(
            mel_db, sr=AUDIO_SAMPLE_RATE, x_axis="time",
            y_axis="mel", ax=ax_m,
        )
        ax_m.set_title("Mel spectrogram")
        fig_m.colorbar(img, ax=ax_m, format="%+2.0f dB")
        fig_m.tight_layout()
        st.pyplot(fig_m)
        plt.close(fig_m)
    except Exception as exc:  # plots are best-effort; verdict above stands
        st.caption(f"Could not render audio plots: {exc}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    # Suspicious segments.
    segments = result.get("segments", []) or []
    if segments:
        st.subheader("Suspicious segments")
        st.bar_chart(
            {"fake probability": [s["probability_fake"] for s in segments]}
        )
        flagged = [s for s in segments if s["probability_fake"] >= eff_threshold]
        if flagged:
            for s in flagged:
                st.write(
                    f"⚠️ {s['start']:.1f}–{s['end']:.1f}s — "
                    f"{s['probability_fake'] * 100:.1f}% fake"
                )
        else:
            st.caption("No segment crossed the fake threshold.")

    st.caption(
        "AI predictions are probabilistic and may err. Treat this as "
        "decision support, not proof — verify with human review."
    )

    st.markdown(
        '<div class="adx-caps"><h2>✦ Audio Analysis Capabilities</h2>'
        '<p class="sub">Our AI model analyzes multiple audio cues to detect '
        'synthetic or manipulated speech.</p>'
        '<div class="adx-caps-grid">'
        '<div class="adx-cap"><div class="cc" style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);color:#4ade80;">🎙</div>'
        '<div><h4>Spectral Analysis</h4><p>Analyzes frequency patterns and spectral features.</p></div></div>'
        '<div class="adx-cap"><div class="cc" style="background:rgba(168,85,247,.14);border:1px solid rgba(168,85,247,.35);color:#c084fc;">⁂</div>'
        '<div><h4>Voice Pattern Detection</h4><p>Detects unnatural voice cloning or synthesis artifacts.</p></div></div>'
        '<div class="adx-cap"><div class="cc" style="background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.3);color:#fb923c;">〜</div>'
        '<div><h4>Temporal Consistency</h4><p>Analyzes timing and sequence-level features.</p></div></div>'
        '<div class="adx-cap"><div class="cc" style="background:rgba(47,123,255,.14);border:1px solid rgba(47,123,255,.35);color:#7db4ff;">▂▄</div>'
        '<div><h4>Confidence Scoring</h4><p>Provides detailed analysis with probability scores.</p></div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def page_gradcam(detector, explainer, threshold: float) -> None:
    from src.config import (MODEL_NAME as _MODEL_NAME,
                            confidence_from_prob as _conf,
                            label_from_prob as _label)

    # ── Hero ──
    st.markdown(
        '<div class="home-topbar"><span class="deploy">Deploy</span>'
        '<span class="dots">⋮</span></div>'
        '<div class="gcam-hero"><div class="gcam-hero-inner">'
        '<div class="gcam-hero-left">'
        '<div class="gcam-badge">◉ VISUAL EXPLANATION</div>'
        '<h1>🔥 Grad-CAM <span class="grad">Explainability</span></h1>'
        '<p class="sub">See which facial regions drove the model\'s decision.</p>'
        '<p class="desc">Grad-CAM highlights the most influential regions in '
        'the image that contribute to the model\'s prediction, making the '
        'detection result more transparent and understandable.</p>'
        '<div class="gcam-trust">'
        '<div class="t"><span class="ic">👁</span><div><b>Model</b><span>Interpretability</span></div></div>'
        '<div class="t"><span class="ic">◎</span><div><b>Visual</b><span>Heatmaps</span></div></div>'
        '<div class="t"><span class="ic">📊</span><div><b>Better</b><span>Trust &amp; Transparency</span></div></div>'
        '<div class="t"><span class="ic">🛡</span><div><b>Support</b><span>Forensic Analysis</span></div></div>'
        '</div>'
        '</div>'
        '<div class="gcam-visual">'
        '<div class="gcam-orb-wrap">'
        '<div class="gcam-orb"></div>'
        '<div class="gcam-ring" style="inset:6px;"></div>'
        '<div class="gcam-ring" style="inset:0;border-color:rgba(59,157,255,.35);"></div>'
        '<div class="gcam-scan"><i></i><i></i><i></i><i></i></div>'
        '</div>'
        '<div class="gcam-ped"></div>'
        '<div class="gcam-side">'
        '<div class="gcam-panel">Grad-CAM'
        '<div class="ln"></div><div class="ln" style="width:75%;"></div>'
        '<div class="ln" style="width:55%;"></div><div class="ln" style="width:85%;"></div>'
        '</div>'
        '<div class="gcam-thumb"></div>'
        '</div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Upload + Example ──
    up_col, ex_col = st.columns([1.65, 1], gap="medium")
    with up_col:
        st.markdown(
            '<div class="gcam-card">'
            '<div class="gcam-head"><div class="hic">🖼</div>'
            '<div><h3>Upload an Image</h3>'
            '<p>Upload an image to generate Grad-CAM heatmap and visualize '
            'important regions.</p></div></div>'
            '<div class="gcam-drop"><div class="cloud">☁⬆</div>'
            '<h4>Drag and drop an image file here</h4>'
            '<p>or click to browse files</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Upload an image",
                                    type=["jpg", "jpeg", "png"],
                                    key="gradcam",
                                    label_visibility="collapsed")
        st.markdown(
            '<p class="gcam-meta">Maximum file size: 200MB '
            '<span class="sep">|</span> Supported formats: JPG, PNG, JPEG</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with ex_col:
        st.markdown(
            '<div class="gcam-card">'
            '<div class="gcam-head"><div class="hic" style="background:rgba(234,179,8,.12);'
            'border-color:rgba(234,179,8,.3);color:#facc15;">💡</div>'
            '<div><h3>Example</h3>'
            '<p>Grad-CAM shows which facial regions influence the model\'s '
            'decision (e.g., eyes, mouth, nose, skin texture). Brighter areas '
            'indicate higher importance.</p></div></div>'
            '<div class="gcam-ex-grid">'
            '<div class="gcam-ex"><div class="tile" style="background:'
            'radial-gradient(circle at 50% 35%, #16244d 0%, #0a1322 75%);"></div>'
            '<div class="cap">Original Image</div></div>'
            '<div class="gcam-ex"><div class="tile" style="background:'
            'radial-gradient(circle at 50% 45%, #ffeb3b 0%, #ff9800 22%, #f44336 42%, '
            '#9c27b0 64%, #0d47a1 100%);"></div>'
            '<div class="cap">Grad-CAM Heatmap</div></div>'
            '<div class="gcam-ex"><div class="tile" style="background:'
            'radial-gradient(circle at 60% 40%, rgba(255,152,0,.85) 0%, rgba(244,67,54,.55) 30%, '
            'rgba(22,36,77,.9) 70%), #0a1322;"></div>'
            '<div class="cap">Overlay Result</div></div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Results (placeholders until analysis completes) ──
    def _stat_cards(pred_text, pred_color, conf_text):
        st.markdown(
            '<div class="gcam-stats">'
            '<div class="gcam-stat"><div class="cc" style="background:rgba(168,85,247,.14);'
            'border:1px solid rgba(168,85,247,.35);color:#c084fc;">🧠</div>'
            '<div><p class="k">Prediction</p>'
            f'<p class="v" style="color:{pred_color};">{pred_text}</p>'
            '<p class="h">Model prediction result</p></div></div>'
            '<div class="gcam-stat"><div class="cc" style="background:rgba(34,197,94,.12);'
            'border:1px solid rgba(34,197,94,.3);color:#4ade80;">🛡</div>'
            '<div><p class="k">Confidence</p>'
            f'<p class="v">{conf_text}</p>'
            '<p class="h">Model confidence score</p></div></div>'
            '<div class="gcam-stat"><div class="cc" style="background:rgba(249,115,22,.12);'
            'border:1px solid rgba(249,115,22,.3);color:#fb923c;">⇄</div>'
            '<div><p class="k">Decision Threshold</p>'
            f'<p class="v" style="color:#fb923c;">{threshold:.2f}</p>'
            '<p class="h">Current threshold value</p></div></div>'
            '<div class="gcam-stat"><div class="cc" style="background:rgba(47,123,255,.14);'
            'border:1px solid rgba(47,123,255,.35);color:#7db4ff;">⦃⦄</div>'
            '<div><p class="k">Model</p>'
            f'<p class="v" style="color:#22d3ee;">{_MODEL_NAME.lower()}</p>'
            '<p class="h">Deep learning model used</p></div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if uploaded is None:
        st.markdown(
            '<div class="gcam-results">'
            '<div class="gcam-head"><div class="hic">📊</div>'
            '<div><h3>Analysis Results</h3>'
            '<p>The original image, Grad-CAM heatmap, and overlay result will '
            'be shown here after analysis.</p></div></div>'
            '<div class="gcam-res-grid">'
            '<div class="gcam-slot"><span class="sic">🖼</span>'
            '<b>Original Image</b><span>Image will appear here</span></div>'
            '<div class="gcam-slot"><span class="sic">◎</span>'
            '<b>Grad-CAM Heatmap</b><span>Heatmap will appear here</span></div>'
            '<div class="gcam-slot"><span class="sic">🖼</span>'
            '<b>Overlay Result</b><span>Overlay image will appear here</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        _stat_cards("–", "#fff", "–")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Identical inference path as before ──
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
    prob = float(out["fake_probability"])
    label = _label(prob, threshold)
    conf = _conf(prob, threshold)
    pred_color = "#ff6b6b" if label == "FAKE" else "#22c55e"

    st.markdown(
        '<div class="gcam-results">'
        '<div class="gcam-head"><div class="hic">📊</div>'
        '<div><h3>Analysis Results</h3>'
        '<p>Warmer colors = more influence on the \'fake\' decision.</p></div></div>'
        '<div class="gcam-res-grid">',
        unsafe_allow_html=True,
    )
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown('<div class="gcam-slot filled"><b>Original Image</b>',
                    unsafe_allow_html=True)
        st.image(out["original"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="gcam-slot filled"><b>Grad-CAM Heatmap</b>',
                    unsafe_allow_html=True)
        st.image(out["overlay"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r3:
        st.markdown('<div class="gcam-slot filled"><b>Overlay Result</b>',
                    unsafe_allow_html=True)
        st.image(out["overlay"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    _stat_cards(label, pred_color, f"{conf * 100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)


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

    st.sidebar.markdown(
        '<div class="side-brand"><div class="shield">'
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#5ea0ff" stroke-width="2">'
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
        '<circle cx="12" cy="10" r="2.2" fill="#5ea0ff" stroke="none"/>'
        '<path d="M7.5 10h2M14.5 10h2" stroke="#5ea0ff" stroke-width="1.6"/></svg>'
        '</div><div><h2>Deepfake Detector</h2><p>AI Security Suite</p></div></div>'
        f'<div style="font-size:12px; color:#5a6b82; margin:-6px 0 10px 2px;">'
        f'Signed in as <strong style="color:#c4cfdf;">{st.session_state.get("username", "user")}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    nav_pages = ["Home", "Image Detection", "Video Detection", "Audio Detection",
                 "Webcam Detection", "Grad-CAM", "Prediction History",
                 "Dashboard", "About"]
    if st.session_state.get("role") == "admin":
        nav_pages.append("User Management")
    # Home-page cards/CTAs link here via ?nav=<Page>; consume once so the
    # sidebar highlight stays in sync without changing routing logic.
    _qn = st.query_params.get("nav", None)
    if isinstance(_qn, str) and _qn in nav_pages:
        st.session_state["nav"] = _qn
        st.query_params.clear()
    if st.session_state.get("nav") not in nav_pages:
        st.session_state["nav"] = nav_pages[0]
    page = st.sidebar.radio("Navigate", nav_pages, label_visibility="collapsed",
                            key="nav")

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    # ── Threshold control ──
    st.sidebar.markdown(
        '<p class="side-thresh-label">Decision Threshold</p>',
        unsafe_allow_html=True,
    )
    threshold = st.sidebar.slider(
        "Decision threshold", 0.0, 1.0,
        DEFAULT_THRESHOLD, 0.01,
        label_visibility="collapsed",
        help="Fake probability >= threshold triggers a FAKE verdict. Lower = more sensitive.",
    )
    st.sidebar.markdown(
        f'<p class="side-thresh-val">{threshold:.2f}</p>'
        f'<p class="side-thresh-hint">'
        f'Verdict fires at <span style="color:#ef4444;font-weight:800;">{threshold:.2f}</span> '
        f'&mdash; lower values flag more samples as fake</p>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    # ── System info ──
    st.sidebar.markdown(
        f'<p class="side-device">'
        f'<span class="dot"></span>'
        f'Device: <code>{DEVICE.lower()}</code>'
        f' &nbsp;|&nbsp; Model: <code>{MODEL_NAME.lower()}</code>'
        f'</p>',
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
    elif page == "Audio Detection":
        audio_detector = get_audio_detector_or_warn()
        page_audio(audio_detector, threshold)
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