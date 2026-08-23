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
    """Global CSS for page-specific accent colors (orange = Home, green = Image Detection)."""
    st.markdown(
        """
        <style>
        /* Orange theme — Home page */
        .home-header {
            color: #ff8c1a;
            font-size: 34px;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .home-intro {
            color: #d1d1d1;
            font-size: 15px;
            margin-bottom: 18px;
        }
        .metric-card-orange {
            background: rgba(255, 140, 26, 0.10);
            border: 1px solid rgba(255, 140, 26, 0.35);
            border-radius: 10px;
            padding: 14px 18px;
            text-align: center;
        }
        .metric-card-orange .metric-label {
            color: #ffb066;
            font-size: 13px;
            font-weight: 600;
        }
        .metric-card-orange .metric-value {
            color: #ff8c1a;
            font-size: 26px;
            font-weight: 800;
        }
        .home-section-title {
            color: #ff8c1a;
            font-size: 20px;
            font-weight: 700;
            margin-top: 10px;
        }
        .home-bullet-card {
            background: rgba(255, 140, 26, 0.07);
            border-left: 3px solid #ff8c1a;
            border-radius: 6px;
            padding: 8px 14px;
            margin: 6px 0;
        }

        /* Green theme — Image Detection page */
        .image-header {
            color: #2ecc71;
            font-size: 30px;
            font-weight: 800;
            margin-bottom: 10px;
        }
        div[data-testid="stFileUploader"] section {
            border: 2px dashed #2ecc71 !important;
            background: rgba(46, 204, 113, 0.06) !important;
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
# ---------------------------------------------------------------------------
def page_home(threshold: float) -> None:
    st.title("🕵️ AI-Powered Explainable Deepfake Detection")
    st.markdown(
        """
        Detect whether an **image**, **video**, or **live webcam** stream is
        **REAL** or a **DEEPFAKE** — with confidence scores and visual
        **Grad-CAM** explanations of *why*.
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", MODEL_NAME.capitalize())
    c2.metric("Input size", f"{IMG_SIZE}px")
    c3.metric("Device", DEVICE.upper())

    st.subheader("What you can do")
    st.markdown(
        """
        - **📷 Image Detection** — upload a photo and get an instant verdict
        - **🎬 Video Detection** — frame-by-frame analysis with majority voting
        - **📹 Webcam Detection** — real-time detection from your camera
        - **🔥 Grad-CAM** — see the regions that drove the decision
        - **🗂️ Prediction History** — every result stored in SQLite
        - **📊 Dashboard** — aggregate analytics
        """
    )
    if not CHECKPOINT_PATH.exists():
        st.warning(
            f"No model checkpoint found at `{CHECKPOINT_PATH}`. "
            "Detection pages will prompt you to add one."
        )


def page_image(detector, threshold: float) -> None:
    st.header("📷 Image Detection")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    col1, col2 = st.columns(2)
    col1.image(uploaded, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        result = run_image(detector, uploaded)

    with col2:
        if result["prediction"] == "NO_FACE":
            st.warning("No face detected — try a clearer, front-facing photo.")
        else:
            probability_fake = float(
                result.get(
                    "probability_fake",
                    result.get("fake_probability", 0.0),
                )
            )

            verdict_banner(
                probability_fake,
                threshold,
            )

            inference_time = float(
                result.get("inference_time", 0.0)
            )

            st.caption(
                f"Inference time: {inference_time * 1000:.0f} ms"
            )


def page_video(detector, threshold: float) -> None:
    st.header("🎬 Video Detection")

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    max_frames = st.slider(
        "Frames to sample",
        4,
        32,
        VIDEO_MAX_FRAMES,
    )

    if uploaded is None:
        return

    st.video(uploaded)

    with st.spinner("Sampling frames and running inference..."):
        result = run_video(
            detector,
            uploaded,
            max_frames=max_frames,
        )

    # Handle videos where no face was detected.
    if result.get("prediction") == "NO_FACE":
        st.warning("No face detected in any sampled frame.")
        return

    # ---------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------
    probability_fake = float(
        result.get(
            "probability_fake",
            result.get("fake_probability", 0.0),
        )
    )

    verdict_banner(probability_fake, threshold)

    # ---------------------------------------------------------
    # Inference time
    # ---------------------------------------------------------
    inference_time = float(
        result.get("inference_time", 0.0)
    )

    st.caption(
        f"Inference time: {inference_time * 1000:.0f} ms"
    )

    # ---------------------------------------------------------
    # Frame statistics
    # ---------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    frames_sampled = int(
        result.get("frames_sampled", 0)
    )

    frames_with_face = int(
        result.get("frames_with_face", 0)
    )

    frame_probabilities = (
        result.get("frame_probabilities", []) or []
    )

    # Some detector versions do not return these two fields.
    # Calculate them from frame_probabilities instead.
    fake_frames = result.get("fake_frames")
    real_frames = result.get("real_frames")

    if fake_frames is None:
        fake_frames = sum(
            1
            for p in frame_probabilities
            if float(p) >= threshold
        )

    if real_frames is None:
        real_frames = sum(
            1
            for p in frame_probabilities
            if float(p) < threshold
        )

    c1.metric(
        "Frames sampled",
        frames_sampled,
    )

    c2.metric(
        "With a face",
        frames_with_face,
    )

    c3.metric(
        "Voted FAKE",
        int(fake_frames),
    )

    c4.metric(
        "Voted REAL",
        int(real_frames),
    )

    # ---------------------------------------------------------
    # Per-frame probability chart
    # ---------------------------------------------------------
    if frame_probabilities:
        st.caption("Per-frame fake probability")
        st.line_chart(frame_probabilities)

    # ---------------------------------------------------------
    # Debug information
    # ---------------------------------------------------------
    with st.expander("🔍 Debug: raw result keys"):
        st.write(list(result.keys()))

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

    verdict_banner(out["probability_fake"], threshold)
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

    st.sidebar.title("🕵️ Deepfake Detector")
    st.sidebar.success(f"👤 Signed in as **{st.session_state.get('username', 'user')}**")
    if st.sidebar.button("🚪 Log out", use_container_width=True):
        logout()
    change_password_sidebar()

    nav_pages = ["Home", "Image Detection", "Video Detection", "Webcam Detection",
                 "Grad-CAM", "Prediction History", "Dashboard", "About"]
    if st.session_state.get("role") == "admin":
        nav_pages.append("User Management")
    page = st.sidebar.radio("Navigate", nav_pages)
    threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0,
                                  DEFAULT_THRESHOLD, 0.01)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Device: `{DEVICE}`")
    st.sidebar.caption(f"Model: `{MODEL_NAME}`")
    if not CHECKPOINT_PATH.exists():
        st.sidebar.error("Model checkpoint missing")

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