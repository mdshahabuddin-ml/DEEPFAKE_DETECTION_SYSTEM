"""
app/webcam.py

Webcam deepfake detection with optional enrolled-profile verification.

Pipeline:

Webcam
   ↓
Face detection / embedding
   ↓
Enrolled profile verification
   ↓
Xception deepfake detector
   ↓
Temporal smoothing
   ↓
REAL / FAKE / UNCERTAIN
   ↓
Prediction history

Important:
Profile verification and deepfake detection are separate signals.

A matching enrolled profile does NOT mathematically prove that a frame is
100% authentic. It means that the detected face matches the user's enrolled
reference profile.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Existing project imports
# ---------------------------------------------------------------------------

from src.config import confidence_from_prob, label_from_prob
from src.webcam_detector import WebcamDetector
from app.predict import store_result


# ---------------------------------------------------------------------------
# Optional FaceNet imports
# ---------------------------------------------------------------------------

try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1

    FACENET_AVAILABLE = True

except ImportError:
    FACENET_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROFILE_DIR = _ROOT / "data" / "profile"
PROFILE_IMAGE = PROFILE_DIR / "enrolled_profile.jpg"

# Cosine similarity threshold.
#
# 0.75 = fairly strict
# 0.70 = more tolerant
#
# We start at 0.75.
PROFILE_MATCH_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# How-it-works cards (rendered in both the live and fallback paths)
# ---------------------------------------------------------------------------

_WC_HOW_HTML = (
    '<div class="wcx-how">'
    '<div style="display:flex;gap:12px;align-items:flex-start;">'
    '<div style="width:44px;height:44px;border-radius:11px;flex-shrink:0;'
    'display:flex;align-items:center;justify-content:center;font-size:20px;'
    'background:rgba(47,123,255,.14);border:1px solid rgba(47,123,255,.35);">⚙</div>'
    '<div><h2 style="color:#fff;margin:0 0 4px 0;font-size:18px;font-weight:800;">'
    'How It Works</h2>'
    '<p style="color:#7c8da6;margin:0;font-size:13px;">Real-time webcam analysis '
    'with optional profile verification for more accurate results.</p></div>'
    '</div>'
    '<div class="wcx-how-grid">'
    '<div class="wcx-step"><div class="cc" style="background:rgba(168,85,247,.14);'
    'border:1px solid rgba(168,85,247,.35);color:#c084fc;">📷</div>'
    '<div><h4>Capture Live Feed</h4><p>Access your webcam and analyze frames '
    'in real-time.</p></div></div>'
    '<div class="wcx-step"><div class="cc" style="background:rgba(34,197,94,.12);'
    'border:1px solid rgba(34,197,94,.3);color:#4ade80;">⦿</div>'
    '<div><h4>Compare with Profile</h4><p>Match live face with enrolled '
    'reference image.</p></div></div>'
    '<div class="wcx-step"><div class="cc" style="background:rgba(249,115,22,.12);'
    'border:1px solid rgba(249,115,22,.3);color:#fb923c;">🧠</div>'
    '<div><h4>AI Deepfake Analysis</h4><p>Detect manipulation using advanced '
    'AI models.</p></div></div>'
    '<div class="wcx-step"><div class="cc" style="background:rgba(47,123,255,.14);'
    'border:1px solid rgba(47,123,255,.35);color:#7db4ff;">📊</div>'
    '<div><h4>Instant Results</h4><p>Get real-time authenticity scores with '
    'high accuracy.</p></div></div>'
    '</div></div>'
)


# ---------------------------------------------------------------------------
# Face embedding verifier
# ---------------------------------------------------------------------------

class ProfileVerifier:
    """
    Face verification using FaceNet embeddings.

    The verifier compares the live webcam face against an enrolled
    reference face.
    """

    def __init__(self):
        if not FACENET_AVAILABLE:
            raise RuntimeError(
                "facenet-pytorch is not installed. "
                "Install it with: pip install facenet-pytorch"
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            post_process=True,
            device=self.device,
        )

        self.resnet = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(self.device)

        self.reference_embedding: Optional[np.ndarray] = None

        if PROFILE_IMAGE.exists():
            try:
                self.reference_embedding = self._embedding_from_file(
                    PROFILE_IMAGE
                )
            except Exception:
                self.reference_embedding = None

    # ------------------------------------------------------------------
    # Face embedding from image
    # ------------------------------------------------------------------

    def embedding_from_bgr(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        if bgr is None or bgr.size == 0:
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        try:
            face = self.mtcnn(rgb)

            if face is None:
                return None

            if face.ndim == 3:
                face = face.unsqueeze(0)

            face = face.to(self.device)

            with torch.no_grad():
                embedding = self.resnet(face)

            embedding = embedding.cpu().numpy()[0]

            norm = np.linalg.norm(embedding)

            if norm <= 1e-8:
                return None

            embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Embedding from saved file
    # ------------------------------------------------------------------

    def _embedding_from_file(self, path: Path) -> Optional[np.ndarray]:

        image = cv2.imread(str(path))

        if image is None:
            return None

        return self.embedding_from_bgr(image)

    # ------------------------------------------------------------------
    # Enroll profile
    # ------------------------------------------------------------------

    def enroll(self, bgr: np.ndarray) -> bool:

        embedding = self.embedding_from_bgr(bgr)

        if embedding is None:
            return False

        PROFILE_DIR.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(PROFILE_IMAGE),
            bgr,
        )

        self.reference_embedding = embedding

        return True

    # ------------------------------------------------------------------
    # Compare live face with profile
    # ------------------------------------------------------------------

    def verify(
        self,
        bgr: np.ndarray,
        threshold: float = PROFILE_MATCH_THRESHOLD,
    ):

        if self.reference_embedding is None:
            return {
                "matched": False,
                "similarity": 0.0,
                "status": "NOT_ENROLLED",
            }

        live_embedding = self.embedding_from_bgr(bgr)

        if live_embedding is None:
            return {
                "matched": False,
                "similarity": 0.0,
                "status": "NO_FACE",
            }

        similarity = float(
            np.dot(
                self.reference_embedding,
                live_embedding,
            )
        )

        similarity = max(
            -1.0,
            min(1.0, similarity),
        )

        matched = similarity >= threshold

        return {
            "matched": matched,
            "similarity": similarity,
            "status": "MATCH" if matched else "NO_MATCH",
        }


# ---------------------------------------------------------------------------
# Cached verifier
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading profile verification model...")
def get_profile_verifier():

    if not FACENET_AVAILABLE:
        return None

    return ProfileVerifier()


# ---------------------------------------------------------------------------
# Webcam temporal state
# ---------------------------------------------------------------------------

class TemporalState:

    def __init__(self, size: int = 8):

        self.probabilities = []
        self.size = size

        self.lock = threading.Lock()

    def add(self, probability: float):

        with self.lock:

            self.probabilities.append(
                float(probability)
            )

            if len(self.probabilities) > self.size:
                self.probabilities.pop(0)

    def average(self) -> float:

        with self.lock:

            if not self.probabilities:
                return 0.0

            return float(
                np.mean(self.probabilities)
            )

    def clear(self):

        with self.lock:

            self.probabilities.clear()


# ---------------------------------------------------------------------------
# Draw result on webcam frame
# ---------------------------------------------------------------------------

def draw_result(
    frame,
    label,
    probability,
    profile_result,
):

    output = frame.copy()

    height, width = output.shape[:2]

    # ---------------------------------------------------------------
    # Face detection box
    # ---------------------------------------------------------------

    gray = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2GRAY,
    )

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    # ---------------------------------------------------------------
    # Determine profile status
    # ---------------------------------------------------------------

    profile_matched = profile_result.get(
        "matched",
        False,
    )

    similarity = profile_result.get(
        "similarity",
        0.0,
    )

    if profile_matched:

        box_color = (0, 200, 0)

        display_label = (
            "VERIFIED PROFILE"
        )

    elif label == "FAKE":

        box_color = (0, 0, 255)

        display_label = (
            f"FAKE {probability * 100:.1f}%"
        )

    elif label == "REAL":

        box_color = (0, 200, 0)

        display_label = (
            f"REAL {probability * 100:.1f}%"
        )

    else:

        box_color = (0, 165, 255)

        display_label = (
            f"UNCERTAIN {probability * 100:.1f}%"
        )

    # ---------------------------------------------------------------
    # Draw face boxes
    # ---------------------------------------------------------------

    for x, y, w, h in faces:

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            box_color,
            3,
        )

        cv2.rectangle(
            output,
            (x, max(0, y - 40)),
            (x + w, y),
            box_color,
            -1,
        )

        cv2.putText(
            output,
            display_label,
            (x + 5, max(25, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # ---------------------------------------------------------------
    # Profile verification information
    # ---------------------------------------------------------------

    if profile_matched:

        text = (
            "ENROLLED PROFILE VERIFIED"
        )

        cv2.putText(
            output,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            f"Profile similarity: {similarity * 100:.1f}%",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    return output


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

def webcam_page(
    detector,
    threshold: float,
) -> None:

    # Page-scoped button/dropdown accents.
    st.markdown(
        '<style>'
        'main div.stButton > button[kind="primary"] {'
        'background: linear-gradient(90deg, #1e90ff 0%, #7c3aed 100%) !important;'
        'border: none !important; border-radius: 10px !important;'
        'font-weight: 700 !important;'
        'box-shadow: 0 0 20px rgba(80,120,255,.45) !important;}'
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
        '<div class="wcx-hero"><div class="wcx-hero-inner">'
        '<div class="wcx-hero-left">'
        '<div class="wcx-badge"><span class="rd"></span>LIVE ANALYSIS</div>'
        '<h1>Webcam <span class="grad">Deepfake</span> Detection</h1>'
        '<p class="sub">Detect real-time deepfakes from your webcam using '
        'advanced AI models. Optional profile verification for higher accuracy.</p>'
        '<div class="wcx-trust">'
        '<div class="t"><span class="ic">⚡</span><div><b>Real-time</b><span>Detection</span></div></div>'
        '<div class="t"><span class="ic">🛡</span><div><b>Anti-Spoofing</b><span>Protection</span></div></div>'
        '<div class="t"><span class="ic">◉</span><div><b>Live Face</b><span>Analysis</span></div></div>'
        '<div class="t"><span class="ic">📊</span><div><b>High</b><span>Accuracy</span></div></div>'
        '</div>'
        '</div>'
        '<div class="wcx-visual">'
        '<div style="position:relative;z-index:2;">'
        '<svg width="215" height="185" viewBox="0 0 200 172">'
        '<defs>'
        '<linearGradient id="wcxbody" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#16244d"/><stop offset="1" stop-color="#0a1328"/>'
        '</linearGradient>'
        '<radialGradient id="wcxlens" cx=".4" cy=".35" r=".9">'
        '<stop offset="0" stop-color="#7db4ff"/><stop offset=".5" stop-color="#1a5cff"/>'
        '<stop offset="1" stop-color="#0a1440"/>'
        '</radialGradient>'
        '</defs>'
        '<g stroke="#3b9dff" stroke-width="2" opacity="0.8">'
        '<line x1="6" y1="50" x2="6" y2="86"/><line x1="13" y1="42" x2="13" y2="94"/>'
        '<line x1="20" y1="56" x2="20" y2="80"/><line x1="27" y1="46" x2="27" y2="90"/>'
        '<line x1="34" y1="60" x2="34" y2="76"/>'
        '</g>'
        '<g stroke="#a855f7" stroke-width="2" opacity="0.8">'
        '<line x1="166" y1="60" x2="166" y2="76"/><line x1="173" y1="46" x2="173" y2="90"/>'
        '<line x1="180" y1="56" x2="180" y2="80"/><line x1="187" y1="42" x2="187" y2="94"/>'
        '<line x1="194" y1="50" x2="194" y2="86"/>'
        '</g>'
        '<circle cx="100" cy="66" r="60" fill="none" stroke="#22d3ee" stroke-width="1.5" '
        'stroke-dasharray="7 6" opacity="0.75"/>'
        '<circle cx="100" cy="66" r="48" fill="none" stroke="#3b9dff" stroke-width="1" opacity="0.5"/>'
        '<rect x="90" y="4" width="20" height="12" rx="6" fill="#16244d" stroke="#3b9dff" stroke-width="1.5"/>'
        '<rect x="56" y="28" width="88" height="76" rx="22" fill="url(#wcxbody)" stroke="#3b9dff" stroke-width="2"/>'
        '<circle cx="100" cy="66" r="27" fill="#04070f" stroke="#22d3ee" stroke-width="2.5"/>'
        '<circle cx="100" cy="66" r="19" fill="none" stroke="#a855f7" stroke-width="2"/>'
        '<circle cx="100" cy="66" r="11" fill="url(#wcxlens)"/>'
        '<circle cx="94" cy="60" r="3.5" fill="#fff" opacity="0.85"/>'
        '<circle cx="132" cy="38" r="3" fill="#ff5252"/>'
        '<path d="M86 110h28l7 24H79z" fill="#101c3a" stroke="#3b9dff" stroke-width="1.5"/>'
        '<ellipse cx="100" cy="142" rx="54" ry="9" fill="none" stroke="#a855f7" stroke-width="2"/>'
        '<ellipse cx="100" cy="142" rx="38" ry="6" fill="rgba(124,58,237,.25)"/>'
        '</svg></div>'
        '<div class="wcx-scores">'
        '<div class="wcx-score real">✔REAL<br>0.12</div>'
        '<div class="wcx-score fake">✖FAKE<br>0.88</div>'
        '</div>'
        '<div class="wcx-minis">'
        '<div class="wcx-mini">'
        '<svg width="100%" height="40" viewBox="0 0 80 40"><defs>'
        '<linearGradient id="wcxheat" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#22d3ee"/><stop offset=".5" stop-color="#a855f7"/>'
        '<stop offset="1" stop-color="#ff5c8a"/></linearGradient></defs>'
        '<rect x="4" y="4" width="72" height="32" rx="5" fill="url(#wcxheat)" opacity="0.35"/>'
        '<circle cx="30" cy="18" r="7" fill="none" stroke="#ffeb3b" stroke-width="1.5"/>'
        '<circle cx="52" cy="24" r="5" fill="none" stroke="#ff5252" stroke-width="1.5"/>'
        '</svg></div>'
        '<div class="wcx-mini">'
        '<svg width="100%" height="40" viewBox="0 0 80 40">'
        '<polyline points="2,32 14,26 26,28 38,16 50,20 62,10 78,12" fill="none" stroke="#3b9dff" stroke-width="2"/>'
        '<polygon points="2,32 14,26 26,28 38,16 50,20 62,10 78,12 78,40 2,40" fill="rgba(59,157,255,.15"/>'
        '</svg></div>'
        '</div>'
        '<div class="wcx-playbar"><span class="pp">▶</span>'
        '<svg width="100%" height="22" viewBox="0 0 200 22" preserveAspectRatio="none">'
        '<g fill="#3b9dff">'
        '<rect x="2" y="8" width="2" height="6" rx="1"/><rect x="9" y="5" width="2" height="12" rx="1"/>'
        '<rect x="16" y="3" width="2" height="16" rx="1"/><rect x="23" y="6" width="2" height="10" rx="1"/>'
        '<rect x="30" y="2" width="2" height="18" rx="1"/><rect x="37" y="5" width="2" height="12" rx="1"/>'
        '<rect x="44" y="8" width="2" height="6" rx="1"/><rect x="51" y="4" width="2" height="14" rx="1"/>'
        '<rect x="58" y="7" width="2" height="8" rx="1"/><rect x="65" y="3" width="2" height="16" rx="1"/>'
        '<rect x="72" y="6" width="2" height="10" rx="1"/><rect x="79" y="8" width="2" height="6" rx="1"/>'
        '<rect x="86" y="4" width="2" height="14" rx="1"/><rect x="93" y="7" width="2" height="8" rx="1"/>'
        '<rect x="100" y="5" width="2" height="12" rx="1"/><rect x="107" y="8" width="2" height="6" rx="1"/>'
        '<rect x="114" y="3" width="2" height="16" rx="1"/><rect x="121" y="6" width="2" height="10" rx="1"/>'
        '<rect x="128" y="8" width="2" height="6" rx="1"/><rect x="135" y="4" width="2" height="14" rx="1"/>'
        '<rect x="142" y="7" width="2" height="8" rx="1"/><rect x="149" y="5" width="2" height="12" rx="1"/>'
        '<rect x="156" y="8" width="2" height="6" rx="1"/><rect x="163" y="4" width="2" height="14" rx="1"/>'
        '<rect x="170" y="7" width="2" height="8" rx="1"/><rect x="177" y="5" width="2" height="12" rx="1"/>'
        '<rect x="184" y="8" width="2" height="6" rx="1"/><rect x="191" y="6" width="2" height="10" rx="1"/>'
        '</g></svg></div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Panels ──
    pc, lc = st.columns([1.55, 1], gap="medium")

    # ------------------------------------------------------------------
    # Profile verification section (logic unchanged)
    # ------------------------------------------------------------------

    with pc:
        st.markdown(
            '<div class="wcx-card"><div class="wcx-head">'
            '<div class="hic">◉</div>'
            '<div><h3>Enrolled Profile Verification</h3>'
            '<p>Upload your genuine profile photo once. During webcam detection, '
            'the system compares the live face with this enrolled reference.</p></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    verifier = get_profile_verifier()

    if verifier is None:

        with pc:
            st.warning(
                "Face verification dependencies are unavailable."
            )

            st.code(
                "pip install facenet-pytorch"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    else:

        with pc:
            pcol1, pcol2 = st.columns([1.3, 1], gap="small")

        with pcol1:
            st.markdown(
                '<div class="wcx-zone">'
                '<div class="pic">'
                '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" '
                'stroke="#7db4ff" stroke-width="1.8">'
                '<circle cx="12" cy="8" r="3.5"/>'
                '<path d="M5 20c1.5-3.5 4-5 7-5s5.5 1.5 7 5"/></svg>'
                '</div>',
                unsafe_allow_html=True,
            )
            profile_upload = st.file_uploader(
                "Upload genuine profile photo",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                ],
                key="profile_enrollment",
                label_visibility="collapsed",
            )
            st.markdown(
                '<p class="cap">200MB per file • JPG, PNG</p></div>',
                unsafe_allow_html=True,
            )

        with pcol2:

            if PROFILE_IMAGE.exists():

                st.markdown(
                    '<div class="wcx-enrolled">'
                    '<div class="wcx-ok">✔ Profile already enrolled</div>',
                    unsafe_allow_html=True,
                )

                st.image(
                    str(PROFILE_IMAGE),
                    caption="Enrolled profile",
                    width=180,
                )

                st.markdown(
                    '<p class="lbl">Enrolled profile</p>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    "⟳ Replace Photo",
                    key="wc_replace",
                    width="stretch",
                ):
                    try:
                        PROFILE_IMAGE.unlink(missing_ok=True)
                    except Exception:
                        pass
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            else:

                st.markdown(
                    '<div class="wcx-enrolled">',
                    unsafe_allow_html=True,
                )
                st.warning(
                    "No profile enrolled yet."
                )
                st.markdown('</div>', unsafe_allow_html=True)

        with pc:
            if profile_upload is not None:

                data = np.frombuffer(
                    profile_upload.getvalue(),
                    dtype=np.uint8,
                )

                profile_bgr = cv2.imdecode(
                    data,
                    cv2.IMREAD_COLOR,
                )

                if profile_bgr is None:

                    st.error(
                        "Could not read the profile image."
                    )

                else:

                    st.image(
                        cv2.cvtColor(
                            profile_bgr,
                            cv2.COLOR_BGR2RGB,
                        ),
                        caption="Profile to enroll",
                        width=250,
                    )

                    if st.button(
                        "🔐 Enroll This Profile",
                        type="primary",
                        use_container_width=True,
                    ):

                        with st.spinner(
                            "Creating face profile..."
                        ):

                            success = verifier.enroll(
                                profile_bgr
                            )

                        if success:

                            st.success(
                                "✅ Profile enrolled successfully."
                            )

                            st.rerun()

                        else:

                            st.error(
                                "❌ No usable face was detected. "
                                "Use a clear front-facing profile photo."
                            )
            st.markdown('</div>', unsafe_allow_html=True)

    with lc:
        st.markdown(
            '<div class="wcx-card"><div class="wcx-head">'
            '<div class="hic">◉</div>'
            '<div><h3>Live Webcam Detection</h3>'
            '<p>Allow camera access when prompted, then click START.</p></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # Webcam (streamer call itself is unchanged)
    # ------------------------------------------------------------------

    try:

        import av

        from streamlit_webrtc import (
            WebRtcMode,
            webrtc_streamer,
        )

        temporal = TemporalState(
            size=8
        )

        # --------------------------------------------------------------
        # Webcam detector
        # --------------------------------------------------------------

        wd = WebcamDetector(
            detector,
            threshold=threshold,
        )

        def video_frame_callback(frame):

            img = frame.to_ndarray(
                format="bgr24"
            )

            # ----------------------------------------------------------
            # Existing deepfake detector
            # ----------------------------------------------------------

            try:

                processed = wd.process_frame(
                    img
                )

            except Exception:

                processed = img

            # ----------------------------------------------------------
            # Profile verification
            # ----------------------------------------------------------

            profile_result = {
                "matched": False,
                "similarity": 0.0,
                "status": "NOT_ENROLLED",
            }

            if verifier is not None:

                try:

                    profile_result = verifier.verify(
                        img
                    )

                except Exception:

                    profile_result = {
                        "matched": False,
                        "similarity": 0.0,
                        "status": "ERROR",
                    }

            # ----------------------------------------------------------
            # If profile matches, show trusted enrolled profile
            # ----------------------------------------------------------

            if profile_result["matched"]:

                label = "REAL"

                # We intentionally DO NOT change the Xception
                # probability to 0.0.
                #
                # The profile verification is shown separately.

                probability = 0.0

            else:

                # If WebcamDetector exposes a prediction, use it.
                #
                # The processed frame already contains the detector
                # visualization, so we primarily use it as the output.

                label = "UNCERTAIN"

                probability = 0.5

            output = draw_result(
                processed,
                label,
                probability,
                profile_result,
            )

            return av.VideoFrame.from_ndarray(
                output,
                format="bgr24",
            )

        with lc:
            preview_slot = st.empty()
            st.selectbox(
                "Select Camera",
                ["Default Camera"],
                label_visibility="collapsed",
                key="wc_camera",
            )
            if st.button(
                "▶ Start Detection",
                key="wc_start",
                width="stretch",
            ):
                # One-shot autostart: the streamer keeps its default
                # behaviour on every other run.
                st.session_state["wc_autostart"] = True
                st.rerun()
            ctx = webrtc_streamer(
                key="deepfake-live",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=video_frame_callback,
                media_stream_constraints={
                    "video": True,
                    "audio": False,
                },
                rtc_configuration={
                    "iceServers": [
                        {
                            "urls": [
                                "stun:stun.l.google.com:19302"
                            ]
                        }
                    ]
                },
                async_processing=True,
                desired_playing_state=True
                if st.session_state.get("wc_autostart") else None,
            )
            if not ctx.state.playing and not st.session_state.get("wc_autostart"):
                preview_slot.markdown(
                    '<div class="wcx-preview"><i></i><i></i><i></i><i></i>'
                    '<div style="text-align:center;">'
                    '<div style="font-size:34px;color:#5ea0ff;">◉</div>'
                    '<div class="off" style="margin-top:8px;">'
                    '<span class="rd"></span>Camera not started</div>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
            if ctx.state.playing:
                # Connected: stop forcing autostart (persists, no rerun).
                st.session_state.pop("wc_autostart", None)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(_WC_HOW_HTML, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    except ImportError:

        st.info(
            "In-browser live video needs "
            "`streamlit-webrtc`."
        )

        st.code(
            "python -m pip install streamlit-webrtc av"
        )

        st.markdown("### 📸 Snapshot Detection")

        snap = st.camera_input(
            "Take a snapshot to analyze"
        )

        if snap is None:
            return

        file_bytes = np.frombuffer(
            snap.getvalue(),
            dtype=np.uint8,
        )

        bgr = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR,
        )

        if bgr is None:
            st.error(
                "Could not read the camera image."
            )
            return

        # --------------------------------------------------------------
        # Xception deepfake detection
        # --------------------------------------------------------------

        try:
            result = detector.predict_image_array(bgr)
        except Exception as exc:
            st.error(
                f"Detection failed: {exc}"
            )
            return

        if result.get("prediction") == "NO_FACE":
            st.warning(
                "No face detected — face the camera "
                "and try again."
            )
            return

        # --------------------------------------------------------------
        # Normalize fake probability
        # --------------------------------------------------------------

        probability = float(
            result.get(
                "probability_fake",
                result.get(
                    "fake_probability",
                    result.get(
                        "probability",
                        0.0,
                    ),
                ),
            )
        )

        probability = max(
            0.0,
            min(1.0, probability),
        )

        # --------------------------------------------------------------
        # Profile verification
        # --------------------------------------------------------------

        profile_result = {
            "matched": False,
            "similarity": 0.0,
            "status": "NOT_ENROLLED",
        }

        if verifier is not None:
            try:
                profile_result = verifier.verify(
                    bgr,
                    threshold=PROFILE_MATCH_THRESHOLD,
                )
            except Exception as exc:
                st.warning(
                    f"Profile verification failed: {exc}"
                )

        # --------------------------------------------------------------
        # Deepfake result
        # --------------------------------------------------------------

        label = label_from_prob(
            probability,
            threshold,
        )

        conf = confidence_from_prob(
            probability,
            threshold,
        )

        st.subheader("🤖 Deepfake Detection")

        if label == "FAKE":
            st.error(
                f"🔴 DEEPFAKE — "
                f"{conf * 100:.1f}% confidence"
            )
        else:
            st.success(
                f"🟢 REAL — "
                f"{conf * 100:.1f}% confidence"
            )

        st.progress(
            min(
                max(probability, 0.0),
                1.0,
            )
        )

        st.caption(
            f"Xception fake probability: "
            f"{probability * 100:.2f}%"
        )

        # --------------------------------------------------------------
        # Profile verification result
        # --------------------------------------------------------------

        st.subheader("👤 Profile Verification")

        similarity = float(
            profile_result.get(
                "similarity",
                0.0,
            )
        )

        status = profile_result.get(
            "status",
            "UNKNOWN",
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Profile Similarity",
                f"{similarity * 100:.1f}%",
            )

        with col2:
            st.metric(
                "Verification Threshold",
                f"{PROFILE_MATCH_THRESHOLD * 100:.1f}%",
            )

        if status == "MATCH":
            st.success("🟢 PROFILE VERIFIED")
        elif status == "NO_MATCH":
            st.warning("🟠 PROFILE NOT VERIFIED")
        elif status == "NO_FACE":
            st.error("🔴 NO FACE DETECTED")
        elif status == "NOT_ENROLLED":
            st.info(
                "No enrolled profile is available."
            )
        else:
            st.info(
                f"Profile status: {status}"
            )

        # --------------------------------------------------------------
        # Combined assessment
        # --------------------------------------------------------------

        st.subheader("📋 Final Assessment")

        if status == "MATCH" and label == "REAL":
            st.success(
                "🟢 LIKELY REAL + PROFILE VERIFIED"
            )

        elif status == "MATCH" and label == "FAKE":
            st.error(
                "🔴 PROFILE MATCHED, BUT THE "
                "DEEPFAKE MODEL DETECTED POSSIBLE MANIPULATION"
            )

        elif status == "NO_MATCH" and label == "REAL":
            st.warning(
                "🟠 FACE APPEARS REAL, BUT THE "
                "ENROLLED PROFILE COULD NOT BE VERIFIED"
            )

        elif status == "NO_MATCH" and label == "FAKE":
            st.error(
                "🔴 POSSIBLE DEEPFAKE + PROFILE NOT VERIFIED"
            )

        elif status == "NOT_ENROLLED":
            st.info(
                "ℹ️ Deepfake result is available, but "
                "profile identity was not verified."
            )

        else:
            st.warning(
                "🟠 INSUFFICIENT EVIDENCE — "
                "TRY AGAIN WITH A CLEARER FACE"
            )

        # --------------------------------------------------------------
        # Save result
        # --------------------------------------------------------------

        result["file_name"] = "webcam_snapshot.jpg"
        result["profile_verified"] = bool(
            profile_result.get("matched", False)
        )
        result["profile_similarity"] = similarity

        store_result(
            result,
            "webcam",
        )

        st.markdown(_WC_HOW_HTML, unsafe_allow_html=True)