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

    st.header(
        "📹 Real-Time Webcam Detection"
    )

    st.write(
        "Live deepfake detection with optional "
        "enrolled-profile verification."
    )

    # ------------------------------------------------------------------
    # Profile verification section
    # ------------------------------------------------------------------

    st.subheader(
        "👤 Enrolled Profile Verification"
    )

    st.info(
        "Upload your genuine profile photo once. "
        "During webcam detection, the system compares "
        "the live face with this enrolled reference."
    )

    verifier = get_profile_verifier()

    if verifier is None:

        st.warning(
            "Face verification dependencies are unavailable."
        )

        st.code(
            "pip install facenet-pytorch"
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            profile_upload = st.file_uploader(
                "Upload genuine profile photo",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                ],
                key="profile_enrollment",
            )

        with col2:

            if PROFILE_IMAGE.exists():

                st.success(
                    "✅ Profile already enrolled"
                )

                st.image(
                    str(PROFILE_IMAGE),
                    caption="Enrolled profile",
                    width=180,
                )

            else:

                st.warning(
                    "No profile enrolled yet."
                )

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

    st.markdown("---")

    # ------------------------------------------------------------------
    # Webcam
    # ------------------------------------------------------------------

    try:

        import av

        from streamlit_webrtc import (
            WebRtcMode,
            webrtc_streamer,
        )

        st.caption(
            "Allow camera access when prompted, "
            "then click START."
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

        webrtc_streamer(
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
        )

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