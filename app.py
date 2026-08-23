"""
Streamlit web app for the deepfake detection project.

Save this file as `app.py` in your project ROOT
(C:\\Users\\mdsha\\OneDrive\\Desktop\\deepfake_detection_system\\app.py)

Run with:
    streamlit run app.py
"""

import sys
import os
import tempfile

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

# ---------------------------------------------------------------------------
# Config -- EDIT THESE if your paths differ
# ---------------------------------------------------------------------------
MODEL_NAME = "xception"
CHECKPOINT_PATH = "models/best_xception_celebdf.pth"
IMG_SIZE = 299
DEFAULT_THRESHOLD = 0.5

# Audio deepfake detector (separate model/checkpoint)
AUDIO_MODEL_NAME = DEFAULT_AUDIO_MODEL
AUDIO_CHECKPOINT_PATH = DEFAULT_AUDIO_CHECKPOINT

st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="centered")


# ---------------------------------------------------------------------------
# Cached loaders (run once per session, not on every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(MODEL_NAME, pretrained=False)
    model = load_checkpoint(model, CHECKPOINT_PATH)
    model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def load_face_detector():
    # Shared FaceDetector: prefers the fast Haar cascade, and automatically
    # falls back to MTCNN on OpenCV builds without Haar support (e.g. OpenCV 5).
    return FaceDetector(backend="auto")


@st.cache_resource
def load_audio_model():
    return load_audio_detector(AUDIO_MODEL_NAME, AUDIO_CHECKPOINT_PATH)


def extract_face(bgr_image, face_detector):
    # Returns the largest detected face crop (BGR), or None.
    return face_detector.crop_largest(bgr_image)


def predict(face_bgr, model, device, transform):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    augmented = transform(image=face_rgb)
    tensor = augmented["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    return prob


def sample_video_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, min(max_frames, total)).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🕵️ Deepfake Detector")
st.caption(f"Model: {MODEL_NAME} · Checkpoint: {CHECKPOINT_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    st.error(f"Checkpoint not found at `{CHECKPOINT_PATH}`. Update CHECKPOINT_PATH in app.py.")
    st.stop()

model, device = load_model()
face_cascade = load_face_detector()
transform = get_val_transforms(IMG_SIZE)

threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
fusion_weight = st.sidebar.slider(
    "Video weight (audio+video fusion)", 0.0, 1.0, 0.5, 0.05,
    help="Weight of the face score in the combined video verdict; audio gets the rest.",
)
st.sidebar.write(f"Device: `{device}`")

tab_img, tab_vid, tab_cam, tab_audio = st.tabs(
    ["📷 Image", "🎬 Video", "📹 Live Webcam", "🔊 Audio"]
)

# --- Image tab -------------------------------------------------------------
with tab_img:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img")
    if uploaded_img is not None:
        pil_img = Image.open(uploaded_img).convert("RGB")
        bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        face = extract_face(bgr_img, face_cascade)
        col1, col2 = st.columns(2)
        col1.image(pil_img, caption="Uploaded image", use_column_width=True)

        if face is None:
            st.warning("No face detected — try a clearer, front-facing photo.")
        else:
            col2.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption="Detected face", use_column_width=True)

            show_heatmap = st.checkbox("Show Grad-CAM heatmap", value=False)

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            augmented = transform(image=face_rgb)
            input_tensor = augmented["image"].unsqueeze(0).to(device)

            if show_heatmap:
                rgb_face_float = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
                overlay, prob = generate_gradcam_overlay(model, input_tensor, rgb_face_float, device)
                col2.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
            else:
                prob = predict(face, model, device, transform)

            label = "FAKE" if prob >= threshold else "REAL"
            st.subheader(f"Prediction: {'🔴' if label == 'FAKE' else '🟢'} {label}")
            st.progress(min(max(prob, 0.0), 1.0))
            st.write(f"Fake probability: **{prob:.4f}** (threshold: {threshold:.2f})")

# --- Video tab ---------------------------------------------------------------
with tab_vid:
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], key="vid")
    max_frames = st.slider("Frames to sample", 4, 32, 16)

    audio_ready = os.path.exists(AUDIO_CHECKPOINT_PATH)
    use_fusion = st.checkbox(
        "Combine face + voice (audio+video fusion)", value=audio_ready,
        disabled=not audio_ready,
        help=None if audio_ready else "Train an audio model first to enable fusion.",
    )
    if not audio_ready:
        st.caption("Face-only: no audio checkpoint found — see the 🔊 Audio tab to train one.")

    if uploaded_vid is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1]) as tmp:
            tmp.write(uploaded_vid.read())
            video_path = tmp.name

        st.video(uploaded_vid)
    
        # --- Visual (face) score ---   
        with st.spinner("Sampling frames and running inference..."):
            frames = sample_video_frames(video_path, max_frames=max_frames)
            probs = []
            for frame in frames:
                face = extract_face(frame, face_cascade)
            if face is not None:
                    probs.append(predict(face, model, device, transform))
        video_prob = float(np.mean(probs)) if probs else None

        # --- Audio (voice) score from the video's audio track ---
        audio_prob = None
        if use_fusion and audio_ready:
            with st.spinner("Extracting and analyzing the audio track..."):
                wav = load_audio_from_video(video_path, SAMPLE_RATE)
                if wav is not None:
                    audio_model, audio_device = load_audio_model()
                    audio_prob = predict_from_array(wav, audio_model, audio_device)

        os.unlink(video_path)

        # --- Fuse ---
        fused = fuse_scores(video_prob, audio_prob, w_video=fusion_weight)

        if fused is None:
            st.warning("No face detected in any sampled frame, and no usable audio track.")
        else:
            label = "FAKE" if fused >= threshold else "REAL"
            st.subheader(f"Combined verdict: {'🔴' if label == 'FAKE' else '🟢'} {label}")
            st.progress(min(max(fused, 0.0), 1.0))
            st.write(f"Fused fake probability: **{fused:.4f}** (threshold: {threshold:.2f})")

            c_vid, c_aud = st.columns(2)
            if video_prob is not None:
                c_vid.metric("🎬 Face score", f"{video_prob:.3f}",
                             help=f"Averaged over {len(probs)} frames with a detected face")
            else:
                c_vid.metric("🎬 Face score", "n/a", help="No face detected")
            if audio_prob is not None:
                c_aud.metric("🔊 Voice score", f"{audio_prob:.3f}")
            elif use_fusion and audio_ready:
                c_aud.metric("🔊 Voice score", "n/a", help="No usable audio track (needs ffmpeg for mp4/mov)")
            else:
                c_aud.metric("🔊 Voice score", "off")

            if video_prob is not None and audio_prob is not None:
                weights = f"video {fusion_weight:.2f} / audio {1 - fusion_weight:.2f}"
                st.caption(f"Fusion = weighted average ({weights}).")
                if abs(video_prob - audio_prob) >= 0.4:
                    st.info("⚠️ Face and voice scores disagree strongly — inspect the clip manually.")

            if probs:
                st.write("Per-frame face fake probability:")
                st.line_chart(probs)

# --- Live Webcam tab ---------------------------------------------------------
with tab_cam:
    st.write("Real-time deepfake detection straight from your webcam.")

    try:
        from collections import deque
        import av  # provided by streamlit-webrtc
        from streamlit_webrtc import webrtc_streamer, WebRtcMode

        st.caption("Allow camera access when your browser asks, then click **START**.")

        # Rolling window smooths per-frame jitter into a stable prediction.
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
            draw_overlay(
                img, box, prob, threshold, smoothed,
                fps=0.0, show_help=False,
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="deepfake-live",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        st.caption(
            "Green box = REAL, red box = FAKE. Prediction is averaged over the "
            "last several frames using the current sidebar threshold."
        )

    except ImportError:
        st.info(
            "In-browser live video needs the optional `streamlit-webrtc` package:\n\n"
            "```\npip install streamlit-webrtc\n```\n\n"
            "You can still take a **snapshot** below, or run the full-speed desktop "
            "detector from a terminal:\n\n"
            "```\npython -m src.realtime_webcam\n```"
        )

        snap = st.camera_input("Take a snapshot to analyze")
        if snap is not None:
            pil_snap = Image.open(snap).convert("RGB")
            bgr_snap = cv2.cvtColor(np.array(pil_snap), cv2.COLOR_RGB2BGR)
            face = extract_face(bgr_snap, face_cascade)
            if face is None:
                st.warning("No face detected — face the camera and try again.")
            else:
                prob = predict(face, model, device, transform)
                label = "FAKE" if prob >= threshold else "REAL"
                col1, col2 = st.columns(2)
                col1.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
                           caption="Detected face", use_column_width=True)
                col2.subheader(f"{'🔴' if label == 'FAKE' else '🟢'} {label}")
                col2.progress(min(max(prob, 0.0), 1.0))
                col2.write(f"Fake probability: **{prob:.4f}** (threshold: {threshold:.2f})")

# --- Audio tab ---------------------------------------------------------------
with tab_audio:
    st.write("Detect AI-generated / cloned **voice** deepfakes from an audio clip.")

    if not os.path.exists(AUDIO_CHECKPOINT_PATH):
        st.warning(
            f"Audio model checkpoint not found at `{AUDIO_CHECKPOINT_PATH}`.\n\n"
            "Train one first:\n\n"
            "```\npython -m src.audio_train --data_root data/audio --epochs 15\n```\n\n"
            "(expects `data/audio/real/` and `data/audio/fake/` subfolders), "
            "then reload this page."
        )
    else:
        uploaded_audio = st.file_uploader(
            "Upload an audio clip", type=["wav", "mp3", "flac", "m4a", "ogg"], key="aud"
        )
        if uploaded_audio is not None:
            import librosa
            import matplotlib.pyplot as plt
            from src.audio_dataset import (
                SAMPLE_RATE, waveform_to_logmel, fix_waveform_length,
            )

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]
            ) as tmp:
                tmp.write(uploaded_audio.read())
                audio_path = tmp.name

            st.audio(uploaded_audio)

            audio_model, audio_device = load_audio_model()
            with st.spinner("Analyzing audio..."):
                y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                prob = predict_from_array(y, audio_model, audio_device)
            os.unlink(audio_path)

            label = "FAKE" if prob >= threshold else "REAL"
            st.subheader(f"Prediction: {'🔴' if label == 'FAKE' else '🟢'} {label}")
            st.progress(min(max(prob, 0.0), 1.0))
            st.write(f"Fake probability: **{prob:.4f}** (threshold: {threshold:.2f})")

            # Waveform + log-mel spectrogram of the analyzed window
            y_fixed = fix_waveform_length(y, sr=SAMPLE_RATE)
            logmel = waveform_to_logmel(y_fixed, sr=SAMPLE_RATE)
            fig, axes = plt.subplots(2, 1, figsize=(8, 4), constrained_layout=True)
            axes[0].plot(np.linspace(0, len(y_fixed) / SAMPLE_RATE, len(y_fixed)),
                         y_fixed, linewidth=0.5)
            axes[0].set_title("Waveform")
            axes[0].set_xlabel("Time (s)")
            im = axes[1].imshow(logmel, aspect="auto", origin="lower", cmap="magma")
            axes[1].set_title("Log-mel spectrogram")
            axes[1].set_ylabel("Mel bin")
            fig.colorbar(im, ax=axes[1], format="%+2.0f dB")
            st.pyplot(fig)

        st.caption(
            "For continuous **live microphone** detection, run:\n\n"
            "```\npython -m src.realtime_audio\n```"
        )

st.sidebar.markdown("---")
st.sidebar.caption("Deepfake Detection System · Xception · Celeb-DF v2")