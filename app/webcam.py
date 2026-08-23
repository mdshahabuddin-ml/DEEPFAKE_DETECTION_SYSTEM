"""
app/webcam.py
Streamlit webcam page.

Uses streamlit-webrtc for genuine in-browser real-time detection when the
package is installed, and falls back to a single-snapshot detector (built-in
st.camera_input) plus instructions for the desktop loop otherwise.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import confidence_from_prob, label_from_prob
from src.webcam_detector import WebcamDetector
from app.predict import store_result


def webcam_page(detector, threshold: float) -> None:
    st.header("📹 Real-Time Webcam Detection")
    st.write("Live deepfake detection from your webcam feed.")

    try:
        import av  # noqa: F401  (provided by streamlit-webrtc)
        from streamlit_webrtc import webrtc_streamer, WebRtcMode

        st.caption("Allow camera access when prompted, then click **START**. "
                   "Green box = REAL, red box = FAKE.")

        wd = WebcamDetector(detector, threshold=threshold)

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            img = wd.process_frame(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="deepfake-live",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

    except ImportError:
        st.info(
            "In-browser live video needs `streamlit-webrtc`:\n\n"
            "```\npip install streamlit-webrtc\n```\n\n"
            "You can still take a snapshot below, or run the full-speed desktop "
            "detector:\n\n```\npython -m src.webcam_detector\n```"
        )
        snap = st.camera_input("Take a snapshot to analyze")
        if snap is not None:
            file_bytes = np.frombuffer(snap.getvalue(), np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            result = detector.predict_image_array(bgr)
            if result["prediction"] == "NO_FACE":
                st.warning("No face detected — face the camera and try again.")
            else:
                prob = result["fake_probability"]
                label = label_from_prob(prob, threshold)
                conf = confidence_from_prob(prob, threshold)
                st.subheader(f"{'🔴' if label == 'FAKE' else '🟢'} {label}")
                st.progress(min(max(prob, 0.0), 1.0))
                st.write(f"Confidence: **{conf * 100:.1f}%** · fake prob {prob:.4f}")
                result["file_name"] = "webcam_snapshot.jpg"
                store_result(result, "webcam")
