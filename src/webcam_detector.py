"""
src/webcam_detector.py
Real-time webcam deepfake detection as a reusable class.

WebcamDetector.process_frame() annotates a single BGR frame with a face box,
REAL/FAKE label, confidence, and FPS — usable both by the standalone OpenCV
loop (WebcamDetector.run) and by the Streamlit WebRTC callback.

Run standalone from the project root:
    python -m src.webcam_detector
    python -m src.webcam_detector --camera 0 --threshold 0.5 --mirror
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import (
    DEFAULT_THRESHOLD, WEBCAM_INFER_EVERY_N, WEBCAM_SMOOTH_WINDOW,
    confidence_from_prob,
)
from src.inference import DeepfakeDetector, FaceDetector, ModelNotFoundError, get_detector
from src.logger import get_logger

logger = get_logger(__name__)

Box = Tuple[int, int, int, int]


class WebcamDetector:
    """Stateful per-frame webcam detector with smoothing and FPS tracking."""

    def __init__(self, detector: Optional[DeepfakeDetector] = None,
                 threshold: float = DEFAULT_THRESHOLD,
                 infer_every_n: int = WEBCAM_INFER_EVERY_N,
                 smooth_window: int = WEBCAM_SMOOTH_WINDOW):
        self.detector = detector or get_detector()
        # Prefer the fast Haar backend for a smooth live feed; fall back to
        # MTCNN automatically on OpenCV builds without Haar cascades.
        self.face_detector = FaceDetector(backend="auto", device=self.detector.device)
        self.threshold = threshold
        self.infer_every_n = max(1, infer_every_n)
        self.prob_history: deque = deque(maxlen=smooth_window)

        self._frame_idx = 0
        self._last_box: Optional[Box] = None
        self._last_prob: Optional[float] = None
        self._fps = 0.0
        self._fps_timer = time.time()
        self._fps_frames = 0

    def _update_fps(self) -> None:
        self._fps_frames += 1
        now = time.time()
        if now - self._fps_timer >= 0.5:
            self._fps = self._fps_frames / (now - self._fps_timer)
            self._fps_timer = now
            self._fps_frames = 0

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Annotate and return a single BGR frame (drawn in place)."""
        box = self.face_detector.largest_face_box(frame_bgr)

        if box is not None and (self._frame_idx % self.infer_every_n == 0):
            x0, y0, x1, y1 = box
            crop = frame_bgr[y0:y1, x0:x1]
            if crop.size > 0:
                self._last_prob = self.detector._predict_face(crop)
                self.prob_history.append(self._last_prob)
        self._last_box = box

        smoothed = float(np.mean(self.prob_history)) if self.prob_history else None
        self._update_fps()
        self._draw(frame_bgr, box, smoothed)
        self._frame_idx += 1
        return frame_bgr

    def _draw(self, frame: np.ndarray, box: Optional[Box],
              smoothed: Optional[float]) -> None:
        h_img, w_img = frame.shape[:2]

        if box is not None and smoothed is not None:
            x0, y0, x1, y1 = box
            is_fake = smoothed >= self.threshold
            color = (0, 0, 255) if is_fake else (0, 200, 0)  # BGR red / green
            label = "FAKE" if is_fake else "REAL"
            conf = confidence_from_prob(smoothed, self.threshold)

            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            text = f"{label}  {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            chip_y0 = max(0, y0 - th - 12)
            cv2.rectangle(frame, (x0, chip_y0), (x0 + tw + 12, y0), color, -1)
            cv2.putText(frame, text, (x0 + 6, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            bar_w = x1 - x0
            fill = int(bar_w * min(max(smoothed, 0.0), 1.0))
            cv2.rectangle(frame, (x0, y1 + 6), (x1, y1 + 16), (60, 60, 60), -1)
            cv2.rectangle(frame, (x0, y1 + 6), (x0 + fill, y1 + 16), color, -1)
        else:
            cv2.putText(frame, "No face detected", (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)

        fps_text = f"{self._fps:4.1f} FPS"
        (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, fps_text, (w_img - fw - 16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def run(self, camera: int = 0, mirror: bool = False) -> None:
        """Open the webcam and run the live detection loop until 'q'/ESC."""
        # CAP_DSHOW gives a faster camera start on Windows.
        cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise SystemExit(
                f"Could not open camera index {camera}. Try --camera 1 / 2."
            )

        window = "Deepfake Detector - Live"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        logger.info("Webcam loop started (camera=%d)", camera)
        print("Running. Focus the window and press 'q' or ESC to quit.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Failed to read frame from camera; stopping.")
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)

                frame = self.process_frame(frame)
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("m"):
                    mirror = not mirror
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam loop stopped.")
            print("Stopped.")


def parse_args():
    p = argparse.ArgumentParser(description="Real-time webcam deepfake detection")
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--mirror", action="store_true", help="Mirror (selfie) view")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        detector = get_detector()
    except ModelNotFoundError as exc:
        raise SystemExit(str(exc))
    WebcamDetector(detector, threshold=args.threshold).run(
        camera=args.camera, mirror=args.mirror
    )


if __name__ == "__main__":
    main()
