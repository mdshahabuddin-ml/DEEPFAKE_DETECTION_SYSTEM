"""
src/live_detection.py

Professional Live Webcam + Audio deepfake detection engine.

This module reuses the existing inference logic without modification:
  - Video: src.realtime_webcam.detect_largest_face / predict_fake_prob
  - Audio: src.audio_infer.predict_from_array (+ src.audio_dataset constants)
  - Fusion: src.fusion.fuse_scores

It adds the live-session machinery the Streamlit UI needs:
  - Thread-safe shared state (video + audio + fusion + performance)
  - Background video capture thread (OpenCV, server-side camera)
  - Background audio capture thread (sounddevice, optional)
  - REAL / FAKE / UNCERTAIN classification with confidence + uncertainty flag
  - Temporal smoothing (rolling average) for both modalities
  - Segment-level audio history for the evidence dashboard
  - Camera / microphone enumeration with graceful degradation

No Streamlit imports here so the engine stays testable outside the UI.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.fusion import fuse_scores

# ---------------------------------------------------------------------------
# Classification with uncertainty
# ---------------------------------------------------------------------------

DEFAULT_UNCERTAIN_MARGIN: float = 0.10


def classify_live(
    prob: Optional[float],
    threshold: float = 0.47,
    margin: float = DEFAULT_UNCERTAIN_MARGIN,
) -> Tuple[str, float, bool]:
    """Map a fake-probability to (label, confidence, is_uncertain).

    - prob None (no signal yet) -> ("NO_SIGNAL", 0.0, True)
    - |prob - threshold] <= margin -> "UNCERTAIN"
    - prob >= threshold -> "FAKE", else "REAL"
    - confidence = distance from the decision boundary scaled to [0, 1]:
      confidence = min(|prob - threshold| / max(threshold, 1-threshold), 1.0)
      so a prediction far from the boundary is confident, near = uncertain.
    """
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return "NO_SIGNAL", 0.0, True
    prob = float(min(max(prob, 0.0), 1.0))
    distance = abs(prob - threshold)
    span = max(threshold, 1.0 - threshold, 1e-6)
    confidence = float(min(distance / span, 1.0))
    if distance <= margin:
        return "UNCERTAIN", confidence, True
    if prob >= threshold:
        return "FAKE", confidence, False
    return "REAL", confidence, False


# ---------------------------------------------------------------------------
# Device enumeration (graceful — never raises)
# ---------------------------------------------------------------------------

def list_cameras(max_index: int = 5) -> List[Dict[str, Any]]:
    """Probe cv2.VideoCapture indices 0..max_index-1. Returns usable entries."""
    found: List[Dict[str, Any]] = []
    for idx in range(max_index):
        cap = None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                label = f"Camera {idx}"
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    label = f"Camera {idx} ({w}x{h})"
                found.append({"index": idx, "label": label})
        except Exception:
            continue
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
    if not found:
        # Always offer index 0 so the UI can show a proper "unavailable" error
        # instead of an empty selectbox.
        found.append({"index": 0, "label": "Camera 0 (default)"})
    return found


def list_microphones() -> List[Dict[str, Any]]:
    """List input audio devices via sounddevice when available."""
    try:
        import sounddevice as sd
    except ImportError:
        return []
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    try:
        defaults = sd.default.device
        default_in = defaults[0] if isinstance(defaults, (list, tuple)) else None
    except Exception:
        default_in = None
    for i, d in enumerate(devices):
        try:
            if int(d.get("max_input_channels", 0)) > 0:
                name = str(d.get("name", f"Microphone {i}"))
                suffix = " (default)" if (default_in is not None and i == default_in) else ""
                out.append({"index": i, "label": f"{name}{suffix}"})
        except Exception:
            continue
    return out


def audio_backend_available() -> Tuple[bool, str]:
    """Return (available, reason) for live microphone capture."""
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        return False, "sounddevice package not installed (pip install sounddevice)"
    return True, "ok"


def webrtc_available() -> bool:
    try:
        import streamlit_webrtc  # noqa: F401
        import av  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Frame annotation (preserves existing overlay style + UNCERTAIN state)
# ---------------------------------------------------------------------------

def annotate_frame(
    frame_bgr: np.ndarray,
    box: Optional[Tuple[int, int, int, int]],
    smoothed_prob: Optional[float],
    threshold: float,
    margin: float = DEFAULT_UNCERTAIN_MARGIN,
    fps: float = 0.0,
    latency_ms: float = 0.0,
) -> np.ndarray:
    """Draw face box + label chip + confidence bar + FPS/latency HUD in place."""
    out = frame_bgr
    if box is not None and smoothed_prob is not None:
        x0, y0, x1, y1 = (int(v) for v in box)
        label, conf, _ = classify_live(smoothed_prob, threshold, margin)
        if label == "FAKE":
            color = (0, 0, 255)  # BGR red
        elif label == "REAL":
            color = (0, 200, 0)  # BGR green
        elif label == "UNCERTAIN":
            color = (0, 165, 255)  # BGR orange
        else:
            color = (150, 150, 150)  # grey — no signal
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        text = f"{label}  {smoothed_prob * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        chip_y0 = max(0, y0 - th - 12)
        cv2.rectangle(out, (x0, chip_y0), (x0 + tw + 12, y0), color, -1)
        cv2.putText(out, text, (x0 + 6, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        bar_w = max(x1 - x0, 1)
        fill = int(bar_w * min(max(smoothed_prob, 0.0), 1.0))
        cv2.rectangle(out, (x0, y1 + 6), (x1, y1 + 16), (60, 60, 60), -1)
        cv2.rectangle(out, (x0, y1 + 6), (x0 + fill, y1 + 16), color, -1)
    else:
        cv2.putText(out, "No face detected", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)
    hud = f"{fps:4.1f} FPS  {latency_ms:4.0f} ms"
    (fw, _), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(out, hud, (out.shape[1] - fw - 12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AudioSegment:
    t_start: float
    t_end: float
    fake_prob: float
    label: str
    rms: float


@dataclass
class LiveSessionConfig:
    camera_index: int = 0
    mic_index: Optional[int] = None  # None = default input device
    threshold: float = 0.47
    uncertain_margin: float = DEFAULT_UNCERTAIN_MARGIN
    infer_every_n: int = 2
    smooth_window: int = 8
    audio_smooth_window: int = 5
    fusion_w_video: float = 0.5
    silence_rms: float = 0.01
    audio_hop_sec: float = 1.0
    # Frame quality gating (webcam path only — image/video pages untouched).
    face_margin: float = 0.2      # must match FACE_MARGIN in app image path
    min_face_size: int = 80       # ignore faces smaller than this (pixels)
    blur_threshold: float = 20.0  # Laplacian variance below this = blurred.
    # 20.0 measured: passes ~95% of clean native-res webcam frames
    # (10th pct = 21.7 on Celeb-DF video frames) while still rejecting
    # severely degraded crops. A higher value was shown to reject 100%
    # of genuine webcam-like input and must not be used.


class LiveSession:
    """Thread-safe holder for one live detection session.

    The UI creates a LiveSession, calls start(video_model_bundle, audio_bundle),
    polls snapshot() from a @st.fragment, and calls stop() on Start/Stop toggle.
    """

    def __init__(self, config: LiveSessionConfig):
        self.config = config
        self.lock = threading.RLock()
        self.running = False

        # Video state
        self.video_prob_history: Deque[float] = deque(maxlen=config.smooth_window)
        self.video_frame_count = 0
        self.video_faces_detected = 0
        self.video_last_prob: Optional[float] = None
        self.video_smoothed: Optional[float] = None
        self.video_last_box: Optional[Tuple[int, int, int, int]] = None
        self.video_fps = 0.0
        self.video_latency_ms = 0.0
        self.video_error: Optional[str] = None
        self.latest_frame_jpeg: Optional[bytes] = None
        # Quality gating counters (frames skipped before inference).
        self.video_rejected_small = 0
        self.video_rejected_blur = 0
        self.video_last_reject: Optional[str] = None
        self.video_last_blur: Optional[float] = None

        # Audio state
        self.audio_prob_history: Deque[float] = deque(maxlen=config.audio_smooth_window)
        self.audio_last_prob: Optional[float] = None
        self.audio_smoothed: Optional[float] = None
        self.audio_level = 0.0
        self.audio_rms = 0.0
        self.audio_segments: Deque[AudioSegment] = deque(maxlen=12)
        self.audio_analyses = 0
        self.audio_silence = True
        self.audio_error: Optional[str] = None
        self.audio_waveform: Deque[float] = deque(maxlen=160)
        self.audio_available = False

        # Fusion state
        self.fused_prob: Optional[float] = None

        self.session_start: float = 0.0
        self.session_end: float = 0.0

        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -- lifecycle ------------------------------------------------------
    @property
    def is_running(self) -> bool:
        with self.lock:
            return self.running

    def start(self, video_bundle: Optional[dict], audio_bundle: Optional[dict]) -> None:
        with self.lock:
            if self.running:
                return
            self.running = True
            self.session_start = time.time()
            self.session_end = 0.0
            self._stop_event.clear()
            self.audio_available = audio_bundle is not None

        self._video_thread = threading.Thread(
            target=_video_worker, args=(self, video_bundle), daemon=True,
            name="live-video",
        )
        self._video_thread.start()
        if audio_bundle is not None:
            self._audio_thread = threading.Thread(
                target=_audio_worker, args=(self, audio_bundle), daemon=True,
                name="live-audio",
            )
            self._audio_thread.start()

    def stop(self) -> Dict[str, Any]:
        with self.lock:
            if not self.running:
                return self.snapshot()
            self.running = False
            self.session_end = time.time()
        self._stop_event.set()
        for t in (self._video_thread, self._audio_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        with self.lock:
            self._video_thread = None
            self._audio_thread = None
            return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            cfg = self.config
            v_prob = self.video_smoothed
            a_prob = self.audio_smoothed
            fused = fuse_scores(v_prob, a_prob, w_video=cfg.fusion_w_video)
            self.fused_prob = fused
            v_label, v_conf, v_unc = classify_live(v_prob, cfg.threshold, cfg.uncertain_margin)
            a_label, a_conf, a_unc = classify_live(a_prob, cfg.threshold, cfg.uncertain_margin)
            f_label, f_conf, f_unc = classify_live(fused, cfg.threshold, cfg.uncertain_margin)
            return {
                "running": self.running,
                "threshold": cfg.threshold,
                "margin": cfg.uncertain_margin,
                "fusion_w_video": cfg.fusion_w_video,
                "video": {
                    "prob": v_prob, "last_prob": self.video_last_prob,
                    "label": v_label, "confidence": v_conf, "uncertain": v_unc,
                    "faces": self.video_faces_detected, "frames": self.video_frame_count,
                    "fps": self.video_fps, "latency_ms": self.video_latency_ms,
                    "box": self.video_last_box, "error": self.video_error,
                    "history": list(self.video_prob_history),
                    "rejected_small": self.video_rejected_small,
                    "rejected_blur": self.video_rejected_blur,
                    "last_reject": self.video_last_reject,
                    "last_blur": self.video_last_blur,
                },
                "audio": {
                    "prob": a_prob, "last_prob": self.audio_last_prob,
                    "label": a_label if self.audio_available else "UNAVAILABLE",
                    "confidence": a_conf, "uncertain": a_unc,
                    "level": self.audio_level, "rms": self.audio_rms,
                    "silence": self.audio_silence, "analyses": self.audio_analyses,
                    "segments": [s.__dict__ for s in list(self.audio_segments)],
                    "waveform": list(self.audio_waveform),
                    "error": self.audio_error, "available": self.audio_available,
                    "history": list(self.audio_prob_history),
                },
                "fusion": {
                    "prob": fused, "label": f_label if fused is not None else "NO_SIGNAL",
                    "confidence": f_conf, "uncertain": f_unc,
                },
                "uptime_sec": (self.session_end or time.time()) - self.session_start
                if self.session_start else 0.0,
            }


# ---------------------------------------------------------------------------
# Shared webcam frame pipeline (worker + offline tests use the same code)
# ---------------------------------------------------------------------------

def crop_face_with_margin(
    frame_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    margin: float = 0.2,
) -> Optional[np.ndarray]:
    """Crop a face box with margin, clipped to the frame.

    Matches FaceDetector.crop_largest() used by the app image path so the
    webcam sees the same framing the image pipeline was validated with.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = frame_bgr.shape[:2]
    mx = int((x1 - x0) * margin)
    my = int((y1 - y0) * margin)
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w, x1 + mx)
    y1 = min(h, y1 + my)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    return crop if crop is not None and crop.size > 0 else None


def face_blur_score(face_bgr: np.ndarray) -> float:
    """Sharpness of a face crop (variance of Laplacian). Higher = sharper."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_face_quality(
    face_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    min_face_size: int = 80,
    blur_threshold: float = 100.0,
) -> Tuple[bool, Optional[str], Optional[float]]:
    """Return (ok, reject_reason, blur_score).

    Frames that are too small or blurred are rejected BEFORE inference so a
    stray low-quality crop can never drive a confident REAL/FAKE verdict.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    if min(x1 - x0, y1 - y0) < min_face_size:
        return False, "too_small", None
    blur = face_blur_score(face_bgr)
    if blur < blur_threshold:
        return False, "blurred", blur
    return True, None, blur


def process_video_frame(
    session: LiveSession,
    frame_bgr: np.ndarray,
    model,
    device: str,
    transform,
    face_detector,
    frame_idx: int,
) -> Dict[str, Any]:
    """Run one webcam frame through detect -> quality-gate -> predict.

    Updates the session's rolling history on accepted frames and returns
    {"raw_prob", "smoothed", "box", "reject"} for logging/tests.
    """
    cfg = session.config
    try:
        from src.realtime_webcam import detect_largest_face, predict_fake_prob
    except Exception as exc:
        with session.lock:
            session.video_error = f"Video inference import failed: {exc}"
        return {"raw_prob": None, "smoothed": None, "box": None,
                "reject": "import_error"}

    try:
        box = detect_largest_face(frame_bgr, face_detector)
    except Exception:
        box = None

    raw_prob = None
    reject = None
    if box is not None and (frame_idx % max(1, cfg.infer_every_n) == 0):
        face = crop_face_with_margin(frame_bgr, box, margin=cfg.face_margin)
        if face is None:
            reject = "bad_crop"
        else:
            ok, reason, blur = check_face_quality(
                face, box, min_face_size=cfg.min_face_size,
                blur_threshold=cfg.blur_threshold)
            with session.lock:
                session.video_last_blur = blur
                session.video_last_reject = reason
            if not ok:
                reject = reason
                with session.lock:
                    if reason == "too_small":
                        session.video_rejected_small += 1
                    elif reason == "blurred":
                        session.video_rejected_blur += 1
            else:
                try:
                    # predict_fake_prob handles BGR->RGB + Resize(299) +
                    # ImageNet norm — the exact training preprocessing.
                    raw_prob = float(predict_fake_prob(face, model, device,
                                                      transform))
                    with session.lock:
                        session.video_last_prob = raw_prob
                        session.video_prob_history.append(raw_prob)
                        session.video_faces_detected += 1
                except Exception:
                    reject = "inference_error"

    with session.lock:
        session.video_last_box = box
        session.video_smoothed = (
            float(np.mean(session.video_prob_history))
            if session.video_prob_history else None
        )
        session.video_frame_count += 1
        smoothed = session.video_smoothed
    return {"raw_prob": raw_prob, "smoothed": smoothed, "box": box,
            "reject": reject}


# ---------------------------------------------------------------------------
# Background workers (import heavy deps lazily so module import never fails)
# ---------------------------------------------------------------------------

def _video_worker(session: LiveSession, bundle: Optional[dict]) -> None:
    cfg = session.config
    if bundle is None:
        with session.lock:
            session.video_error = "Video model could not be loaded."
        return
    model = bundle.get("model")
    device = bundle.get("device", "cpu")
    transform = bundle.get("transform")
    face_detector = bundle.get("face_detector")
    if model is None or transform is None or face_detector is None:
        with session.lock:
            session.video_error = "Video model bundle incomplete."
        return

    # Local imports so unit tests without torch still import this module.
    try:
        from src.realtime_webcam import detect_largest_face, predict_fake_prob
    except Exception as exc:
        with session.lock:
            session.video_error = f"Video inference import failed: {exc}"
        return

    cap = None
    try:
        cap = cv2.VideoCapture(int(cfg.camera_index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(cfg.camera_index))
        if not cap.isOpened():
            with session.lock:
                session.video_error = (
                    f"Could not open camera index {cfg.camera_index}. "
                    "Try another camera or check OS permissions."
                )
            return
    except Exception as exc:
        with session.lock:
            session.video_error = f"Could not open camera: {exc}"
        return

    fps_frames = 0
    fps_timer = time.time()
    frame_idx = 0
    try:
        while not session._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                with session.lock:
                    session.video_error = "Camera frame read failed."
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            # Shared pipeline: detect -> margin crop -> quality gate ->
            # BGR->RGB + Resize(299) + ImageNet norm -> rolling average.
            result = process_video_frame(
                session, frame, model, device, transform,
                face_detector, frame_idx)
            box = result["box"]
            with session.lock:
                session.video_error = None
                latency_ms = (time.perf_counter() - t0) * 1000.0
                # Smooth latency display with EMA.
                prev = session.video_latency_ms or latency_ms
                session.video_latency_ms = 0.7 * prev + 0.3 * latency_ms
                try:
                    annotated = annotate_frame(
                        frame.copy(), box, session.video_smoothed,
                        cfg.threshold, cfg.uncertain_margin,
                        fps=session.video_fps, latency_ms=session.video_latency_ms,
                    )
                    ok_enc, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok_enc:
                        session.latest_frame_jpeg = bytes(buf.tobytes())
                except Exception:
                    pass
            fps_frames += 1
            now = time.time()
            if now - fps_timer >= 0.5:
                with session.lock:
                    session.video_fps = fps_frames / max(now - fps_timer, 1e-6)
                fps_timer = now
                fps_frames = 0
            frame_idx += 1
    finally:
        try:
            cap.release()
        except Exception:
            pass


def _audio_worker(session: LiveSession, bundle: Optional[dict]) -> None:
    cfg = session.config
    if bundle is None:
        with session.lock:
            session.audio_error = "Audio model unavailable."
        return
    model = bundle.get("model")
    device = bundle.get("device", "cpu")
    sample_rate = int(bundle.get("sample_rate", 16000))
    window_sec = float(bundle.get("duration", 4.0))
    if model is None:
        with session.lock:
            session.audio_error = "Audio model could not be loaded."
        return
    try:
        from src.audio_infer import predict_from_array
    except Exception as exc:
        with session.lock:
            session.audio_error = f"Audio inference import failed: {exc}"
        return
    try:
        import sounddevice as sd
    except ImportError:
        with session.lock:
            session.audio_error = "sounddevice not installed — mic capture unavailable."
        return

    window_samples = int(sample_rate * window_sec)
    hop_samples = int(sample_rate * max(cfg.audio_hop_sec, 0.25))
    ring = np.zeros(window_samples, dtype=np.float32)
    t_start = time.time()

    def _callback(indata, frames, ctime, status):
        nonlocal ring, t_start
        if session._stop_event.is_set():
            return
        try:
            block = np.asarray(indata, dtype=np.float32).reshape(-1)
        except Exception:
            return
        if block.size == 0:
            return
        # Update live level + waveform strip from the raw block.
        rms_block = float(np.sqrt(np.mean(block ** 2)) + 1e-9)
        with session.lock:
            session.audio_rms = rms_block
            session.audio_level = float(min(rms_block * 10.0, 1.0))
            session.audio_silence = rms_block < cfg.silence_rms
            down = block[:: max(1, len(block) // 32)]
            session.audio_waveform.extend([float(v) for v in down])
        # Slide ring buffer.
        if len(block) >= len(ring):
            ring = block[-len(ring):].astype(np.float32)
        else:
            ring = np.concatenate([ring[len(block):], block]).astype(np.float32)

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                            callback=_callback,
                            device=cfg.mic_index if cfg.mic_index is not None else None,
                            blocksize=hop_samples):
            last_infer = time.time()
            while not session._stop_event.is_set():
                time.sleep(0.1)
                if time.time() - last_infer < cfg.audio_hop_sec:
                    continue
                last_infer = time.time()
                with session.lock:
                    snap_ring = ring.copy()
                    rms_win = float(np.sqrt(np.mean(snap_ring ** 2)) + 1e-9)
                if rms_win < cfg.silence_rms:
                    with session.lock:
                        session.audio_silence = True
                    continue
                try:
                    prob = float(predict_from_array(snap_ring, model, device, sr=sample_rate))
                except Exception as exc:
                    with session.lock:
                        session.audio_error = f"Audio inference failed: {exc}"
                    continue
                v_label, _, _ = classify_live(prob, cfg.threshold, cfg.uncertain_margin)
                with session.lock:
                    session.audio_last_prob = prob
                    session.audio_prob_history.append(prob)
                    session.audio_smoothed = float(np.mean(session.audio_prob_history))
                    session.audio_analyses += 1
                    session.audio_silence = False
                    session.audio_error = None
                    now = time.time()
                    session.audio_segments.append(AudioSegment(
                        t_start=now - window_sec, t_end=now,
                        fake_prob=prob, label=v_label, rms=rms_win,
                    ))
    except Exception as exc:
        with session.lock:
            session.audio_error = (
                f"Microphone unavailable (device={cfg.mic_index}): {exc}. "
                "Check OS mic permissions / device selection."
            )
