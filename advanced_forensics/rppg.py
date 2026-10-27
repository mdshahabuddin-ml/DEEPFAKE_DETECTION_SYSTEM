"""
advanced_forensics.rppg
=======================

EXPERIMENTAL remote-photoplethysmography (rPPG) liveness analysis.

Method (POS algorithm, NumPy/SciPy only -- both installed locally):
forehead + cheek skin ROIs (relative to the detected face box, no
landmarks required) -> per-frame spatial-mean RGB trace -> detrending
-> POS projection -> 0.7-4.0 Hz bandpower FFT -> heart-rate estimate,
in-band SNR and peak-periodicity metrics.

This is an EXPERIMENTAL liveness *signal*, not a verdict:
- A clean periodic pulse does NOT prove a video is genuine.
- A missing/weak pulse does NOT prove a deepfake (makeup, lighting,
  motion, compression and low frame-rate all destroy the signal).

Quality gates degrade gracefully: too few frames, tiny faces, excessive
motion, bad lighting or heavy downsampling return
``quality in {"poor", "unusable"}`` with reasons, and the ensemble
excludes (or near-zero-weights) the signal.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

MIN_FRAMES = 48          # ~2 s at 24 fps; below this no reliable spectrum
MIN_FACE_PX = 80         # face box side under this -> "small face" penalty
BAND = (0.7, 4.0)        # Hz: 42-240 bpm


def availability() -> dict:
    try:
        import numpy  # noqa: F401
        import cv2  # noqa: F401
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}
    try:
        import scipy  # noqa: F401
        detrend = "scipy"
    except ImportError:
        detrend = "numpy-fallback"
    return {"available": True,
            "reason": f"POS pipeline on numpy/cv2 ({detrend} detrend). "
                      "EXPERIMENTAL: never proof of genuineness.",
            "experimental": True}


def skin_rois(box: Sequence[float]) -> List[Tuple[slice, slice]]:
    """Forehead + cheek ROIs as (row, col) slices from an xyxy face box."""
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    rois = [
        (slice(int(y0 + 0.08 * h), int(y0 + 0.28 * h)),   # forehead
         slice(int(x0 + 0.25 * w), int(x0 + 0.75 * w))),
        (slice(int(y0 + 0.45 * h), int(y0 + 0.65 * h)),   # left cheek
         slice(int(x0 + 0.10 * w), int(x0 + 0.32 * w))),
        (slice(int(y0 + 0.45 * h), int(y0 + 0.65 * h)),   # right cheek
         slice(int(x0 + 0.68 * w), int(x0 + 0.90 * w))),
    ]
    return rois


def extract_rgb_trace(frames: Sequence[np.ndarray],
                      boxes: Sequence[Sequence[float]]) -> dict:
    """Spatial-mean RGB trace (T, 3) over skin ROIs + quality notes."""
    if len(frames) < MIN_FRAMES:
        return {"available": False,
                "reason": f"only {len(frames)} frames (need >={MIN_FRAMES})"}
    trace, sizes, brights, shifts = [], [], [], []
    prev_cx = None
    for frame, box in zip(frames, boxes):
        if frame is None or box is None:
            continue
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
        fw, fh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        sizes.append(min(fw, fh))
        cx = (x0 + x1) / 2.0
        if prev_cx is not None:
            shifts.append(abs(cx - prev_cx) / max(fw, 1.0))
        prev_cx = cx
        means = []
        for rs, cs in skin_rois(box):
            r0, r1 = max(rs.start, 0), min(rs.stop, h)
            c0, c1 = max(cs.start, 0), min(cs.stop, w)
            if r1 <= r0 or c1 <= c0:
                continue
            patch = frame[r0:r1, c0:c1].astype(np.float64)
            means.append(patch.reshape(-1, 3).mean(axis=0))
        if not means:
            continue
        rgb = np.mean(means, axis=0)  # BGR means
        brights.append(float(rgb.mean()))
        trace.append(rgb[::-1])  # -> RGB order
    trace = np.asarray(trace)
    if trace.shape[0] < MIN_FRAMES:
        return {"available": False,
                "reason": "too few frames with valid skin ROIs"}
    notes = []
    if float(np.mean(sizes)) < MIN_FACE_PX:
        notes.append("small face: pulse SNR likely degraded")
    motion = float(np.mean(shifts)) if shifts else 0.0
    if motion > 0.05:
        notes.append(f"high head motion ({motion:.3f} face-widths/frame)")
    mean_bright = float(np.mean(brights))
    if mean_bright < 40:
        notes.append("very dark footage: pulse signal unreliable")
    elif mean_bright > 220:
        notes.append("overexposed footage: pulse signal unreliable")
    return {"available": True, "trace": trace, "motion": motion,
            "brightness": mean_bright, "notes": notes}


def _detrend(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import detrend as sp_detrend
        return sp_detrend(x, axis=0, type="linear")
    except ImportError:
        t = np.arange(x.shape[0], dtype=np.float64)
        coef = np.polyfit(t, x, 1)
        return x - (coef[0] * t[:, None] + coef[1])


def pos_pulse(trace: np.ndarray, fps: float) -> dict:
    """POS projection -> band-limited pulse + spectrum metrics."""
    rgb = _detrend(np.asarray(trace, dtype=np.float64))
    n = rgb.shape[0]
    # Normalise each channel (POS Step 1-2).
    mean = rgb.mean(axis=0, keepdims=True)
    cn = rgb / (np.abs(mean) + 1e-9)
    # Projection matrix S = [0,1,-1; -2,1,1] @ cn.T (POS Step 3-4).
    s1 = cn[:, 1] - cn[:, 2]
    s2 = -2.0 * cn[:, 0] + cn[:, 1] + cn[:, 2]
    alpha = float(s1.std() / (s2.std() + 1e-9))
    pulse = s1 + alpha * s2
    pulse = pulse - np.convolve(pulse, np.ones(5) / 5.0, mode="same")
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    spec = np.abs(np.fft.rfft(pulse)) ** 2
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    if not band.any() or spec.sum() <= 0:
        return {"available": False, "reason": "empty spectrum"}
    inband, total = spec[band].sum(), spec.sum()
    peak_idx = int(np.argmax(spec * band))
    bpm = float(freqs[peak_idx] * 60.0)
    snr = float(10.0 * np.log10((inband + 1e-12) / (total - inband + 1e-12)))
    # Periodicity: peak prominence vs in-band median.
    med = float(np.median(spec[band])) + 1e-12
    periodicity = float(spec[peak_idx] / med)
    return {"available": True, "pulse": pulse, "bpm": bpm,
            "snr_db": snr, "periodicity": periodicity,
            "inband_ratio": float(inband / (total + 1e-12))}


def rppg_report(frames: Sequence[np.ndarray],
                boxes: Sequence[Sequence[float]],
                fps: float = 30.0) -> dict:
    """Experimental liveness report.  Never a genuineness verdict."""
    if float(fps) <= 0:
        return {"available": False, "score": None,
                "reason": "non-positive fps", "experimental": True}
    ext = extract_rgb_trace(frames, boxes)
    if not ext.get("available"):
        return {"available": False, "score": None,
                "reason": ext.get("reason"), "experimental": True}
    res = pos_pulse(ext["trace"], float(fps))
    if not res.get("available"):
        return {"available": False, "score": None,
                "reason": res.get("reason"), "experimental": True}
    quality, qnotes = "good", list(ext.get("notes", []))
    if qnotes:
        quality = "poor"
    # Suspiciousness heuristic INVERTED vs other signals: a strong clean
    # pulse is (weak) evidence of liveness -> LOW score; absent pulse ->
    # mid score (never high: absence proves nothing).
    snr_n = max(0.0, min(1.0, (res["snr_db"] + 5.0) / 15.0))
    per_n = max(0.0, min(1.0, (res["periodicity"] - 1.0) / 9.0))
    liveness = 0.6 * snr_n + 0.4 * per_n
    score = float(0.5 * (1.0 - liveness))  # capped at 0.5 by construction
    return {"available": True, "score": score,
            "bpm": res["bpm"], "snr_db": res["snr_db"],
            "periodicity": res["periodicity"],
            "quality": quality, "quality_notes": qnotes,
            "experimental": True, "uncalibrated_heuristic": True,
            "notes": ["EXPERIMENTAL liveness signal: a clean pulse does NOT "
                      "prove the video is genuine; a weak pulse does NOT "
                      "prove a deepfake."]}
