"""
advanced_forensics.motion
=========================

Micro-expression / facial-motion analysis over landmark tracks.

Operates on caller-supplied tracks (lists of per-frame Nx2 arrays or
per-frame scalar openness values), so it works with 5-point MTCNN tracks
today and denser tracks later.  Output is interpretable forensic
signals -- displacement statistics, jitter, blink rate, openness
variance -- never a hard-coded REAL/FAKE result.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

MIN_TRACK_FRAMES = 5


def _track_array(track: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Stack a landmark track to (T, N, 2) or return None if unusable."""
    if track is None or len(track) < 2:
        return None
    try:
        arr = np.stack([np.asarray(f, dtype=np.float64) for f in track])
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[2] != 2:
        return None
    return arr


def displacement_stats(track: Sequence[np.ndarray]) -> dict:
    """Per-point frame-to-frame displacement mean/std (pixels/frame)."""
    arr = _track_array(track)
    if arr is None:
        return {"available": False, "reason": "need >=2 frames of Nx2 points"}
    disp = np.linalg.norm(np.diff(arr, axis=0), axis=2)  # (T-1, N)
    return {"available": True,
            "mean": float(disp.mean()), "std": float(disp.std()),
            "max": float(disp.max()),
            "per_point_mean": [float(v) for v in disp.mean(axis=0)]}


def jitter_index(track: Sequence[np.ndarray]) -> dict:
    """High-frequency jitter: energy of 2nd differences vs 1st (0..~1).

    Smooth natural motion has small 2nd differences; erratic frame-wise
    landmark wobble (common in face-swap pipelines) raises this ratio.
    """
    arr = _track_array(track)
    if arr is None or len(arr) < MIN_TRACK_FRAMES:
        return {"available": False,
                "reason": f"need >={MIN_TRACK_FRAMES} frames"}
    vel = np.diff(arr, axis=0)
    acc = np.diff(vel, axis=0)
    denom = float(np.mean(vel ** 2)) + 1e-9
    ratio = float(np.mean(acc ** 2)) / denom
    # Map to [0, 1): ratio ~0.2 is smooth, >=2 is very jittery (heuristic).
    score = float(ratio / (ratio + 1.2))
    return {"available": True, "score": score, "ratio": ratio,
            "uncalibrated_heuristic": True}


def analyze_blinks(ear_seq: Sequence[float], fps: float = 30.0,
                   thresh: float = 0.22, min_gap_sec: float = 0.25) -> dict:
    """Blink count/rate from an eye-aspect-ratio sequence.

    Requires a genuine EAR sequence (6-point eyes).  Returns
    ``available=False`` for empty input; callers must NOT fabricate EAR
    from 5-point MTCNN output (see landmarks.ear_supported).
    """
    seq = np.asarray(list(ear_seq) if ear_seq is not None else [],
                       dtype=np.float64)
    if seq.size < 3 or not np.isfinite(seq).all():
        return {"available": False,
                "reason": "EAR sequence too short/invalid; blink analysis "
                          "unsupported without 6-point eye landmarks."}
    below = seq < thresh
    # Count threshold-crossing episodes separated by min_gap_sec.
    min_gap = max(1, int(round(min_gap_sec * fps)))
    blinks, i, last_end = 0, 0, -10 ** 9
    n = len(below)
    while i < n:
        if below[i] and i - last_end >= min_gap:
            j = i
            while j + 1 < n and below[j + 1]:
                j += 1
            blinks += 1
            last_end = j
            i = j + 1
        else:
            i += 1
    duration_min = len(seq) / max(float(fps), 1e-6) / 60.0
    rate = blinks / max(duration_min, 1e-6)
    notes = []
    if not (5.0 <= rate <= 40.0):
        notes.append(f"Blink rate {rate:.1f}/min is outside the typical "
                     "resting range (5-40/min); treat as a weak signal only.")
    return {"available": True, "blinks": int(blinks),
            "blinks_per_min": float(rate),
            "mean_ear": float(seq.mean()),
            "notes": notes}


def openness_stats(seq: Sequence[float], name: str) -> dict:
    """Mean/variance/range of a per-frame openness sequence (eye/mouth)."""
    arr = np.asarray(list(seq) if seq is not None else [], dtype=np.float64)
    if arr.size < 3 or not np.isfinite(arr).all():
        return {"available": False, "reason": f"{name}: need >=3 samples"}
    return {"available": True, "name": name,
            "mean": float(arr.mean()), "std": float(arr.std()),
            "range": float(arr.max() - arr.min())}


def motion_report(track: Sequence[np.ndarray],
                  ear_seq: Optional[Sequence[float]] = None,
                  fps: float = 30.0) -> dict:
    """Aggregate interpretable motion signals + instability score.

    ``score`` in [0, 1] (higher = more temporally unstable) is an
    UNCALIBRATED heuristic blend of jitter and openness variance for
    ensemble input -- not a verdict.
    """
    disp = displacement_stats(track)
    jit = jitter_index(track)
    ear = analyze_blinks(ear_seq or [], fps=fps) if ear_seq else {
        "available": False,
        "reason": "no EAR sequence supplied (needs 6-point eyes)."}
    parts, weights = [], []
    if jit.get("available"):
        parts.append(jit["score"]); weights.append(0.6)
    if disp.get("available"):
        # Normalise mean displacement: ~0.5px calm, >=6px frantic (heuristic).
        parts.append(min(1.0, disp["mean"] / 6.0)); weights.append(0.4)
    score = (float(np.average(parts, weights=weights))
             if parts else None)
    return {"available": score is not None, "score": score,
            "displacement": disp, "jitter": jit, "blinks": ear,
            "uncalibrated_heuristic": True,
            "notes": ["Interpretable motion signals only; no REAL/FAKE "
                      "decision is made here."]}
