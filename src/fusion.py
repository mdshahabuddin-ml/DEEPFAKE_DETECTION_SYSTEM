"""
src/fusion.py
Late-fusion helpers that combine the visual (face) deepfake score and the
audio (voice) deepfake score for a single media file into one verdict.

Both scores are fake-probabilities in [0, 1]. Fusion is a weighted average of
whichever modalities are available, so a clip with no detectable face still
yields an audio-only verdict (and vice-versa).
"""

from pathlib import Path
from typing import Optional

import numpy as np


def fuse_scores(video_prob: Optional[float], audio_prob: Optional[float],
                w_video: float = 0.5) -> Optional[float]:
    """
    Weighted-average fusion of the two fake-probabilities.

    w_video is the weight given to the visual score (audio gets 1 - w_video).
    Missing modalities (None) are dropped and the remaining weights renormalize.
    Returns None only if both inputs are None.
    """
    w_video = min(max(w_video, 0.0), 1.0)
    parts, weights = [], []
    if video_prob is not None:
        parts.append(float(video_prob))
        weights.append(w_video)
    if audio_prob is not None:
        parts.append(float(audio_prob))
        weights.append(1.0 - w_video)

    total = sum(weights)
    if not parts or total == 0:
        # e.g. both present but w_video collapsed a single available side to 0
        return float(np.mean(parts)) if parts else None
    return sum(p * w for p, w in zip(parts, weights)) / total


def load_audio_from_video(video_path, sr: int) -> Optional[np.ndarray]:
    """
    Extract the audio track from a video file as a mono waveform at `sr` Hz.

    Uses librosa (which falls back to audioread/ffmpeg for container formats
    like mp4/mov). Returns None if the file has no usable audio track or the
    audio backend (ffmpeg) is unavailable.
    """
    try:
        import librosa
    except ImportError:
        return None

    try:
        y, _ = librosa.load(str(video_path), sr=sr, mono=True)
    except Exception:
        return None

    if y is None or len(y) < int(sr * 0.2):
        return None
    if float(np.max(np.abs(y))) < 1e-4:   # essentially silent / no track
        return None
    return y.astype(np.float32)


def verdict_label(prob: float, threshold: float = 0.5) -> str:
    return "FAKE" if prob >= threshold else "REAL"
