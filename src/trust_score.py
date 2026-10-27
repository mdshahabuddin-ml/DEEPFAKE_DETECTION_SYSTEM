"""
src/trust_score.py
AI Trust Score — multi-signal authenticity assessment.

Instead of a single REAL/FAKE label, this module fuses several independent
signals into a 0–100 "Trust Score" plus a recommendation, turning the system
into a richer decision-support tool (the kind of report a KYC/forensics product
would surface).

Signals (each in [0, 1], higher = more authentic):
  • face_consistency     : 1 − model fake-probability on the detected face
  • temporal_consistency : frame-to-frame agreement across a video
  • lip_sync             : agreement between the visual and voice verdicts
                           (audio/video fusion; None if there's no audio track)
  • texture              : spectral-artifact proxy — GAN/deepfake faces often
                           have abnormal high-frequency energy distributions

The overall score is a weighted average over whichever signals are available.
All signals are real, computed quantities; weights are documented and tunable.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.config import DEFAULT_THRESHOLD
from src.logger import get_logger

logger = get_logger(__name__)

# Relative weights for the overall score (normalized over available signals).
SIGNAL_WEIGHTS = {
    "face_consistency": 0.40,
    "temporal_consistency": 0.25,
    "lip_sync": 0.20,
    "texture": 0.15,
}


@dataclass
class TrustReport:
    """A full authenticity assessment for one image or video."""
    media_type: str                       # "image" | "video"
    file_name: str
    verdict: str                          # "Likely Authentic" | "Uncertain" | "Likely Deepfake"
    trust_score: float                    # 0–100
    recommendation: str
    fake_probability: float               # primary visual fake prob [0,1]
    signals: Dict[str, Optional[float]] = field(default_factory=dict)   # sub-scores [0,1]
    faces_detected: int = 0
    manipulated_frames: Optional[List[int]] = None   # frame indices voted FAKE (video)
    frames_analyzed: int = 0
    file_hash: str = ""
    inference_time: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Individual signal computations
# ---------------------------------------------------------------------------
def spectral_texture_score(face_bgr: np.ndarray) -> float:
    """
    Spectral-artifact texture score in [0, 1].

    Natural photos carry a characteristic spread of mid/high-frequency energy;
    many synthesized faces are over-smooth or carry periodic upsampling
    artifacts. We measure the fraction of spectral energy outside a central
    low-frequency window and map it to a bounded score. Higher = more natural.
    """
    if face_bgr is None or face_bgr.size == 0:
        return 0.5
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256)).astype(np.float32)

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    total = float(spectrum.sum()) + 1e-8
    cy, cx = 128, 128
    r = 24  # central low-frequency radius
    low = float(spectrum[cy - r:cy + r, cx - r:cx + r].sum())
    high_ratio = 1.0 - (low / total)     # energy outside the low-freq core

    # Empirically, natural faces sit around ~0.6–0.85 here; normalize to [0,1].
    score = (high_ratio - 0.30) / (0.85 - 0.30)
    return float(np.clip(score, 0.0, 1.0))


def temporal_consistency_score(frame_probs: List[float]) -> Optional[float]:
    """
    Frame-to-frame consistency in [0, 1] from per-frame fake-probabilities.

    Authentic clips give a stable, low-variance signal; deepfakes often flicker
    as manipulation quality varies frame to frame. We penalize both variance and
    the mean jump between consecutive frames. None if < 2 frames.
    """
    if not frame_probs or len(frame_probs) < 2:
        return None
    arr = np.asarray(frame_probs, dtype=np.float32)
    std = float(np.std(arr))
    mean_jump = float(np.mean(np.abs(np.diff(arr))))
    instability = 0.6 * std + 0.4 * mean_jump   # both in ~[0,1]
    return float(np.clip(1.0 - 2.0 * instability, 0.0, 1.0))


def lip_sync_score(video_prob: float, audio_prob: Optional[float]) -> Optional[float]:
    """
    Cross-modal agreement in [0, 1] between the visual and voice verdicts.

    A genuine clip should have consistent face and voice authenticity; a strong
    disagreement (real face + fake voice, or vice-versa) is a classic voice-swap
    / lip-sync manipulation signature. Score = 1 − |video_prob − audio_prob|.
    None when there's no usable audio track.
    """
    if audio_prob is None:
        return None
    return float(np.clip(1.0 - abs(float(video_prob) - float(audio_prob)), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_trust(signals: Dict[str, Optional[float]]) -> float:
    """Weighted average (0–100) over available signals."""
    num = den = 0.0
    for name, weight in SIGNAL_WEIGHTS.items():
        val = signals.get(name)
        if val is not None:
            num += weight * float(val)
            den += weight
    if den == 0:
        return 50.0
    return round(100.0 * num / den, 1)


def _verdict_and_reco(trust_score: float) -> tuple[str, str]:
    if trust_score >= 75:
        return "Likely Authentic", "No manipulation indicators above threshold — routine approval."
    if trust_score >= 40:
        return "Uncertain", "Mixed signals — manual verification advised before approval."
    return "Likely Deepfake", "Strong manipulation indicators — reject / escalate for review."


def file_sha256(path) -> str:
    """SHA-256 of a file, for the report's tamper-evidence / content passport."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # pragma: no cover
        return ""


# ---------------------------------------------------------------------------
# High-level analyzers (tie signals to the detector)
# ---------------------------------------------------------------------------
def analyze_image(detector, bgr_image: np.ndarray, file_name: str = "image",
                  file_path: Optional[str] = None) -> TrustReport:
    """Build a TrustReport for a single image."""
    import time
    start = time.perf_counter()

    face = detector.face_detector.crop_largest(bgr_image)
    if face is None:
        return TrustReport(
            media_type="image", file_name=file_name, verdict="No Face",
            trust_score=0.0, recommendation="No face detected — cannot assess.",
            fake_probability=0.0, signals={}, faces_detected=0,
            file_hash=file_sha256(file_path) if file_path else "",
            inference_time=round(time.perf_counter() - start, 4),
        )

    faces = detector.face_detector.detect(bgr_image)
    prob = detector._predict_face(face)
    signals: Dict[str, Optional[float]] = {
        "face_consistency": 1.0 - prob,
        "temporal_consistency": None,           # N/A for a single image
        "lip_sync": None,                        # N/A (no audio)
        "texture": spectral_texture_score(face),
    }
    trust = aggregate_trust(signals)
    verdict, reco = _verdict_and_reco(trust)
    return TrustReport(
        media_type="image", file_name=file_name, verdict=verdict,
        trust_score=trust, recommendation=reco, fake_probability=round(prob, 6),
        signals={k: (round(v, 4) if v is not None else None) for k, v in signals.items()},
        faces_detected=len(faces), frames_analyzed=1,
        file_hash=file_sha256(file_path) if file_path else "",
        inference_time=round(time.perf_counter() - start, 4),
    )


def analyze_video(detector, video_path: str, file_name: Optional[str] = None,
                  max_frames: Optional[int] = None) -> TrustReport:
    """Build a TrustReport for a video (uses audio for lip-sync when available)."""
    import time
    start = time.perf_counter()
    file_name = file_name or Path(video_path).name

    vres = (detector.predict_video(video_path, max_frames=max_frames)
            if max_frames is not None else detector.predict_video(video_path))

    if vres.get("frames_with_face", 0) == 0:
        return TrustReport(
            media_type="video", file_name=file_name, verdict="No Face",
            trust_score=0.0, recommendation="No face detected — cannot assess.",
            fake_probability=0.0, signals={}, faces_detected=0,
            frames_analyzed=vres.get("frames_sampled", 0),
            file_hash=file_sha256(video_path),
            inference_time=round(time.perf_counter() - start, 4),
        )

    frame_probs = vres.get("frame_probabilities", [])
    video_prob = vres["fake_probability"]

    # Optional voice signal for lip-sync agreement.
    audio_prob = _try_audio_prob(video_path)

    threshold = getattr(detector, "threshold", DEFAULT_THRESHOLD)
    manipulated = [i for i, p in enumerate(frame_probs) if p >= threshold]

    signals: Dict[str, Optional[float]] = {
        "face_consistency": 1.0 - video_prob,
        "temporal_consistency": temporal_consistency_score(frame_probs),
        "lip_sync": lip_sync_score(video_prob, audio_prob),
        "texture": None,  # aggregate-level; per-frame texture omitted for speed
    }
    trust = aggregate_trust(signals)
    verdict, reco = _verdict_and_reco(trust)
    return TrustReport(
        media_type="video", file_name=file_name, verdict=verdict,
        trust_score=trust, recommendation=reco, fake_probability=round(video_prob, 6),
        signals={k: (round(v, 4) if v is not None else None) for k, v in signals.items()},
        faces_detected=vres.get("frames_with_face", 0),
        manipulated_frames=manipulated,
        frames_analyzed=vres.get("frames_sampled", 0),
        file_hash=file_sha256(video_path),
        inference_time=round(time.perf_counter() - start, 4),
    )


def _try_audio_prob(video_path: str) -> Optional[float]:
    """Best-effort voice fake-probability from the video's audio track."""
    try:
        from src.config import CHECKPOINT_PATH  # noqa: F401  (ensures config import)
        from src.audio_infer import DEFAULT_AUDIO_CHECKPOINT, load_audio_detector, predict_from_array
        from src.audio_dataset import SAMPLE_RATE
        from src.fusion import load_audio_from_video

        if not Path(DEFAULT_AUDIO_CHECKPOINT).exists():
            return None
        wav = load_audio_from_video(video_path, SAMPLE_RATE)
        if wav is None:
            return None
        model, device = load_audio_detector()
        return float(predict_from_array(wav, model, device))
    except Exception as exc:  # pragma: no cover - audio is optional
        logger.info("Audio signal unavailable for lip-sync: %s", exc)
        return None
