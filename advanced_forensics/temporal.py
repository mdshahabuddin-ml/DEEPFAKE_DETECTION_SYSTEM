"""
advanced_forensics.temporal
============================

Temporal video analysis reusing the LOCAL Xception checkpoint read-only
(no downloads, no retraining).

Primary method (used by the ensemble): lightweight aggregation of
per-frame fake probabilities -- mean, spread, trend slope, flip-flop
rate and face presence.  Faces use the OpenCV Haar cascade (ships with
OpenCV, fully local); frames without a detectable face fall back to
full-frame scoring, mirroring ``src/inference.py`` behaviour.

LSTM/GRU/RNN: ``torch.nn`` (installed locally) provides the
architecture, so a minimal :class:`TemporalLSTMAggregator` is defined
for future training experiments.  It carries ``trained=False`` and
refuses inference until real trained weights exist -- no pretrained
temporal weights are downloaded, so the ensemble never uses it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.config import CHECKPOINT_PATH, DEVICE, IMG_SIZE
from src.logger import get_logger

logger = get_logger(__name__)

FACE_MIN_SIZE = (60, 60)


def availability() -> dict:
    """Check the temporal backend without loading any model."""
    ckpt = (Path(CHECKPOINT_PATH) if Path(CHECKPOINT_PATH).exists()
            else Path("models/best_xception_multidataset.pth"))
    if not ckpt.exists():
        return {"available": False,
                "reason": "no local Xception checkpoint found; not "
                          "downloading one."}
    return {"available": True,
            "reason": f"local Xception checkpoint at {ckpt} (read-only).",
            "checkpoint": str(ckpt)}


def _resolve_checkpoint() -> Path:
    info = availability()
    if not info["available"]:
        raise FileNotFoundError(info["reason"])
    return Path(info["checkpoint"])


class XceptionFeatureExtractor:
    """Per-frame fake probabilities from the local Xception checkpoint.

    Also exposes penultimate-layer embeddings where the architecture
    allows (best-effort forward hook); embeddings are a bonus, the
    probability track is the primary temporal signal.
    """

    def __init__(self, device: Optional[str] = None):
        from src.checkpoint import safe_torch_load
        from src.model import build_model, load_checkpoint

        self.device = device or DEVICE
        ckpt = _resolve_checkpoint()
        model = build_model("xception", pretrained=False, device="cpu")
        state = safe_torch_load(str(ckpt), map_location="cpu")
        sd = state.get("model_state_dict", state)
        model.load_state_dict(sd)
        model.to(self.device)
        model.eval()
        self.model = model
        self._embed_dim: Optional[int] = None

        cascade_path = (cv2.data.haarcascades
                        + "haarcascade_frontalface_default.xml")
        self.haar = (cv2.CascadeClassifier(cascade_path)
                     if Path(cascade_path).exists() else None)
        if self.haar is not None and self.haar.empty():
            self.haar = None

    def _largest_face(self, bgr: np.ndarray):
        if self.haar is None:
            return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=FACE_MIN_SIZE)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        crop = bgr[y:y + h, x:x + w]
        return crop if crop.size else None

    @torch.no_grad()
    def frame_prob(self, bgr: np.ndarray) -> tuple[float, bool]:
        """(fake_probability, face_found) for one BGR frame."""
        face = self._largest_face(bgr)
        src = face if face is not None else bgr
        rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
        tensor = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
        # ImageNet normalisation, matching src/inference.py.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        logit = self.model(tensor)
        return float(torch.sigmoid(logit).item()), face is not None


class TemporalLSTMAggregator(nn.Module):
    """Minimal LSTM over per-frame embeddings -- ARCHITECTURE ONLY.

    ``torch.nn.LSTM`` exists locally, so the architecture is defined for
    future training experiments.  There are NO trained weights (none are
    downloaded), so :meth:`forward` refuses to run until ``trained=True``
    is set after a real training run.  The ensemble never calls this.
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)
        self.trained = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        if not self.trained:
            raise RuntimeError(
                "TemporalLSTMAggregator is untrained: no local temporal "
                "weights exist and none will be downloaded. Train on local "
                "data first, then set .trained = True.")
        _, (hn, _) = self.lstm(x)
        return self.head(hn[-1])


def aggregate_track(probs: List[float], faces: List[bool]) -> dict:
    """Lightweight, transparent aggregation of a probability track."""
    p = np.asarray(list(probs or []), dtype=np.float64)
    if p.size == 0:
        return {"available": False, "reason": "empty probability track"}
    t = np.arange(p.size, dtype=np.float64)
    slope = float(np.polyfit(t, p, 1)[0]) if p.size >= 3 else 0.0
    hard = (p >= 0.5).astype(int)
    flips = float(np.abs(np.diff(hard)).sum() / max(len(hard) - 1, 1))
    face_ratio = float(np.mean(faces)) if faces else 0.0
    return {"available": True,
            "mean": float(p.mean()), "std": float(p.std()),
            "min": float(p.min()), "max": float(p.max()),
            "trend_slope": slope, "flip_rate": flips,
            "face_ratio": face_ratio, "n_frames": int(p.size)}


def _squash(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def temporal_report(frames_bgr: List[np.ndarray],
                    extractor: Optional[XceptionFeatureExtractor] = None,
                    max_frames: int = 32) -> dict:
    """Score clip-level temporal suspiciousness in [0, 1] (heuristic).

    Runs the local Xception model on up to ``max_frames`` evenly sampled
    frames, then aggregates.  Short clips (<3 usable frames) and clips
    with no usable frames return ``available=False`` with a reason --
    never a fabricated score.
    """
    frames = [f for f in (frames_bgr or [])
              if isinstance(f, np.ndarray) and f.size > 0]
    if len(frames) < 1:
        return {"available": False, "score": None,
                "reason": "no usable frames"}
    if extractor is None:
        try:
            extractor = XceptionFeatureExtractor()
        except FileNotFoundError as exc:
            return {"available": False, "score": None, "reason": str(exc)}
    idx = np.linspace(0, len(frames) - 1,
                      min(max_frames, len(frames))).astype(int)
    probs, faces = [], []
    for i in idx:
        try:
            pr, ff = extractor.frame_prob(frames[int(i)])
        except Exception as exc:
            logger.warning("temporal frame %d failed: %s", int(i), exc)
            continue
        probs.append(pr)
        faces.append(ff)
    if len(probs) < 3:
        return {"available": False, "score": None,
                "reason": f"only {len(probs)} usable frames (need >=3)",
                "n_frames": len(probs)}
    agg = aggregate_track(probs, faces)
    # Heuristic: flickering tracks (high spread + flips + trend) look
    # more suspicious than flat confident ones.  Uncalibrated.
    raw = agg["std"] * 2.2 + agg["flip_rate"] * 0.9 \
        + abs(agg["trend_slope"]) * 8.0
    score = _squash(raw - 1.1)
    return {"available": True, "score": score, "aggregation": agg,
            "frame_probs": [float(v) for v in probs],
            "uncalibrated_heuristic": True,
            "notes": ["Lightweight aggregation over local Xception "
                      "per-frame scores; LSTM path intentionally unused "
                      "(no trained weights)."]}
