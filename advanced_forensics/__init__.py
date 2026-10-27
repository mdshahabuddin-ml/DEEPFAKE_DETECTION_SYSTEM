"""
advanced_forensics
==================

Isolated experimental forensic signals for the Deepfake Detection System.

 staring rule: this package NEVER modifies existing behaviour.  It only
reads the local environment, local checkpoints and local media, and it
never downloads datasets, weights or packages.

Module overview
---------------
- ``landmarks``            5-point facial landmarks via the already-installed
                           ``facenet-pytorch`` MTCNN (+ generic Nx2 geometry
                           helpers; EAR only where 6-point eyes are supplied).
- ``motion``               temporal landmark / openness analysis, blink rate,
                           instability signals (interpretable, no verdict).
- ``frequency``            FFT / DCT / spectral artifact features (NumPy /
                           OpenCV / SciPy only) + uncalibrated heuristic score.
- ``temporal``             per-frame Xception aggregation reusing the local
                           checkpoint; LSTM architecture stub (untrained).
- ``transformer_adapter``  audit + adapter interface; reports unavailable
                           (no local transformer weights found, nothing
                           downloaded).
- ``rppg``                 EXPERIMENTAL remote-PPG liveness pipeline with
                           quality gates; never proof of genuineness.
- ``ensemble``             transparent weighted fusion -> REAL / FAKE /
                           UNCERTAIN with reliability-scaled confidence.
- ``report``               unified :class:`ForensicReport` result schema.

Signal convention: every analyzer returns a dict with at least
``{"available": bool, "score": float | None, ...}`` where ``score`` in
[0, 1] means "more suspicious / fake-like".  Scores from heuristic
sources are explicitly flagged ``uncalibrated_heuristic=True`` and get
small ensemble weights.
"""

from __future__ import annotations

__all__ = [
    "availability_summary",
    "AUDIT_NOTES",
]

# Read-only audit findings (2026-10-09, local .venv, no installs/downloads):
# - numpy/cv2/scipy/torch/timm/sklearn/librosa/facenet-pytorch: installed.
# - dlib / mediapipe / transformers: NOT installed (documented, not installed).
# - models/: only Xception / EfficientNet / audio .pth (no ViT/Swin/SIGLIP).
# - MTCNN inits and runs locally with no download (verified).
# - Existing feature areas (temporal_analysis/, media_forensics/,
#   multimodal_fusion/, src/forensic_report*.py, src/trust_score.py,
#   src/unified_orchestrator.py) were inspected read-only and are untouched.
AUDIT_NOTES = (
    "dlib/mediapipe/transformers are not installed; nothing was installed. "
    "No local transformer weights exist; nothing was downloaded. "
    "MTCNN 5-point landmarks work offline via facenet-pytorch. "
    "Frequency/rPPG use numpy/cv2/scipy only. "
    "Temporal reuses the local Xception checkpoint read-only."
)


def availability_summary() -> dict:
    """Collect per-module availability without heavy model initialisation."""
    from advanced_forensics import (
        frequency,
        landmarks,
        rppg,
        temporal,
        transformer_adapter,
    )

    return {
        "landmarks": landmarks.availability(),
        "frequency": frequency.availability(),
        "temporal": temporal.availability(),
        "transformer": transformer_adapter.audit_transformers(),
        "rppg": rppg.availability(),
        "audit_notes": AUDIT_NOTES,
    }
