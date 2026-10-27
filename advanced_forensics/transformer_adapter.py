"""
advanced_forensics.transformer_adapter
======================================

Transformer audit + isolated adapter interface.

Audit outcome (local .venv + ./models, read-only):
- ``transformers`` (Hugging Face) is NOT installed.
- ``timm`` IS installed, and ``src/model.py`` even defines a
  ``ViTDetector`` architecture (``vit_base_patch16_224``) -- but
  instantiating it with ``pretrained=True`` would download weights, and
  instantiating it with ``pretrained=False`` yields random weights that
  are useless for forensics.
- ``models/`` contains only Xception / EfficientNet / audio ``.pth``
  files: no ViT, Swin, SIGLIP or CLIP weights exist locally.

Policy outcome: NO download is performed.  The adapter below exposes the
same signal interface as the other modules and reports
``available=False`` with reasons, so the ensemble cleanly excludes it.
To enable later: place trained transformer weights under
``models/`` (e.g. ``models/vit_detector.pth`` in the ``ViTDetector``
state-dict format) and extend :meth:`TransformerAdapter.load` -- the
callers do not need to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.logger import get_logger

logger = get_logger(__name__)

KNOWN_TRANSFORMER_HINTS = ("vit", "swin", "siglip", "clip", "deit",
                           "beit", "levit", "transformer")


def audit_transformers(models_dir: str = "models") -> dict:
    """Inspect local transformer availability.  Never downloads."""
    reasons: List[str] = []
    try:
        import transformers  # noqa: F401
        hf_present = True
    except ImportError:
        hf_present = False
        reasons.append("Hugging Face `transformers` is not installed.")
    try:
        import timm  # noqa: F401
        timm_present = True
    except ImportError:
        timm_present = False
        reasons.append("`timm` is not installed.")

    local_weights: List[str] = []
    root = Path(models_dir)
    if root.exists():
        for ckpt in sorted(root.glob("*.pth")):
            name = ckpt.name.lower()
            if any(h in name for h in KNOWN_TRANSFORMER_HINTS):
                local_weights.append(ckpt.name)
    if not local_weights:
        reasons.append(f"no ViT/Swin/SIGLIP/CLIP weights in {root}/ "
                       f"(found only CNN/audio checkpoints).")

    available = hf_present and bool(local_weights)
    if timm_present and not local_weights:
        reasons.append("timm ViTDetector architecture exists in src/model.py "
                       "but has no local trained weights; random weights "
                       "would be meaningless, so it is NOT used.")
    return {"available": available,
            "hf_transformers": hf_present, "timm": timm_present,
            "local_weights": local_weights,
            "reasons": reasons or ["transformer inference ready (unexpected)."]}


class TransformerAdapter:
    """Interface for a future local transformer signal.

    With no local weights this object reports unavailable and
    :meth:`analyze` returns an unavailable-signal dict instead of
    raising, so pipelines keep working.
    """

    def __init__(self, checkpoint: Optional[str] = None):
        self.audit = audit_transformers()
        self.checkpoint = checkpoint
        self.model = None
        if self.audit["available"] and checkpoint:
            self.load(checkpoint)

    @property
    def available(self) -> bool:
        return bool(self.audit["available"] and self.model is not None)

    def load(self, checkpoint: str):
        """Plug-in point: load LOCAL weights only (never downloading)."""
        raise NotImplementedError(
            "No local transformer checkpoint exists to load. Place trained "
            "weights under models/ first; see module docstring.")

    def analyze(self, _frames=None) -> dict:
        """Return an unavailable-signal dict (never a fabricated score)."""
        return {"available": False, "score": None,
                "reasons": self.audit["reasons"],
                "notes": ["Transformer inference unavailable locally; "
                          "ensemble excludes this signal."]}
