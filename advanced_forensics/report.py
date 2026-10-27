"""
advanced_forensics.report
=========================

Unified forensic report schema.  Pure data container -- no models, no
I/O, no downloads.  Every score field is ``Optional[float]`` in [0, 1]
("more suspicious / fake-like"); unavailable signals stay ``None`` and
are listed in :attr:`unavailable` so readers always know what is known
versus unknown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ForensicReport:
    """One unified forensic assessment for a clip/frame set."""

    source: str = "unknown"               # file name or stream id
    media_kind: str = "video"             # "video" | "image-frames" | "webcam"

    # --- component scores (None = signal unavailable) ---
    xception_probability: Optional[float] = None
    landmark_score: Optional[float] = None
    blink_ear_consistency: Optional[float] = None  # blinks/min or None
    facial_motion_score: Optional[float] = None
    frequency_score: Optional[float] = None
    temporal_score: Optional[float] = None
    transformer_score: Optional[float] = None
    rppg_score: Optional[float] = None
    rppg_bpm: Optional[float] = None

    # --- fusion outputs ---
    ensemble_score: Optional[float] = None
    final_verdict: str = "UNCERTAIN"      # REAL | FAKE | UNCERTAIN
    confidence: float = 0.0               # reliability-scaled, never inflated
    reliability: float = 0.0

    # --- uncertainty / reliability indicators ---
    unavailable: List[str] = field(default_factory=list)
    signal_notes: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        """JSON-serialisable dict of the full report."""
        return asdict(self)

    def summary(self) -> str:
        """One-line human summary, honest about gaps."""
        base = (f"{self.source}: {self.final_verdict} "
                f"(score={self.ensemble_score}, "
                f"confidence={self.confidence:.2f})")
        if self.unavailable:
            base += f" [unavailable: {', '.join(self.unavailable)}]"
        return base

    @classmethod
    def from_signals(cls, source: str, signals: Dict[str, dict],
                     ensemble_result: dict,
                     media_kind: str = "video") -> "ForensicReport":
        """Assemble a report from per-module signal dicts + ensemble out."""
        def _score(name: str) -> Optional[float]:
            sig = (signals or {}).get(name) or {}
            if not sig.get("available"):
                return None
            try:
                return float(sig.get("score"))
            except (TypeError, ValueError):
                return None

        unavailable = sorted(
            k for k, v in (signals or {}).items()
            if not (v or {}).get("available"))
        notes = {k: list((v or {}).get("notes", []))
                 for k, v in (signals or {}).items()
                 if (v or {}).get("notes")}
        notes.update({f"ensemble": list(ensemble_result.get("notes", []))})
        warnings = []
        if unavailable:
            warnings.append("Missing signals reduce reliability; treat the "
                            "verdict as partial evidence.")
        if ((signals or {}).get("rppg") or {}).get("available"):
            warnings.append("rPPG is experimental and never proves "
                            "genuineness.")
        blink_info = ((signals or {}).get("motion") or {}).get("blinks") or {}
        bpm = ((signals or {}).get("rppg") or {}).get("bpm")
        return cls(
            source=source, media_kind=media_kind,
            xception_probability=_score("xception"),
            landmark_score=_score("landmark"),
            blink_ear_consistency=(
                float(blink_info.get("blinks_per_min"))
                if blink_info.get("available") else None),
            facial_motion_score=_score("motion"),
            frequency_score=_score("frequency"),
            temporal_score=_score("temporal"),
            transformer_score=_score("transformer"),
            rppg_score=_score("rppg"),
            rppg_bpm=float(bpm) if bpm is not None else None,
            ensemble_score=ensemble_result.get("score"),
            final_verdict=ensemble_result.get("verdict", "UNCERTAIN"),
            confidence=float(ensemble_result.get("confidence", 0.0)),
            reliability=float(ensemble_result.get("reliability", 0.0)),
            unavailable=unavailable, signal_notes=notes, warnings=warnings)
