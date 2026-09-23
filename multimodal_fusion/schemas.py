"""Data schemas for multimodal fusion."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConsistencyStatus(Enum):
    """Status for consistency checks."""
    CONSISTENT = "CONSISTENT"
    POTENTIAL_MISMATCH = "POTENTIAL_MISMATCH"
    INCONSISTENT = "INCONSISTENT"
    UNKNOWN = "UNKNOWN"


class FusionVerdict(Enum):
    """Overall fusion verdict."""
    AUTHENTIC = "AUTHENTIC"
    POTENTIALLY_AUTHENTIC = "POTENTIALLY_AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    POTENTIALLY_MANIPULATED = "POTENTIALLY_MANIPULATED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    UNKNOWN = "UNKNOWN"


@dataclass
class Evidence:
    """Single piece of evidence."""
    source: str = ""
    score: float = 0.0
    confidence: float = 0.0
    status: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyResult:
    """Result of consistency check."""
    sync_score: float = 0.0
    status: ConsistencyStatus = ConsistencyStatus.UNKNOWN
    evidence: List[Evidence] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalResult:
    """Result of multimodal analysis."""
    verdict: FusionVerdict = FusionVerdict.UNKNOWN
    confidence: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)
    consistency: Optional[ConsistencyResult] = None
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "scores": self.scores,
            "n_evidence": len(self.evidence),
            "recommendations": self.recommendations,
        }

    def get_summary(self) -> str:
        lines = [
            "=" * 60,
            "MULTIMODAL ANALYSIS REPORT",
            "=" * 60,
            f"Verdict: {self.verdict.value}",
            f"Confidence: {self.confidence:.3f}",
            "",
            "SCORES",
            "-" * 40,
        ]
        for key, val in self.scores.items():
            lines.append(f"  {key}: {val:.3f}")

        if self.evidence:
            lines.extend(["", "EVIDENCE", "-" * 40])
            for e in self.evidence[:5]:
                lines.append(f"  [{e.source}] {e.status}: {e.score:.3f}")

        if self.recommendations:
            lines.extend(["", "RECOMMENDATIONS", "-" * 40])
            for r in self.recommendations:
                lines.append(f"  - {r}")

        lines.append("=" * 60)
        return "\n".join(lines)
