"""Evidence collection from multiple sources."""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .config import MultimodalConfig
from .schemas import Evidence

logger = logging.getLogger(__name__)


class EvidenceCollector:
    """Collect and organize evidence from different analysis sources."""

    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()

    def collect_video_evidence(
        self, temporal_score: float = 0.0, metadata: Dict[str, Any] = None
    ) -> Evidence:
        """Collect video-related evidence."""
        metadata = metadata or {}

        # Determine status
        if temporal_score >= self.config.suspicious_threshold:
            status = "SUSPICIOUS"
        elif temporal_score >= 0.3:
            status = "LOW_SUSPICION"
        else:
            status = "NORMAL"

        return Evidence(
            source="video_analysis",
            score=temporal_score,
            confidence=abs(temporal_score - 0.5) * 2,
            status=status,
            details=metadata,
        )

    def collect_audio_evidence(
        self, manipulation_score: float = 0.0, metadata: Dict[str, Any] = None
    ) -> Evidence:
        """Collect audio-related evidence."""
        metadata = metadata or {}

        if manipulation_score >= self.config.suspicious_threshold:
            status = "SUSPICIOUS"
        elif manipulation_score >= 0.3:
            status = "LOW_SUSPICION"
        else:
            status = "NORMAL"

        return Evidence(
            source="audio_analysis",
            score=manipulation_score,
            confidence=abs(manipulation_score - 0.5) * 2,
            status=status,
            details=metadata,
        )

    def collect_sync_evidence(
        self, sync_score: float = 0.5, metadata: Dict[str, Any] = None
    ) -> Evidence:
        """Collect synchronization evidence."""
        metadata = metadata or {}

        # Low sync score = suspicious
        anomaly_score = 1.0 - sync_score

        if anomaly_score >= self.config.suspicious_threshold:
            status = "POTENTIAL_MISMATCH"
        elif anomaly_score >= 0.3:
            status = "LOW_SUSPICION"
        else:
            status = "CONSISTENT"

        return Evidence(
            source="sync_analysis",
            score=anomaly_score,
            confidence=abs(anomaly_score - 0.5) * 2,
            status=status,
            details=metadata,
        )

    def collect_metadata_evidence(
        self, findings: Dict[str, Any] = None
    ) -> Evidence:
        """Collect metadata-related evidence."""
        findings = findings or {}

        # Simple scoring based on findings
        suspicious_count = sum(
            1 for v in findings.values()
            if isinstance(v, str) and "suspicious" in v.lower()
        )

        score = min(1.0, suspicious_count * 0.2)

        return Evidence(
            source="metadata_analysis",
            score=score,
            confidence=0.5,
            status="SUSPICIOUS" if score > 0.3 else "NORMAL",
            details=findings,
        )

    def collect_all(
        self,
        video_score: float = 0.0,
        audio_score: float = 0.0,
        sync_score: float = 0.5,
        metadata: Dict[str, Any] = None,
    ) -> List[Evidence]:
        """Collect all evidence."""
        evidence = [
            self.collect_video_evidence(video_score),
            self.collect_audio_evidence(audio_score),
            self.collect_sync_evidence(sync_score),
        ]

        if metadata:
            evidence.append(self.collect_metadata_evidence(metadata))

        return evidence
