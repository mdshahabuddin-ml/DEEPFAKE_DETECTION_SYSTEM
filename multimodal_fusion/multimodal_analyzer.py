"""Main multimodal analyzer orchestrator."""
import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from .config import MultimodalConfig
from .consistency_checker import ConsistencyChecker
from .evidence_collector import EvidenceCollector
from .fusion_engine import FusionEngine
from .schemas import (
    ConsistencyResult,
    Evidence,
    FusionVerdict,
    MultimodalResult,
)

logger = logging.getLogger(__name__)


class MultimodalAnalyzerError(Exception):
    """Custom exception for multimodal analysis errors."""
    pass


class MultimodalAnalyzer:
    """Main multimodal analysis orchestrator.

    Usage:
        config = MultimodalConfig()
        analyzer = MultimodalAnalyzer(config)
        result = analyzer.analyze(
            video_score=0.3,
            audio_score=0.7,
            sync_score=0.5,
        )
    """

    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()
        self._checker = ConsistencyChecker(self.config)
        self._collector = EvidenceCollector(self.config)
        self._fusion = FusionEngine(self.config)

    def analyze(
        self,
        video_score: float = 0.0,
        audio_score: float = 0.0,
        sync_score: float = 0.5,
        temporal_score: float = 0.0,
        metadata: Dict[str, Any] = None,
        method: str = "weighted",
    ) -> MultimodalResult:
        """Perform multimodal analysis.

        Args:
            video_score: Video manipulation score.
            audio_score: Audio manipulation score.
            sync_score: Audio-video sync score (1=good, 0=bad).
            temporal_score: Temporal consistency score.
            metadata: Additional metadata findings.
            method: Fusion method ('weighted', 'bayesian', 'ds').

        Returns:
            MultimodalResult with analysis results.
        """
        try:
            # Collect evidence
            evidence_list = self._collector.collect_all(
                video_score=video_score,
                audio_score=audio_score,
                sync_score=sync_score,
                metadata=metadata,
            )

            # Add temporal evidence if provided
            if temporal_score > 0:
                evidence_list.append(
                    Evidence(
                        source="temporal_analysis",
                        score=temporal_score,
                        confidence=abs(temporal_score - 0.5) * 2,
                        status="SUSPICIOUS" if temporal_score > 0.5 else "NORMAL",
                    )
                )

            # Fuse evidence
            if method == "bayesian":
                fusion_result = self._fusion.bayesian_fusion(evidence_list)
            elif method == "ds":
                fusion_result = self._fusion.ds_fusion(evidence_list)
            else:
                fusion_result = self._fusion.fuse(evidence_list)

            # Build consistency result
            consistency = ConsistencyResult(
                sync_score=sync_score,
                status=self._sync_to_status(sync_score),
            )

            # Build scores dict
            scores = {
                "video": video_score,
                "audio": audio_score,
                "sync": sync_score,
                "temporal": temporal_score,
                "fused": fusion_result.get("weighted_score", fusion_result.get("posterior", 0.0)),
            }

            # Build recommendations
            recommendations = self._build_recommendations(
                fusion_result["verdict"], evidence_list
            )

            return MultimodalResult(
                verdict=fusion_result["verdict"],
                confidence=fusion_result["confidence"],
                scores=scores,
                evidence=evidence_list,
                consistency=consistency,
                recommendations=recommendations,
                metadata={"method": method},
            )

        except Exception as e:
            logger.error(f"Multimodal analysis failed: {e}")
            return MultimodalResult(
                verdict=FusionVerdict.REQUIRES_REVIEW,
                confidence=0.0,
                recommendations=["Analysis failed - manual review required"],
                metadata={"error": str(e)},
            )

    def analyze_with_features(
        self,
        audio_features: np.ndarray,
        video_features: np.ndarray,
        video_score: float = 0.0,
        audio_score: float = 0.0,
    ) -> MultimodalResult:
        """Analyze using feature arrays."""
        # Check consistency
        consistency = self._checker.check_sync(
            audio_features, video_features
        )

        return self.analyze(
            video_score=video_score,
            audio_score=audio_score,
            sync_score=consistency.sync_score,
        )

    def _sync_to_status(self, sync_score: float) -> str:
        """Convert sync score to status string."""
        if sync_score >= self.config.sync_threshold:
            return "CONSISTENT"
        elif sync_score >= self.config.lip_sync_threshold:
            return "POTENTIAL_MISMATCH"
        else:
            return "INCONSISTENT"

    def _build_recommendations(
        self, verdict: FusionVerdict, evidence: List[Evidence]
    ) -> List[str]:
        """Build recommendations based on verdict."""
        recs = []

        if verdict in [FusionVerdict.SUSPICIOUS, FusionVerdict.POTENTIALLY_MANIPULATED]:
            recs.append("Manual review recommended")
            recs.append("Consider additional forensic analysis")

        # Check specific evidence
        for e in evidence:
            if e.status in ["SUSPICIOUS", "POTENTIAL_MISMATCH"]:
                recs.append(f"Review {e.source} findings")

        if not recs:
            recs.append("No immediate action required")

        return list(set(recs))[:5]  # Dedupe and limit


def get_multimodal_analyzer(
    config: Optional[MultimodalConfig] = None,
) -> MultimodalAnalyzer:
    """Factory function to get multimodal analyzer."""
    return MultimodalAnalyzer(config)
