"""Fusion engine for combining multiple evidence sources."""
import logging
from typing import Dict, List, Optional

import numpy as np

from .config import MultimodalConfig
from .schemas import Evidence, FusionVerdict

logger = logging.getLogger(__name__)


class FusionEngine:
    """Fuse evidence from multiple sources into a verdict."""

    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()

    def fuse(self, evidence_list: List[Evidence]) -> Dict[str, any]:
        """Fuse multiple evidence pieces into a verdict.

        Args:
            evidence_list: List of Evidence objects.

        Returns:
            Dictionary with verdict and confidence.
        """
        if not evidence_list:
            return {
                "verdict": FusionVerdict.UNKNOWN,
                "confidence": 0.0,
                "weighted_score": 0.0,
            }

        # Weighted average fusion
        weights = self._get_weights(evidence_list)
        scores = [e.score for e in evidence_list]

        weighted_score = np.average(scores, weights=weights)

        # Calculate confidence based on agreement
        if len(scores) > 1:
            agreement = 1.0 - np.std(scores)
        else:
            agreement = 0.5

        confidence = float(agreement * weighted_score)

        # Determine verdict
        verdict = self._score_to_verdict(weighted_score)

        return {
            "verdict": verdict,
            "confidence": confidence,
            "weighted_score": float(weighted_score),
            "agreement": float(agreement),
        }

    def _get_weights(self, evidence_list: List[Evidence]) -> List[float]:
        """Get weights for each evidence source."""
        source_weights = {
            "video_analysis": self.config.video_weight,
            "audio_analysis": self.config.audio_weight,
            "sync_analysis": self.config.sync_weight,
            "temporal_analysis": self.config.temporal_weight,
            "metadata_analysis": 0.1,
        }

        weights = []
        for evidence in evidence_list:
            weight = source_weights.get(evidence.source, 0.1)
            weights.append(weight)

        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(evidence_list)] * len(evidence_list)

        return weights

    def _score_to_verdict(self, score: float) -> FusionVerdict:
        """Convert score to verdict."""
        if score >= self.config.high_confidence_threshold:
            return FusionVerdict.POTENTIALLY_MANIPULATED
        elif score >= self.config.suspicious_threshold:
            return FusionVerdict.SUSPICIOUS
        elif score >= 0.3:
            return FusionVerdict.POTENTIALLY_AUTHENTIC
        else:
            return FusionVerdict.AUTHENTIC

    def bayesian_fusion(
        self, evidence_list: List[Evidence]
    ) -> Dict[str, any]:
        """Bayesian fusion of evidence.

        Simple Bayesian approach: combine likelihood ratios.
        """
        if not evidence_list:
            return {
                "verdict": FusionVerdict.UNKNOWN,
                "confidence": 0.0,
                "posterior": 0.5,
            }

        # Prior: 50% authentic, 50% fake
        log_prior_ratio = 0.0  # log(1) = 0

        # Accumulate log likelihood ratios
        log_lr = log_prior_ratio
        for evidence in evidence_list:
            # Convert score to likelihood ratio
            score = evidence.score
            confidence = evidence.confidence

            # LR = P(evidence|fake) / P(evidence|authentic)
            lr = (score + 0.01) / (1.01 - score)
            log_lr += np.log(lr) * confidence

        # Posterior probability
        posterior = 1.0 / (1.0 + np.exp(-log_lr))

        # Verdict
        if posterior >= 0.8:
            verdict = FusionVerdict.POTENTIALLY_MANIPULATED
        elif posterior >= 0.6:
            verdict = FusionVerdict.SUSPICIOUS
        elif posterior >= 0.4:
            verdict = FusionVerdict.POTENTIALLY_AUTHENTIC
        else:
            verdict = FusionVerdict.AUTHENTIC

        return {
            "verdict": verdict,
            "confidence": abs(posterior - 0.5) * 2,
            "posterior": float(posterior),
        }

    def ds_fusion(
        self, evidence_list: List[Evidence]
    ) -> Dict[str, any]:
        """Dempster-Shafer inspired fusion."""
        if not evidence_list:
            return {
                "verdict": FusionVerdict.UNKNOWN,
                "confidence": 0.0,
                "belief_fake": 0.0,
                "belief_authentic": 0.0,
            }

        # Basic belief assignments
        belief_fake = 0.0
        belief_authentic = 0.0
        uncertainty = 1.0

        for evidence in evidence_list:
            score = evidence.score
            conf = evidence.confidence

            #分配基本概率
            m_fake = score * conf
            m_authentic = (1.0 - score) * conf
            m_uncertain = 1.0 - conf

            # Dempster's rule (simplified)
            K = belief_fake * m_authentic + belief_authentic * m_fake

            if (1 - K) > 0:
                new_fake = (belief_fake * m_fake + belief_fake * m_uncertain +
                           m_fake * m_uncertain) / (1 - K)
                new_authentic = (belief_authentic * m_authentic + belief_authentic * m_uncertain +
                                m_authentic * m_uncertain) / (1 - K)
            else:
                new_fake = belief_fake
                new_authentic = belief_authentic

            belief_fake = new_fake
            belief_authentic = new_authentic

        # Determine verdict
        if belief_fake > belief_authentic:
            score = belief_fake
        else:
            score = 1.0 - belief_authentic

        verdict = self._score_to_verdict(score)

        return {
            "verdict": verdict,
            "confidence": abs(belief_fake - belief_authentic),
            "belief_fake": float(belief_fake),
            "belief_authentic": float(belief_authentic),
        }
