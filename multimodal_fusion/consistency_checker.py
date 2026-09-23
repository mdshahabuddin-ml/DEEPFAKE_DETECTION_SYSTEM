"""Audio-video consistency checking."""
import logging
from typing import Dict, List, Optional

import numpy as np

from .config import MultimodalConfig
from .schemas import ConsistencyResult, ConsistencyStatus, Evidence

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """Check consistency between audio and video streams."""

    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()

    def check_sync(
        self,
        audio_features: np.ndarray,
        video_features: np.ndarray,
    ) -> ConsistencyResult:
        """Check audio-video synchronization.

        Args:
            audio_features: Audio feature sequence.
            video_features: Video feature sequence.

        Returns:
            ConsistencyResult with sync analysis.
        """
        if audio_features is None or video_features is None:
            return ConsistencyResult(
                sync_score=0.0,
                status=ConsistencyStatus.UNKNOWN,
            )

        # Align feature sequences
        min_len = min(len(audio_features), len(video_features))
        if min_len == 0:
            return ConsistencyResult(
                sync_score=0.0,
                status=ConsistencyStatus.UNKNOWN,
            )

        audio_feat = audio_features[:min_len]
        video_feat = video_features[:min_len]

        # Calculate cross-correlation
        sync_score = self._calculate_sync_score(audio_feat, video_feat)

        # Determine status
        if sync_score >= self.config.sync_threshold:
            status = ConsistencyStatus.CONSISTENT
        elif sync_score >= self.config.lip_sync_threshold:
            status = ConsistencyStatus.POTENTIAL_MISMATCH
        else:
            status = ConsistencyStatus.INCONSISTENT

        evidence = [
            Evidence(
                source="sync_checker",
                score=sync_score,
                confidence=abs(sync_score - 0.5) * 2,
                status=status.value,
            )
        ]

        return ConsistencyResult(
            sync_score=sync_score,
            status=status,
            evidence=evidence,
            details={"min_len": min_len},
        )

    def _calculate_sync_score(
        self, audio_feat: np.ndarray, video_feat: np.ndarray
    ) -> float:
        """Calculate synchronization score."""
        # Normalize features
        audio_norm = audio_feat / (np.linalg.norm(audio_feat, axis=1, keepdims=True) + 1e-10)
        video_norm = video_feat / (np.linalg.norm(video_feat, axis=1, keepdims=True) + 1e-10)

        # Calculate frame-wise correlation
        correlations = []
        for i in range(min(len(audio_norm), len(video_norm))):
            a = audio_norm[i]
            v = video_norm[i]
            min_dim = min(len(a), len(v))
            corr = np.corrcoef(a[:min_dim], v[:min_dim])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        if not correlations:
            return 0.5

        return float(np.mean(correlations))

    def check_lip_sync(
        self,
        audio_energy: np.ndarray,
        mouth_movement: np.ndarray,
    ) -> ConsistencyResult:
        """Check lip synchronization."""
        if audio_energy is None or mouth_movement is None:
            return ConsistencyResult(
                sync_score=0.5,
                status=ConsistencyStatus.UNKNOWN,
            )

        min_len = min(len(audio_energy), len(mouth_movement))
        if min_len == 0:
            return ConsistencyResult(
                sync_score=0.5,
                status=ConsistencyStatus.UNKNOWN,
            )

        audio = audio_energy[:min_len]
        mouth = mouth_movement[:min_len]

        # Normalize
        audio = audio / (np.max(audio) + 1e-10)
        mouth = mouth / (np.max(mouth) + 1e-10)

        # Calculate correlation
        correlation = float(np.corrcoef(audio, mouth)[0, 1])
        if np.isnan(correlation):
            correlation = 0.5

        # Determine status
        if correlation >= self.config.lip_sync_threshold:
            status = ConsistencyStatus.CONSISTENT
        elif correlation >= 0.3:
            status = ConsistencyStatus.POTENTIAL_MISMATCH
        else:
            status = ConsistencyStatus.INCONSISTENT

        return ConsistencyResult(
            sync_score=correlation,
            status=status,
            evidence=[
                Evidence(
                    source="lip_sync",
                    score=correlation,
                    confidence=abs(correlation - 0.5) * 2,
                    status=status.value,
                )
            ],
        )

    def check_temporal_alignment(
        self,
        audio_timestamps: np.ndarray,
        video_timestamps: np.ndarray,
    ) -> float:
        """Check temporal alignment between modalities."""
        if len(audio_timestamps) == 0 or len(video_timestamps) == 0:
            return 0.0

        # Calculate timing differences
        audio_diffs = np.diff(audio_timestamps)
        video_diffs = np.diff(video_timestamps)

        if len(audio_diffs) == 0 or len(video_diffs) == 0:
            return 0.0

        # Compare frame rates
        audio_fps = 1.0 / np.mean(audio_diffs) if np.mean(audio_diffs) > 0 else 0
        video_fps = 1.0 / np.mean(video_diffs) if np.mean(video_diffs) > 0 else 0

        if audio_fps == 0 or video_fps == 0:
            return 0.0

        # Similarity of frame rates
        fps_ratio = min(audio_fps, video_fps) / max(audio_fps, video_fps)
        return float(fps_ratio)
