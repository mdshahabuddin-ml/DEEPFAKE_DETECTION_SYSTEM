"""Sequence analysis for temporal patterns."""

import logging
from typing import List, Optional

import numpy as np

from .config import TemporalConfig

logger = logging.getLogger(__name__)


class SequenceAnalyzer:
    """Analyze temporal sequences for anomalies."""

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()

    def analyze_sequence(
        self, features: np.ndarray
    ) -> dict:
        """Analyze feature sequence for temporal patterns.

        Args:
            features: Feature array (n_frames, n_features).

        Returns:
            Analysis results dictionary.
        """
        if len(features) < self.config.min_frames:
            return {
                "status": "insufficient_frames",
                "n_frames": len(features),
                "consistency_score": 0.0,
                "anomaly_segments": [],
            }

        # Calculate temporal consistency
        consistency = self._calculate_consistency(features)

        # Detect anomaly segments
        anomaly_segments = self._detect_anomalies(features)

        # Calculate overall score
        anomaly_score = self._calculate_anomaly_score(
            consistency, anomaly_segments, len(features)
        )

        return {
            "status": "analyzed",
            "n_frames": len(features),
            "consistency_score": consistency,
            "anomaly_segments": anomaly_segments,
            "anomaly_score": anomaly_score,
            "mean_frame_diff": float(np.mean(np.abs(np.diff(features, axis=0)))),
            "max_frame_diff": float(np.max(np.abs(np.diff(features, axis=0)))),
        }

    def _calculate_consistency(self, features: np.ndarray) -> float:
        """Calculate temporal consistency score."""
        if len(features) < 2:
            return 1.0

        # Calculate frame-to-frame differences
        diffs = np.diff(features, axis=0)

        # Consistency is inverse of variance in differences
        mean_diff = np.mean(np.abs(diffs))
        std_diff = np.std(diffs)

        # High consistency = low variance in changes
        if mean_diff == 0:
            return 1.0

        consistency = 1.0 / (1.0 + std_diff / mean_diff)
        return float(np.clip(consistency, 0, 1))

    def _detect_anomalies(self, features: np.ndarray) -> List[dict]:
        """Detect anomalous segments in the sequence."""
        anomalies = []

        if len(features) < self.config.window_size:
            return anomalies

        # Calculate local statistics
        for i in range(0, len(features) - self.config.window_size + 1,
                       self.config.stride):
            window = features[i:i + self.config.window_size]

            # Check for sudden changes
            window_diffs = np.abs(np.diff(window, axis=0))
            local_mean = np.mean(window_diffs)
            local_std = np.std(window_diffs)

            # Global statistics
            all_diffs = np.abs(np.diff(features, axis=0))
            global_mean = np.mean(all_diffs)
            global_std = np.std(all_diffs)

            # Detect anomaly: local deviation from global
            if global_std > 0:
                z_score = (local_mean - global_mean) / global_std
                if z_score > 2.0:  # Significant deviation
                    anomalies.append({
                        "start_frame": i,
                        "end_frame": i + self.config.window_size,
                        "z_score": float(z_score),
                        "local_mean": float(local_mean),
                        "global_mean": float(global_mean),
                    })

        return anomalies

    def _calculate_anomaly_score(
        self,
        consistency: float,
        anomaly_segments: list,
        n_frames: int,
    ) -> float:
        """Calculate overall anomaly score."""
        # Base score from consistency
        consistency_score = 1.0 - consistency

        # Score from anomaly segments
        if n_frames > 0:
            anomaly_ratio = sum(
                seg["end_frame"] - seg["start_frame"]
                for seg in anomaly_segments
            ) / n_frames
        else:
            anomaly_ratio = 0.0

        # Combined score
        anomaly_score = 0.6 * consistency_score + 0.4 * anomaly_ratio
        return float(np.clip(anomaly_score, 0, 1))

    def detect_sudden_changes(
        self, features: np.ndarray, threshold: float = 2.0
    ) -> List[int]:
        """Detect frames with sudden changes."""
        if len(features) < 2:
            return []

        diffs = np.mean(np.abs(np.diff(features, axis=0)), axis=1)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        if std_diff == 0:
            return []

        # Find frames with sudden changes
        z_scores = (diffs - mean_diff) / std_diff
        sudden_frames = np.where(z_scores > threshold)[0] + 1

        return sudden_frames.tolist()
