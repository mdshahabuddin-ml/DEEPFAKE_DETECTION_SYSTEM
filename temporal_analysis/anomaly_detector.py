"""Anomaly detection for temporal analysis."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import TemporalConfig

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomalous: bool = False
    anomaly_score: float = 0.0
    confidence: float = 0.0
    anomaly_frames: List[int] = field(default_factory=list)
    anomaly_segments: List[Dict[str, Any]] = field(default_factory=list)
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """Detect anomalies in temporal sequences."""

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()

    def detect(
        self, features: np.ndarray, method: str = "statistical"
    ) -> AnomalyResult:
        """Detect anomalies in feature sequence.

        Args:
            features: Feature array (n_frames, n_features).
            method: Detection method ('statistical', 'threshold').

        Returns:
            AnomalyResult with detection results.
        """
        if features is None or len(features) < self.config.min_frames:
            return AnomalyResult(
                is_anomalous=False,
                anomaly_score=0.0,
                method="insufficient_data",
            )

        try:
            if method == "statistical":
                return self._detect_statistical(features)
            elif method == "threshold":
                return self._detect_threshold(features)
            else:
                return self._detect_statistical(features)
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return AnomalyResult(
                is_anomalous=False,
                anomaly_score=0.0,
                method="failed",
                metadata={"error": str(e)},
            )

    def _detect_statistical(self, features: np.ndarray) -> AnomalyResult:
        """Statistical anomaly detection."""
        # Calculate frame-to-frame differences
        diffs = np.diff(features, axis=0)
        frame_diffs = np.mean(np.abs(diffs), axis=1)

        # Calculate statistics
        mean_diff = np.mean(frame_diffs)
        std_diff = np.std(frame_diffs)

        # Detect anomalies using z-scores
        if std_diff > 0:
            z_scores = (frame_diffs - mean_diff) / std_diff
        else:
            z_scores = np.zeros_like(frame_diffs)

        # Find anomalous frames
        anomaly_threshold = 2.0
        anomaly_mask = z_scores > anomaly_threshold
        anomaly_frames = np.where(anomaly_mask)[0] + 1  # +1 because diff

        # Calculate anomaly score
        anomaly_ratio = len(anomaly_frames) / len(frame_diffs) if len(frame_diffs) > 0 else 0
        max_z = float(np.max(z_scores)) if len(z_scores) > 0 else 0

        anomaly_score = 0.5 * anomaly_ratio + 0.5 * min(max_z / 5.0, 1.0)
        is_anomalous = anomaly_score > self.config.anomaly_threshold

        return AnomalyResult(
            is_anomalous=is_anomalous,
            anomaly_score=float(anomaly_score),
            confidence=float(1.0 - std_diff / (mean_diff + 1e-10)),
            anomaly_frames=anomaly_frames.tolist(),
            method="statistical",
            metadata={
                "mean_diff": float(mean_diff),
                "std_diff": float(std_diff),
                "max_z_score": max_z,
                "n_anomalies": len(anomaly_frames),
            },
        )

    def _detect_threshold(self, features: np.ndarray) -> AnomalyResult:
        """Simple threshold-based detection."""
        diffs = np.diff(features, axis=0)
        frame_diffs = np.mean(np.abs(diffs), axis=1)

        # Use configured threshold
        threshold = self.config.anomaly_threshold
        anomaly_frames = np.where(frame_diffs > threshold)[0] + 1

        anomaly_ratio = len(anomaly_frames) / len(frame_diffs) if len(frame_diffs) > 0 else 0
        is_anomalous = anomaly_ratio > 0.1  # More than 10% anomalous

        return AnomalyResult(
            is_anomalous=is_anomalous,
            anomaly_score=float(anomaly_ratio),
            confidence=0.5,
            anomaly_frames=anomaly_frames.tolist(),
            method="threshold",
            metadata={"threshold": threshold, "n_anomalies": len(anomaly_frames)},
        )

    def detect_segments(
        self, features: np.ndarray, min_segment_length: int = 3
    ) -> List[Dict[str, Any]]:
        """Detect anomalous segments."""
        result = self.detect(features)

        segments = []
        if result.anomaly_frames:
            # Group consecutive frames
            current_segment = [result.anomaly_frames[0]]

            for i in range(1, len(result.anomaly_frames)):
                if result.anomaly_frames[i] == current_segment[-1] + 1:
                    current_segment.append(result.anomaly_frames[i])
                else:
                    if len(current_segment) >= min_segment_length:
                        segments.append({
                            "start_frame": current_segment[0],
                            "end_frame": current_segment[-1],
                            "length": len(current_segment),
                        })
                    current_segment = [result.anomaly_frames[i]]

            # Don't forget last segment
            if len(current_segment) >= min_segment_length:
                segments.append({
                    "start_frame": current_segment[0],
                    "end_frame": current_segment[-1],
                    "length": len(current_segment),
                })

        return segments
