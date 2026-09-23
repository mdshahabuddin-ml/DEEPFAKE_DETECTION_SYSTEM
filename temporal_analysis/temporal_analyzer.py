"""Main temporal analyzer orchestrator."""

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from .anomaly_detector import AnomalyDetector, AnomalyResult
from .config import TemporalConfig
from .feature_extractor import TemporalFeatureExtractor
from .frame_sampler import FrameSampler
from .sequence_analyzer import SequenceAnalyzer

logger = logging.getLogger(__name__)


class TemporalAnalyzerError(Exception):
    """Custom exception for temporal analysis errors."""
    pass


class TemporalResult:
    """Result of temporal analysis."""

    def __init__(
        self,
        video_path: str = "",
        analysis_id: str = "",
        n_frames: int = 0,
        consistency_score: float = 0.0,
        anomaly_score: float = 0.0,
        is_suspicious: bool = False,
        anomaly_segments: List[Dict[str, Any]] = None,
        evidence: List[str] = None,
        recommendations: List[str] = None,
    ):
        self.video_path = video_path
        self.analysis_id = analysis_id
        self.n_frames = n_frames
        self.consistency_score = consistency_score
        self.anomaly_score = anomaly_score
        self.is_suspicious = is_suspicious
        self.anomaly_segments = anomaly_segments or []
        self.evidence = evidence or []
        self.recommendations = recommendations or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "analysis_id": self.analysis_id,
            "n_frames": self.n_frames,
            "consistency_score": self.consistency_score,
            "anomaly_score": self.anomaly_score,
            "is_suspicious": self.is_suspicious,
            "n_anomaly_segments": len(self.anomaly_segments),
            "evidence": self.evidence,
            "recommendations": self.recommendations,
        }

    def get_summary(self) -> str:
        lines = [
            "=" * 60,
            "TEMPORAL ANALYSIS REPORT",
            "=" * 60,
            f"Video: {self.video_path}",
            f"Analysis ID: {self.analysis_id}",
            f"Frames Analyzed: {self.n_frames}",
            "",
            "RESULTS",
            "-" * 40,
            f"Consistency Score: {self.consistency_score:.3f}",
            f"Anomaly Score: {self.anomaly_score:.3f}",
            f"Suspicious: {'Yes' if self.is_suspicious else 'No'}",
            f"Anomaly Segments: {len(self.anomaly_segments)}",
        ]

        if self.evidence:
            lines.extend(["", "EVIDENCE", "-" * 40])
            for e in self.evidence:
                lines.append(f"  - {e}")

        if self.recommendations:
            lines.extend(["", "RECOMMENDATIONS", "-" * 40])
            for r in self.recommendations:
                lines.append(f"  - {r}")

        lines.append("=" * 60)
        return "\n".join(lines)


class TemporalAnalyzer:
    """Main temporal analysis orchestrator.

    Usage:
        config = TemporalConfig()
        analyzer = TemporalAnalyzer(config)
        result = analyzer.analyze("video.mp4")
    """

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        self._sampler = FrameSampler(self.config)
        self._feature_extractor = TemporalFeatureExtractor(self.config)
        self._sequence_analyzer = SequenceAnalyzer(self.config)
        self._anomaly_detector = AnomalyDetector(self.config)

    def analyze(
        self, video_path: str, method: str = "uniform"
    ) -> TemporalResult:
        """Analyze video for temporal anomalies.

        Args:
            video_path: Path to video file.
            method: Frame sampling method.

        Returns:
            TemporalResult with analysis results.
        """
        analysis_id = f"TEMP-{uuid.uuid4().hex[:8].upper()}"

        try:
            # Sample frames
            frames = self._sampler.sample_frames(video_path, method)
            if not frames:
                return TemporalResult(
                    video_path=video_path,
                    analysis_id=analysis_id,
                    evidence=["No frames could be extracted"],
                )

            return self.analyze_frames(frames, video_path, analysis_id)

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return TemporalResult(
                video_path=video_path,
                analysis_id=analysis_id,
                evidence=[f"Analysis failed: {str(e)}"],
            )

    def analyze_frames(
        self,
        frames: List[np.ndarray],
        video_path: str = "",
        analysis_id: str = "",
    ) -> TemporalResult:
        """Analyze pre-extracted frames.

        Args:
            frames: List of video frames.
            video_path: Original video path.
            analysis_id: Analysis identifier.

        Returns:
            TemporalResult with analysis results.
        """
        if not analysis_id:
            analysis_id = f"TEMP-{uuid.uuid4().hex[:8].upper()}"

        # Extract features
        features = self._feature_extractor.extract_features(frames)
        if len(features) == 0:
            return TemporalResult(
                video_path=video_path,
                analysis_id=analysis_id,
                evidence=["Could not extract features from frames"],
            )

        # Analyze sequence
        seq_result = self._sequence_analyzer.analyze_sequence(features)

        # Detect anomalies
        anomaly_result = self._anomaly_detector.detect(features)

        # Build result
        consistency = seq_result.get("consistency_score", 0.0)
        anomaly_score = anomaly_result.anomaly_score
        is_suspicious = anomaly_result.is_anomalous

        # Build evidence
        evidence = []
        if is_suspicious:
            evidence.append(
                f"Temporal anomaly detected (score: {anomaly_score:.3f})"
            )
        if consistency < 0.5:
            evidence.append(
                f"Low temporal consistency ({consistency:.3f})"
            )
        if len(anomaly_result.anomaly_frames) > 0:
            evidence.append(
                f"{len(anomaly_result.anomaly_frames)} anomalous frames detected"
            )

        # Build recommendations
        recommendations = []
        if is_suspicious:
            recommendations.append("Manual review recommended")
            recommendations.append("Consider additional analysis")

        return TemporalResult(
            video_path=video_path,
            analysis_id=analysis_id,
            n_frames=len(frames),
            consistency_score=consistency,
            anomaly_score=anomaly_score,
            is_suspicious=is_suspicious,
            anomaly_segments=seq_result.get("anomaly_segments", []),
            evidence=evidence,
            recommendations=recommendations,
        )

    def analyze_array(
        self, video_frames: np.ndarray, video_path: str = ""
    ) -> TemporalResult:
        """Analyze video from numpy array.

        Args:
            video_frames: Video frames array (N, H, W, C).
            video_path: Original video path.

        Returns:
            TemporalResult with analysis results.
        """
        frames = self._sampler.sample_from_array(video_frames)
        return self.analyze_frames(frames, video_path)


def get_temporal_analyzer(
    config: Optional[TemporalConfig] = None,
) -> TemporalAnalyzer:
    """Factory function to get temporal analyzer."""
    return TemporalAnalyzer(config)
