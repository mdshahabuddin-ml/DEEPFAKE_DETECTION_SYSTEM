"""Main audio forensics orchestrator.

This module provides the top-level interface for audio forensic analysis.
It orchestrates all components: preprocessing, segmentation, feature extraction,
detection, and reporting.

Usage:
    from audio_forensics import AudioForensics

    forensics = AudioForensics()
    result = forensics.analyze("path/to/audio.wav")
    result.to_dict()
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .config import AudioForensicsConfig
from .features import AudioFeatureExtractor, get_feature_extractor
from .inference import AudioInferencePipeline, get_pipeline
from .schemas import (
    AudioAnalysisResult,
    AudioStatus,
    SegmentResult,
    SpeakerSimilarity,
)
from .segmentation import AudioSegmenter, get_segmenter
from .timeline import AudioForensicTimeline, get_timeline

logger = logging.getLogger(__name__)


class AudioForensicsError(Exception):
    """Custom exception for audio forensics errors."""
    pass


class AudioForensics:
    """Main interface for audio forensic analysis.

    This class provides a high-level API for analyzing audio files
    for potential manipulation. It orchestrates all analysis components.

    Usage:
        # Basic usage
        forensics = AudioForensics()
        result = forensics.analyze("audio.wav")

        # With custom config
        config = AudioForensicsConfig(fake_threshold=0.6)
        forensics = AudioForensics(config)
        result = forensics.analyze("audio.wav")

        # Access results
        print(result.overall_prediction.value)
        print(result.overall_probability)
        print(result.get_summary())
    """

    def __init__(self, config: Optional[AudioForensicsConfig] = None):
        """Initialize audio forensics system.

        Args:
            config: Configuration. Uses defaults if None.
        """
        self.config = config or AudioForensicsConfig()

        # Initialize components
        self._pipeline = get_pipeline(self.config)
        self._timeline = get_timeline(self.config)
        self._segmenter = get_segmenter(self.config)
        self._feature_extractor = get_feature_extractor(self.config)

        logger.info("Audio forensics system initialized")

    def analyze(
        self,
        audio_input: Union[str, np.ndarray],
        sample_rate: int = 16000,
        case_id: Optional[str] = None,
        analyst_notes: Optional[str] = None,
    ) -> AudioForensicsResult:
        """Analyze audio for potential manipulation.

        Args:
            audio_input: Path to audio file or numpy array.
            sample_rate: Sample rate (required if input is array).
            case_id: Optional case identifier.
            analyst_notes: Optional notes from analyst.

        Returns:
            AudioForensicsResult with complete analysis.
        """
        # Create case ID if not provided
        if case_id is None:
            case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"

        logger.info(f"Starting analysis for case {case_id}")

        try:
            if isinstance(audio_input, (str, Path)):
                # File path
                audio_path = str(audio_input)
                result = self._pipeline.analyze(audio_path)
            else:
                # numpy array
                result = self._pipeline.analyze_audio_array(
                    audio_input, sample_rate
                )

            # Create timeline
            timeline = self._timeline.create_timeline(result)

            # Create forensics result
            forensics_result = AudioForensicsResult(
                case_id=case_id,
                analysis_result=result,
                timeline=timeline,
                analyst_notes=analyst_notes or "",
            )

            logger.info(
                f"Analysis complete for {case_id}: "
                f"{result.overall_prediction.value}"
            )

            return forensics_result

        except Exception as e:
            logger.error(f"Analysis failed for {case_id}: {e}")
            raise AudioForensicsError(f"Analysis failed: {e}") from e

    def analyze_segments(
        self,
        audio_input: Union[str, np.ndarray],
        sample_rate: int = 16000,
    ) -> List[Dict[str, Any]]:
        """Analyze audio and return detailed segment information.

        Args:
            audio_input: Path to audio file or numpy array.
            sample_rate: Sample rate.

        Returns:
            List of segment analysis dictionaries.
        """
        if isinstance(audio_input, (str, Path)):
            result = self._pipeline.analyze(str(audio_input))
        else:
            result = self._pipeline.analyze_audio_array(
                audio_input, sample_rate
            )

        segments = []
        for segment in result.segments:
            segments.append({
                "id": segment.segment_id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration,
                "prediction": segment.prediction.value,
                "probability": segment.probability,
                "confidence": segment.confidence,
                "speech_type": segment.speech_type.value,
            })

        return segments

    def get_timeline(
        self,
        audio_input: Union[str, np.ndarray],
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Get visual timeline representation.

        Args:
            audio_input: Path to audio file or numpy array.
            sample_rate: Sample rate.

        Returns:
            Timeline data for visualization.
        """
        if isinstance(audio_input, (str, Path)):
            result = self._pipeline.analyze(str(audio_input))
        else:
            result = self._pipeline.analyze_audio_array(
                audio_input, sample_rate
            )

        timeline = self._timeline.create_timeline(result)
        return timeline.to_dict()

    def compare_segments(
        self,
        audio_input: Union[str, np.ndarray],
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Compare different segments of the audio.

        Args:
            audio_input: Path to audio file or numpy array.
            sample_rate: Sample rate.

        Returns:
            Comparison analysis.
        """
        if isinstance(audio_input, (str, Path)):
            result = self._pipeline.analyze(str(audio_input))
        else:
            result = self._pipeline.analyze_audio_array(
                audio_input, sample_rate
            )

        if not result.segments:
            return {"comparison": "No segments to compare"}

        # Compare segments
        probabilities = [s.probability for s in result.segments]
        confidences = [s.confidence for s in result.segments]

        # Find most and least suspicious
        max_prob_idx = np.argmax(probabilities)
        min_prob_idx = np.argmin(probabilities)

        return {
            "n_segments": len(result.segments),
            "mean_probability": float(np.mean(probabilities)),
            "std_probability": float(np.std(probabilities)),
            "most_suspicious": {
                "segment_id": result.segments[max_prob_idx].segment_id,
                "start_time": result.segments[max_prob_idx].start_time,
                "end_time": result.segments[max_prob_idx].end_time,
                "probability": result.segments[max_prob_idx].probability,
            },
            "least_suspicious": {
                "segment_id": result.segments[min_prob_idx].segment_id,
                "start_time": result.segments[min_prob_idx].start_time,
                "end_time": result.segments[min_prob_idx].end_time,
                "probability": result.segments[min_prob_idx].probability,
            },
            "consistency": float(1.0 - np.std(probabilities)),
        }

    def get_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "version": "0.1.0",
            "pipeline": self._pipeline.get_info(),
            "config": {
                "target_sample_rate": self.config.target_sample_rate,
                "fake_threshold": self.config.fake_threshold,
                "suspicious_threshold": self.config.suspicious_threshold,
                "supported_formats": list(self.config.supported_formats),
            },
        }


class AudioForensicsResult:
    """Complete result of audio forensic analysis."""

    def __init__(
        self,
        case_id: str,
        analysis_result: AudioAnalysisResult,
        timeline: AudioForensicTimeline,
        analyst_notes: str = "",
    ):
        """Initialize forensics result.

        Args:
            case_id: Case identifier.
            analysis_result: Analysis result from pipeline.
            timeline: Forensic timeline.
            analyst_notes: Optional analyst notes.
        """
        self.case_id = case_id
        self.analysis = analysis_result
        self.timeline = timeline
        self.analyst_notes = analyst_notes

    @property
    def prediction(self) -> AudioStatus:
        """Get overall prediction."""
        return self.analysis.overall_prediction

    @property
    def probability(self) -> float:
        """Get overall probability."""
        return self.analysis.overall_probability

    @property
    def confidence(self) -> float:
        """Get overall confidence."""
        return self.analysis.overall_confidence

    @property
    def file_hash(self) -> str:
        """Get file hash."""
        return self.analysis.file_hash

    @property
    def duration(self) -> float:
        """Get audio duration."""
        return self.analysis.properties.duration

    @property
    def segments(self) -> List[SegmentResult]:
        """Get segment results."""
        return self.analysis.segments

    @property
    def suspicious_segments(self) -> int:
        """Get count of suspicious segments."""
        return self.analysis.suspicious_segments

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 60,
            "AUDIO FORENSIC ANALYSIS REPORT",
            "=" * 60,
            f"Case ID: {self.case_id}",
            f"File: {self.analysis.file_path}",
            f"Hash: {self.analysis.file_hash[:32]}...",
            f"Duration: {self.duration:.1f}s",
            "",
            "OVERALL ASSESSMENT",
            "-" * 40,
            f"Prediction: {self.prediction.value}",
            f"Probability: {self.probability:.3f}",
            f"Confidence: {self.confidence:.3f}",
            "",
            "SEGMENT ANALYSIS",
            "-" * 40,
            f"Total Segments: {self.analysis.total_segments}",
            f"Suspicious Segments: {self.suspicious_segments}",
            "",
        ]

        # Add timeline summary
        if self.timeline and self.timeline.segments:
            lines.append("TIMELINE")
            lines.append("-" * 40)
            for seg in self.timeline.segments[:5]:  # Show first 5
                lines.append(
                    f"  {seg['start_time']:.1f}s - {seg['end_time']:.1f}s: "
                    f"{seg['status']}"
                )
            if len(self.timeline.segments) > 5:
                lines.append(f"  ... and {len(self.timeline.segments) - 5} more")
            lines.append("")

        # Add evidence
        if self.analysis.evidence:
            lines.append("EVIDENCE")
            lines.append("-" * 40)
            for e in self.analysis.evidence:
                lines.append(f"  - {e}")
            lines.append("")

        # Add warnings
        if self.analysis.warnings:
            lines.append("WARNINGS")
            lines.append("-" * 40)
            for w in self.analysis.warnings:
                lines.append(f"  - {w}")
            lines.append("")

        # Add recommendations
        if self.analysis.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 40)
            for r in self.analysis.recommendations:
                lines.append(f"  - {r}")
            lines.append("")

        # Add analyst notes
        if self.analyst_notes:
            lines.append("ANALYST NOTES")
            lines.append("-" * 40)
            lines.append(f"  {self.analyst_notes}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case_id": self.case_id,
            "analysis": self.analysis.to_dict(),
            "timeline": self.timeline.to_dict() if self.timeline else None,
            "analyst_notes": self.analyst_notes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, default=str)


def get_audio_forensics(
    config: Optional[AudioForensicsConfig] = None,
) -> AudioForensics:
    """Factory function to get audio forensics system."""
    return AudioForensics(config)
