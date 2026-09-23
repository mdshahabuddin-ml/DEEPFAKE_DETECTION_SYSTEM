"""Data schemas for Audio Forensics module.

These schemas define the structure of analysis results and are independent
from existing prediction schemas in the system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AudioStatus(Enum):
    """Status categories for audio analysis."""
    NORMAL = "NORMAL"
    LOW_SUSPICION = "LOW_SUSPICION"
    SUSPICIOUS = "SUSPICIOUS"
    POTENTIAL_MANIPULATION = "POTENTIAL_MANIPULATION"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    UNKNOWN = "UNKNOWN"


class SpeechType(Enum):
    """Type of speech detected in segment."""
    SPEECH = "speech"
    SILENCE = "silence"
    NOISE = "noise"
    MUSIC = "music"
    UNKNOWN = "unknown"


@dataclass
class AudioProperties:
    """Properties of the input audio file."""
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    bit_depth: Optional[int] = None
    format: str = ""
    file_size_bytes: int = 0
    rms_level: float = 0.0
    peak_level: float = 0.0
    snr_estimate: float = 0.0
    has_clipping: bool = False
    has_silence: bool = False


@dataclass
class SegmentResult:
    """Result for a single audio segment."""
    segment_id: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    speech_type: SpeechType = SpeechType.UNKNOWN
    prediction: AudioStatus = AudioStatus.UNKNOWN
    probability: float = 0.0
    confidence: float = 0.0
    features_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeakerSimilarity:
    """Speaker verification result."""
    speaker_id: Optional[str] = None
    similarity_score: float = 0.0
    is_consistent: bool = False
    embedding_distance: float = 0.0
    verification_method: str = "unknown"
    notes: str = ""


@dataclass
class AudioAnalysisResult:
    """Complete result of audio analysis."""
    # File identification
    file_path: str = ""
    file_hash: str = ""
    analysis_id: str = ""

    # Audio properties
    properties: AudioProperties = field(default_factory=AudioProperties)

    # Overall assessment
    overall_prediction: AudioStatus = AudioStatus.UNKNOWN
    overall_probability: float = 0.0
    overall_confidence: float = 0.0

    # Segment results
    segments: List[SegmentResult] = field(default_factory=list)
    total_segments: int = 0
    suspicious_segments: int = 0

    # Speaker verification
    speaker_similarity: Optional[SpeakerSimilarity] = None

    # Timeline
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    # Model information
    model_name: str = ""
    model_version: str = ""
    processing_time: float = 0.0

    # Evidence
    evidence: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "analysis_id": self.analysis_id,
            "properties": {
                "duration": self.properties.duration,
                "sample_rate": self.properties.sample_rate,
                "channels": self.properties.channels,
                "format": self.properties.format,
                "rms_level": self.properties.rms_level,
                "peak_level": self.properties.peak_level,
                "snr_estimate": self.properties.snr_estimate,
                "has_clipping": self.properties.has_clipping,
            },
            "overall_prediction": self.overall_prediction.value,
            "overall_probability": self.overall_probability,
            "overall_confidence": self.overall_confidence,
            "total_segments": self.total_segments,
            "suspicious_segments": self.suspicious_segments,
            "segments": [
                {
                    "segment_id": s.segment_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "prediction": s.prediction.value,
                    "probability": s.probability,
                    "confidence": s.confidence,
                }
                for s in self.segments
            ],
            "speaker_similarity": {
                "similarity_score": self.speaker_similarity.similarity_score,
                "is_consistent": self.speaker_similarity.is_consistent,
                "verification_method": self.speaker_similarity.verification_method,
            } if self.speaker_similarity else None,
            "timeline": self.timeline,
            "model_name": self.model_name,
            "processing_time": self.processing_time,
            "evidence": self.evidence,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Audio Analysis Summary",
            f"{'=' * 50}",
            f"File: {self.file_path}",
            f"Duration: {self.properties.duration:.1f}s",
            f"Prediction: {self.overall_prediction.value}",
            f"Probability: {self.overall_probability:.3f}",
            f"Confidence: {self.overall_confidence:.3f}",
            f"Segments: {self.total_segments} total, {self.suspicious_segments} suspicious",
        ]
        if self.speaker_similarity:
            lines.append(
                f"Speaker Similarity: {self.speaker_similarity.similarity_score:.1%}"
            )
        if self.evidence:
            lines.append("Evidence:")
            for e in self.evidence[:5]:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings[:3]:
                lines.append(f"  - {w}")
        return "\n".join(lines)
