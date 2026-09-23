"""
src/forensic_report_enhanced.py
Enhanced forensic report data structures for comprehensive deepfake analysis.

Defines all data classes for the multi-modal forensic report including:
- Case information and file integrity
- Video, audio, speaker, and lip-sync analysis
- Provenance and metadata verification
- Technical details and model versions
- Evidence timeline and limitations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class CaseInformation:
    """Case identification and file integrity data."""
    case_id: str
    file_hash_sha256: str
    file_hash_md5: str
    file_name: str
    file_size: int
    media_type: str  # "image" | "video" | "audio"
    analysis_timestamp: str


@dataclass
class VideoAnalysis:
    """Video-specific analysis results."""
    faces_detected: int
    face_consistency_score: float
    temporal_consistency_score: float
    visual_artifact_score: float
    manipulated_frames: List[int]
    frame_analysis: List[Dict[str, Any]]


@dataclass
class AudioAnalysis:
    """Audio-specific analysis results."""
    audio_deepfake_score: float
    quality_score: float
    snr_estimate: float
    spectral_centroid: float
    clipping_ratio: float
    audio_duration: float


@dataclass
class SpeakerSimilarity:
    """Speaker verification results."""
    claimed_speaker: str
    embedding_similarity: float
    is_verified: bool
    verification_confidence: float
    enrollment_samples: int


@dataclass
class LipSyncAnalysis:
    """Audio-visual synchronization analysis."""
    sync_score: float
    anomalies_count: int
    anomalies: List[Dict[str, Any]]
    consistency: float
    timeline: List[Dict[str, Any]]


@dataclass
class ProvenanceMetadata:
    """Provenance and metadata verification."""
    c2pa_status: str
    trust_level: str
    has_c2pa: bool
    metadata_consistent: bool
    editing_indicators: Dict[str, Any]


@dataclass
class TechnicalDetails:
    """Model versions and technical metadata."""
    visual_model: str
    audio_model: str
    sync_model: str
    confidence_interval: Optional[List[float]]
    uncertainty: float
    predictive_entropy: float


@dataclass
class EvidenceTimeline:
    """Timeline of analysis events."""
    events: List[Dict[str, Any]]
    temporal_summary: str


@dataclass
class Limitations:
    """Known limitations and recommendations."""
    known_limitations: List[str]
    confidence_factors: List[str]
    recommendations: List[str]


@dataclass
class EnhancedForensicReport:
    """Complete enhanced forensic report."""
    case_info: CaseInformation
    video_analysis: Optional[VideoAnalysis]
    audio_analysis: Optional[AudioAnalysis]
    speaker_similarity: Optional[SpeakerSimilarity]
    lip_sync_analysis: Optional[LipSyncAnalysis]
    provenance_metadata: ProvenanceMetadata
    technical_details: TechnicalDetails
    evidence_timeline: EvidenceTimeline
    limitations: Limitations
    overall_verdict: str
    overall_confidence: float
    trust_score: float
