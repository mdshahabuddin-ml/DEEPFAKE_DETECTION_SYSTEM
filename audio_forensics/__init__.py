"""Audio Forensics Module - Isolated audio deepfake detection pipeline.

This module provides audio manipulation detection capabilities that are
completely independent from the existing visual detection system.

Architecture:
    Audio Input -> Preprocessing -> Feature Extraction -> Segmentation ->
    Detection -> Timeline -> Forensic Assessment

Privacy:
    - Original audio is never modified
    - Temporary working copies are used for processing
    - Evidence retention only when explicitly requested

Limitations:
    - Cannot guarantee 100% detection accuracy
    - Performance depends on audio quality and recording conditions
    - Domain shift between training and deployment data may affect results
    - Compression and re-recording can degrade detection performance
"""

__version__ = "0.1.0"
__author__ = "Deepfake Detection System"

# Lazy imports to avoid loading heavy dependencies at package import time
def _lazy_import():
    """Lazily import core components to reduce startup time."""
    from .config import AudioForensicsConfig
    from .schemas import AudioAnalysisResult, SegmentResult
    return {
        "AudioForensicsConfig": AudioForensicsConfig,
        "AudioAnalysisResult": AudioAnalysisResult,
        "SegmentResult": SegmentResult,
    }

# Expose public API
__all__ = [
    "AudioForensicsConfig",
    "AudioAnalysisResult",
    "SegmentResult",
]
