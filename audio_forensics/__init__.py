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

_LAZY_IMPORTS = {
    "AudioForensicsConfig": ("audio_forensics.config", "AudioForensicsConfig"),
    "AudioAnalysisResult": ("audio_forensics.schemas", "AudioAnalysisResult"),
    "SegmentResult": ("audio_forensics.schemas", "SegmentResult"),
}

# Expose public API
__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    """PEP 562 lazy attribute access - avoids heavy imports at package import."""
    import importlib

    if name in _LAZY_IMPORTS:
        module_name, attr = _LAZY_IMPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
