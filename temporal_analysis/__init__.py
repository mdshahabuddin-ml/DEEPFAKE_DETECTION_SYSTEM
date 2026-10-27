"""Temporal Analysis Module - Video temporal consistency analysis.

This module provides temporal analysis capabilities for detecting
manipulation in video sequences by analyzing frame-to-frame consistency.

Architecture:
    Video -> Frame Sampling -> Feature Extraction ->
    Temporal Sequence Analysis -> Anomaly Detection -> Result

Privacy:
    - Original video is never modified
    - Temporary frames are created for processing
    - Frame data is not stored permanently
"""

__version__ = "0.1.0"

_LAZY_IMPORTS = {
    "TemporalConfig": ("temporal_analysis.config", "TemporalConfig"),
    "TemporalAnalyzer": ("temporal_analysis.temporal_analyzer", "TemporalAnalyzer"),
    "TemporalResult": ("temporal_analysis.temporal_analyzer", "TemporalResult"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    """PEP 562 lazy attribute access."""
    import importlib

    if name in _LAZY_IMPORTS:
        module_name, attr = _LAZY_IMPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
