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

__all__ = [
    "TemporalConfig",
    "TemporalAnalyzer",
    "TemporalResult",
]
