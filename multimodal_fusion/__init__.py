"""Multimodal Fusion Module - Audio-video consistency and evidence fusion.

This module provides multimodal analysis capabilities for detecting
inconsistencies between audio and video streams.

Architecture:
    Video Evidence ─┐
                    ├──→ Evidence Fusion ──→ Assessment
    Audio Evidence ─┘

Privacy:
    - Original media is never modified
    - Analysis is performed on extracted features only
"""
__version__ = "0.1.0"
__all__ = [
    "MultimodalConfig",
    "MultimodalAnalyzer",
    "MultimodalResult",
]
