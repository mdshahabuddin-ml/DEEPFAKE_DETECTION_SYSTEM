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

_LAZY_IMPORTS = {
    "MultimodalConfig": ("multimodal_fusion.config", "MultimodalConfig"),
    "MultimodalAnalyzer": ("multimodal_fusion.multimodal_analyzer", "MultimodalAnalyzer"),
    "MultimodalResult": ("multimodal_fusion.schemas", "MultimodalResult"),
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
