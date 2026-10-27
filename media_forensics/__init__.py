"""Media Forensics Module - File integrity and metadata analysis.

This module provides forensic analysis of media files including
hashing, format detection, metadata extraction, and integrity checks.

Important:
    - Original files are NEVER modified
    - Hashes are calculated from original file bytes
    - Do not interpret missing provenance as proof of fake media
"""
__version__ = "0.1.0"

_LAZY_IMPORTS = {
    "MediaForensicsConfig": ("media_forensics.config", "MediaForensicsConfig"),
    "ForensicAnalyzer": ("media_forensics.forensic_analyzer", "ForensicAnalyzer"),
    "ForensicResult": ("media_forensics.forensic_analyzer", "ForensicResult"),
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
