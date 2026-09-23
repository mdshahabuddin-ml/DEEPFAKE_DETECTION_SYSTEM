"""Media Forensics Module - File integrity and metadata analysis.

This module provides forensic analysis of media files including
hashing, format detection, metadata extraction, and integrity checks.

Important:
    - Original files are NEVER modified
    - Hashes are calculated from original file bytes
    - Do not interpret missing provenance as proof of fake media
"""
__version__ = "0.1.0"
__all__ = [
    "MediaForensicsConfig",
    "ForensicAnalyzer",
    "ForensicResult",
]
