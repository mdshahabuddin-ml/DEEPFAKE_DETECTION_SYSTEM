"""Media integrity checking."""
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class IntegrityChecker:
    """Check media file integrity."""

    def __init__(self):
        pass

    def check_file(self, file_path: str) -> Dict[str, Any]:
        """Check file integrity.

        Args:
            file_path: Path to file.

        Returns:
            Integrity check results.
        """
        if not os.path.exists(file_path):
            return {"valid": False, "error": "File not found"}

        results = {
            "valid": True,
            "checks": [],
        }

        # Check file size
        size_result = self._check_file_size(file_path)
        results["checks"].append(size_result)
        if not size_result["passed"]:
            results["valid"] = False

        # Check file header
        header_result = self._check_file_header(file_path)
        results["checks"].append(header_result)
        if not header_result["passed"]:
            results["valid"] = False

        # Check for truncation
        truncation_result = self._check_truncation(file_path)
        results["checks"].append(truncation_result)
        if not truncation_result["passed"]:
            results["valid"] = False

        return results

    def _check_file_size(self, file_path: str) -> Dict[str, Any]:
        """Check if file size is reasonable."""
        size = os.path.getsize(file_path)
        return {
            "check": "file_size",
            "passed": size > 0,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
        }

    def _check_file_header(self, file_path: str) -> Dict[str, Any]:
        """Check file header matches extension."""
        ext = os.path.splitext(file_path)[1].lower()

        # Expected magic bytes
        magic_bytes = {
            ".mp4": [b"ftyp"],
            ".wav": [b"RIFF"],
            ".mp3": [b"ID3", b"\xff\xfb", b"\xff\xf3"],
            ".jpg": [b"\xff\xd8\xff"],
            ".png": [b"\x89PNG"],
            ".gif": [b"GIF8"],
            ".avi": [b"RIFF"],
        }

        expected = magic_bytes.get(ext)
        if not expected:
            return {"check": "file_header", "passed": True, "note": "No magic bytes to check"}

        try:
            with open(file_path, "rb") as f:
                header = f.read(16)

            matched = any(header.startswith(magic) for magic in expected)
            return {
                "check": "file_header",
                "passed": matched,
                "header_hex": header[:8].hex(),
            }
        except Exception as e:
            return {"check": "file_header", "passed": False, "error": str(e)}

    def _check_truncation(self, file_path: str) -> Dict[str, Any]:
        """Check if file appears truncated."""
        ext = os.path.splitext(file_path)[1].lower()
        size = os.path.getsize(file_path)

        # Minimum expected sizes
        min_sizes = {
            ".mp4": 1024,
            ".wav": 44,
            ".mp3": 100,
            ".jpg": 100,
            ".png": 100,
        }

        min_size = min_sizes.get(ext, 0)
        passed = size >= min_size

        return {
            "check": "truncation",
            "passed": passed,
            "min_expected": min_size,
        }

    def detect_compression_artifacts(
        self, file_path: str
    ) -> Dict[str, Any]:
        """Detect potential compression artifacts."""
        ext = os.path.splitext(file_path)[1].lower()

        # Simple heuristic based on extension
        is_compressed = ext in [".mp4", ".mp3", ".jpg", ".webm", ".webp"]

        return {
            "is_compressed": is_compressed,
            "format": ext,
            "note": "Compression detected" if is_compressed else "Lossless or uncompressed format",
        }
