"""File hashing for forensic evidence."""
import hashlib
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FileHasher:
    """Calculate file hashes for forensic evidence."""

    def __init__(self, algorithms: list = None):
        self.algorithms = algorithms or ["sha256", "md5"]

    def hash_file(self, file_path: str) -> Dict[str, str]:
        """Calculate hashes for a file.

        Args:
            file_path: Path to file.

        Returns:
            Dictionary of algorithm -> hash value.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        hashes = {}
        for algo in self.algorithms:
            hashes[algo] = self._calculate_hash(file_path, algo)

        return hashes

    def hash_bytes(self, data: bytes, algorithms: list = None) -> Dict[str, str]:
        """Calculate hashes for bytes data."""
        algorithms = algorithms or self.algorithms
        hashes = {}

        for algo in algorithms:
            try:
                h = hashlib.new(algo)
                h.update(data)
                hashes[algo] = h.hexdigest()
            except Exception as e:
                logger.warning(f"Failed to calculate {algo}: {e}")

        return hashes

    def _calculate_hash(self, file_path: str, algorithm: str) -> str:
        """Calculate hash for file."""
        try:
            h = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            return ""

    def generate_evidence_id(self, file_path: str) -> str:
        """Generate unique evidence ID from file hash."""
        sha256 = self._calculate_hash(file_path, "sha256")
        return f"EV-{sha256[:16].upper()}" if sha256 else "EV-UNKNOWN"

    def verify_integrity(
        self, file_path: str, expected_hash: str, algorithm: str = "sha256"
    ) -> bool:
        """Verify file integrity against expected hash."""
        actual_hash = self._calculate_hash(file_path, algorithm)
        return actual_hash.lower() == expected_hash.lower()
