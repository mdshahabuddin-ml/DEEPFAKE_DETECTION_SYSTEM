"""Main forensic analyzer orchestrator."""
import logging
import uuid
from typing import Any, Dict, List, Optional

from .config import MediaForensicsConfig
from .file_hasher import FileHasher
from .format_analyzer import FormatAnalyzer
from .integrity_checker import IntegrityChecker
from .metadata_analyzer import MetadataAnalyzer

logger = logging.getLogger(__name__)


class ForensicAnalyzerError(Exception):
    """Custom exception for forensic analysis errors."""
    pass


class ForensicResult:
    """Result of forensic analysis."""

    def __init__(
        self,
        file_path: str = "",
        evidence_id: str = "",
        hashes: Dict[str, str] = None,
        format_info: Dict[str, Any] = None,
        integrity: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        manipulation_indicators: List[str] = None,
    ):
        self.file_path = file_path
        self.evidence_id = evidence_id
        self.hashes = hashes or {}
        self.format_info = format_info or {}
        self.integrity = integrity or {}
        self.metadata = metadata or {}
        self.manipulation_indicators = manipulation_indicators or []

    @property
    def is_valid(self) -> bool:
        return self.integrity.get("valid", False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "evidence_id": self.evidence_id,
            "hashes": self.hashes,
            "format_info": self.format_info,
            "integrity": self.integrity,
            "metadata": self.metadata,
            "manipulation_indicators": self.manipulation_indicators,
            "is_valid": self.is_valid,
        }

    def get_summary(self) -> str:
        lines = [
            "=" * 60,
            "MEDIA FORENSIC ANALYSIS REPORT",
            "=" * 60,
            f"File: {self.file_path}",
            f"Evidence ID: {self.evidence_id}",
            "",
            "HASHES",
            "-" * 40,
        ]
        for algo, hash_val in self.hashes.items():
            lines.append(f"  {algo.upper()}: {hash_val[:32]}...")

        lines.extend(["", "FORMAT", "-" * 40])
        for key in ["media_type", "extension", "file_size_mb", "duration"]:
            if key in self.format_info:
                lines.append(f"  {key}: {self.format_info[key]}")

        lines.extend(["", "INTEGRITY", "-" * 40])
        lines.append(f"  Valid: {self.is_valid}")

        if self.manipulation_indicators:
            lines.extend(["", "INDICATORS", "-" * 40])
            for ind in self.manipulation_indicators:
                lines.append(f"  - {ind}")

        lines.append("=" * 60)
        return "\n".join(lines)


class ForensicAnalyzer:
    """Main forensic analysis orchestrator.

    Usage:
        config = MediaForensicsConfig()
        analyzer = ForensicAnalyzer(config)
        result = analyzer.analyze("video.mp4")
    """

    def __init__(self, config: Optional[MediaForensicsConfig] = None):
        self.config = config or MediaForensicsConfig()
        self._hasher = FileHasher(self.config.hash_algorithms)
        self._format = FormatAnalyzer()
        self._integrity = IntegrityChecker()
        self._metadata = MetadataAnalyzer(self.config.sanitize_metadata)

    def analyze(self, file_path: str) -> ForensicResult:
        """Perform complete forensic analysis.

        Args:
            file_path: Path to media file.

        Returns:
            ForensicResult with analysis results.
        """
        evidence_id = self._hasher.generate_evidence_id(file_path)

        try:
            # Hash file
            hashes = self._hasher.hash_file(file_path)

            # Analyze format
            format_info = self._format.analyze(file_path)

            # Check integrity
            integrity = self._integrity.check_file(file_path)

            # Extract metadata
            metadata = self._metadata.analyze(file_path)

            # Detect indicators
            indicators = self._metadata.detect_manipulation_indicators(metadata)

            return ForensicResult(
                file_path=file_path,
                evidence_id=evidence_id,
                hashes=hashes,
                format_info=format_info,
                integrity=integrity,
                metadata=metadata,
                manipulation_indicators=indicators.get("indicators", []),
            )

        except Exception as e:
            logger.error(f"Forensic analysis failed: {e}")
            return ForensicResult(
                file_path=file_path,
                evidence_id=evidence_id,
                integrity={"valid": False, "error": str(e)},
            )

    def verify_hash(
        self, file_path: str, expected_hash: str, algorithm: str = "sha256"
    ) -> bool:
        """Verify file hash."""
        return self._hasher.verify_integrity(file_path, expected_hash, algorithm)

    def get_quick_info(self, file_path: str) -> Dict[str, Any]:
        """Get quick file information."""
        return self._format.analyze(file_path)


def get_forensic_analyzer(
    config: Optional[MediaForensicsConfig] = None,
) -> ForensicAnalyzer:
    """Factory function to get forensic analyzer."""
    return ForensicAnalyzer(config)
