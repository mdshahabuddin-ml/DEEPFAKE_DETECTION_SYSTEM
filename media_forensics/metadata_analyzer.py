"""Metadata extraction and analysis."""
import logging
import os
import struct
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MetadataAnalyzer:
    """Extract and analyze media metadata."""

    def __init__(self, sanitize: bool = True):
        self.sanitize = sanitize

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze file metadata.

        Args:
            file_path: Path to file.

        Returns:
            Metadata dictionary.
        """
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        result = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "modified_time": os.path.getmtime(file_path),
            "created_time": os.path.getctime(file_path),
        }

        ext = os.path.splitext(file_path)[1].lower()

        # Extract format-specific metadata
        if ext in [".mp4", ".mov", ".m4v"]:
            result.update(self._extract_mp4_metadata(file_path))
        elif ext in [".wav"]:
            result.update(self._extract_wav_metadata(file_path))
        elif ext in [".jpg", ".jpeg"]:
            result.update(self._extract_jpeg_metadata(file_path))
        elif ext in [".png"]:
            result.update(self._extract_png_metadata(file_path))

        # Sanitize if requested
        if self.sanitize:
            result = self._sanitize_metadata(result)

        return result

    def _extract_mp4_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract MP4/MOV metadata."""
        metadata = {"format": "mp4"}

        try:
            with open(file_path, "rb") as f:
                # Read ftyp box
                data = f.read(32)
                if len(data) >= 8:
                    box_size = struct.unpack(">I", data[:4])[0]
                    box_type = data[4:8].decode("ascii", errors="ignore")
                    metadata["ftyp"] = box_type

                    # Get major brand
                    if len(data) >= 12:
                        metadata["major_brand"] = data[8:12].decode("ascii", errors="ignore")
        except Exception as e:
            metadata["parse_error"] = str(e)

        return metadata

    def _extract_wav_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract WAV metadata."""
        metadata = {"format": "wav"}

        try:
            with open(file_path, "rb") as f:
                riff = f.read(4)
                if riff == b"RIFF":
                    # Skip size
                    f.read(4)
                    wave = f.read(4)
                    if wave == b"WAVE":
                        # Read fmt chunk
                        chunk_id = f.read(4)
                        chunk_size = struct.unpack("<I", f.read(4))[0]

                        if chunk_id == b"fmt " and chunk_size >= 16:
                            audio_format = struct.unpack("<H", f.read(2))[0]
                            channels = struct.unpack("<H", f.read(2))[0]
                            sample_rate = struct.unpack("<I", f.read(4))[0]
                            byte_rate = struct.unpack("<I", f.read(4))[0]
                            block_align = struct.unpack("<H", f.read(2))[0]
                            bits_per_sample = struct.unpack("<H", f.read(2))[0]

                            metadata["channels"] = channels
                            metadata["sample_rate"] = sample_rate
                            metadata["bits_per_sample"] = bits_per_sample
                            metadata["audio_format"] = audio_format
        except Exception as e:
            metadata["parse_error"] = str(e)

        return metadata

    def _extract_jpeg_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract JPEG EXIF metadata."""
        metadata = {"format": "jpeg"}

        try:
            with open(file_path, "rb") as f:
                # Check SOI marker
                soi = f.read(2)
                if soi != b"\xff\xd8":
                    metadata["valid"] = False
                    return metadata

                # Read markers
                while True:
                    marker = f.read(2)
                    if len(marker) < 2:
                        break

                    if marker[0] != 0xFF:
                        break

                    if marker[1] == 0xE1:  # APP1 (EXIF)
                        size = struct.unpack(">H", f.read(2))[0]
                        exif_data = f.read(size - 2)
                        if exif_data[:6] == b"Exif\x00\x00":
                            metadata["has_exif"] = True
                            break
                    elif marker[1] == 0xDA:  # SOS
                        break
                    else:
                        size = struct.unpack(">H", f.read(2))[0]
                        f.seek(size - 2, 1)
        except Exception as e:
            metadata["parse_error"] = str(e)

        return metadata

    def _extract_png_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PNG metadata."""
        metadata = {"format": "png"}

        try:
            with open(file_path, "rb") as f:
                # Check PNG signature
                signature = f.read(8)
                if signature != b"\x89PNG\r\n\x1a\n":
                    metadata["valid"] = False
                    return metadata

                # Read IHDR chunk
                chunk_size = struct.unpack(">I", f.read(4))[0]
                chunk_type = f.read(4)
                if chunk_type == b"IHDR":
                    width = struct.unpack(">I", f.read(4))[0]
                    height = struct.unpack(">I", f.read(4))[0]
                    bit_depth = struct.unpack("B", f.read(1))[0]
                    color_type = struct.unpack("B", f.read(1))[0]

                    metadata["width"] = width
                    metadata["height"] = height
                    metadata["bit_depth"] = bit_depth
                    metadata["color_type"] = color_type
        except Exception as e:
            metadata["parse_error"] = str(e)

        return metadata

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from metadata."""
        sanitized = {}

        # Keep only safe fields
        safe_fields = [
            "format", "file_name", "file_size", "width", "height",
            "channels", "sample_rate", "bits_per_sample",
        ]

        for key, value in metadata.items():
            if any(safe in key.lower() for safe in safe_fields):
                sanitized[key] = value
            elif key in ["error", "parse_error"]:
                sanitized[key] = value

        return sanitized

    def detect_manipulation_indicators(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect potential manipulation indicators from metadata."""
        indicators = []

        # Check for unusual properties
        if metadata.get("has_exif"):
            indicators.append("EXIF data present")

        if metadata.get("parse_error"):
            indicators.append("Metadata parsing error")

        return {
            "indicators": indicators,
            "n_indicators": len(indicators),
            "has_indicators": len(indicators) > 0,
        }
