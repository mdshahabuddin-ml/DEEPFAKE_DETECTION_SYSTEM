"""Configuration for media forensics module."""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MediaForensicsConfig:
    """Configuration for media forensics."""
    # Hash algorithms
    hash_algorithms: List[str] = field(
        default_factory=lambda: ["sha256", "md5", "sha1"]
    )

    # Supported formats
    supported_video_formats: List[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]
    )
    supported_audio_formats: List[str] = field(
        default_factory=lambda: [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma"]
    )
    supported_image_formats: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    )

    # File size limits
    max_file_size_mb: float = 500.0

    # Metadata
    extract_exif: bool = True
    extract_format_info: bool = True
    sanitize_metadata: bool = True

    @classmethod
    def from_env(cls) -> "MediaForensicsConfig":
        return cls(
            max_file_size_mb=float(os.getenv("MEDIA_MAX_SIZE_MB", "500.0")),
        )
