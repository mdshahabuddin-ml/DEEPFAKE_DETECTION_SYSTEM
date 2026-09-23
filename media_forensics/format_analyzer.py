"""Format detection and analysis."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FormatAnalyzer:
    """Analyze media file format and codec information."""

    def __init__(self):
        self._has_ffprobe = False
        self._check_ffprobe()

    def _check_ffprobe(self):
        """Check if ffprobe is available."""
        import subprocess
        try:
            subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True, timeout=5
            )
            self._has_ffprobe = True
        except Exception:
            self._has_ffprobe = False

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze file format.

        Args:
            file_path: Path to media file.

        Returns:
            Dictionary with format information.
        """
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        ext = Path(file_path).suffix.lower()
        file_size = os.path.getsize(file_path)

        result = {
            "file_path": file_path,
            "extension": ext,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
        }

        # Detect media type
        result["media_type"] = self._detect_media_type(ext)

        # Try ffprobe for detailed info
        if self._has_ffprobe:
            probe_info = self._probe_file(file_path)
            result.update(probe_info)

        return result

    def _detect_media_type(self, ext: str) -> str:
        """Detect media type from extension."""
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"}
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}

        if ext in video_exts:
            return "video"
        elif ext in audio_exts:
            return "audio"
        elif ext in image_exts:
            return "image"
        else:
            return "unknown"

    def _probe_file(self, file_path: str) -> Dict[str, Any]:
        """Probe file with ffprobe."""
        import subprocess
        import json

        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                file_path
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                info = json.loads(result.stdout)
                return self._parse_probe_info(info)
            else:
                return {"probe_error": result.stderr}

        except Exception as e:
            return {"probe_error": str(e)}

    def _parse_probe_info(self, info: dict) -> Dict[str, Any]:
        """Parse ffprobe output."""
        result = {}

        # Format info
        fmt = info.get("format", {})
        if fmt:
            result["format_name"] = fmt.get("format_name", "")
            result["format_long_name"] = fmt.get("format_long_name", "")
            result["duration"] = float(fmt.get("duration", 0))
            result["bit_rate"] = int(fmt.get("bit_rate", 0))

        # Stream info
        streams = info.get("streams", [])
        for i, stream in enumerate(streams):
            codec_type = stream.get("codec_type", "")
            prefix = f"stream_{i}"

            if codec_type == "video":
                result[f"{prefix}_type"] = "video"
                result[f"{prefix}_codec"] = stream.get("codec_name", "")
                result[f"{prefix}_width"] = stream.get("width", 0)
                result[f"{prefix}_height"] = stream.get("height", 0)
                result[f"{prefix}_fps"] = eval(stream.get("r_frame_rate", "0/1"))
            elif codec_type == "audio":
                result[f"{prefix}_type"] = "audio"
                result[f"{prefix}_codec"] = stream.get("codec_name", "")
                result[f"{prefix}_sample_rate"] = stream.get("sample_rate", 0)
                result[f"{prefix}_channels"] = stream.get("channels", 0)

        return result

    def get_codec_info(self, file_path: str) -> Dict[str, str]:
        """Get codec information."""
        info = self.analyze(file_path)
        codecs = {}
        for key, val in info.items():
            if "codec" in key and isinstance(val, str):
                codecs[key] = val
        return codecs
