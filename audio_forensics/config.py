"""Configuration for Audio Forensics module.

All configuration is isolated and does not modify existing system config.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioForensicsConfig:
    """Configuration for audio forensics pipeline."""

    # Audio preprocessing
    target_sample_rate: int = 16000
    target_channels: int = 1
    target_format: str = "wav"
    max_duration_seconds: float = 300.0
    min_duration_seconds: float = 0.5

    # Normalization
    normalize_audio: bool = True
    target_db_level: float = -20.0
    remove_silence: bool = False
    silence_threshold_db: float = -40.0

    # Feature extraction
    n_mels: int = 128
    n_mfcc: int = 40
    hop_length: int = 512
    n_fft: int = 2048
    f_min: int = 0
    f_max: int = 8000

    # Segmentation
    vad_energy_threshold: float = 0.01
    min_speech_duration: float = 0.5
    max_speech_duration: float = 30.0
    speech_overlap: float = 0.1

    # Detection thresholds
    fake_threshold: float = 0.5
    suspicious_threshold: float = 0.3
    high_confidence_threshold: float = 0.8

    # Model
    model_name: str = "audio_efficientnet"
    model_checkpoint: Optional[str] = None
    device: str = "auto"
    batch_size: int = 1

    # Speaker verification
    speaker_similarity_threshold: float = 0.75
    use_speaker_verification: bool = True

    # Temporal analysis
    segment_duration: float = 3.0
    segment_overlap: float = 0.5

    # Output
    output_dir: str = "results/audio_forensics"
    save_intermediate: bool = False
    verbose: bool = False

    # Safety
    max_file_size_mb: float = 100.0
    supported_formats: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma")
    allow_external_models: bool = False

    @classmethod
    def from_env(cls) -> "AudioForensicsConfig":
        """Create config from environment variables."""
        return cls(
            target_sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            fake_threshold=float(os.getenv("AUDIO_FAKE_THRESHOLD", "0.5")),
            device=os.getenv("AUDIO_DEVICE", "auto"),
            verbose=os.getenv("AUDIO_VERBOSE", "false").lower() == "true",
        )

    def validate(self) -> list:
        """Validate configuration. Returns list of errors."""
        errors = []
        if self.target_sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            errors.append(f"Unsupported sample rate: {self.target_sample_rate}")
        if not 0.0 <= self.fake_threshold <= 1.0:
            errors.append(f"Invalid fake threshold: {self.fake_threshold}")
        if self.max_duration_seconds <= 0:
            errors.append(f"Invalid max duration: {self.max_duration_seconds}")
        if self.min_duration_seconds < 0:
            errors.append(f"Invalid min duration: {self.min_duration_seconds}")
        return errors
