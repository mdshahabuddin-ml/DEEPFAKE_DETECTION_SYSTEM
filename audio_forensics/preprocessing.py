"""Audio preprocessing pipeline for deepfake detection.

This module handles audio loading, validation, normalization, and
preprocessing without modifying the original audio files.

Privacy:
    - Original audio files are never modified
    - Temporary working copies are created for processing
    - All operations work on numpy arrays, not file handles
"""

import hashlib
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import AudioForensicsConfig
from .schemas import AudioProperties, AudioStatus

logger = logging.getLogger(__name__)


class AudioPreprocessingError(Exception):
    """Custom exception for audio preprocessing errors."""
    pass


class AudioValidationError(AudioPreprocessingError):
    """Exception for audio validation failures."""
    pass


@dataclass
class PreprocessingResult:
    """Result of audio preprocessing."""
    audio: np.ndarray
    sample_rate: int
    properties: AudioProperties
    original_path: str
    file_hash: str = ""
    working_copy_path: Optional[str] = None
    preprocessing_time: float = 0.0
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class AudioPreprocessor:
    """Isolated audio preprocessing pipeline.

    This class handles all audio preprocessing operations without
    modifying the original audio files.

    Usage:
        config = AudioForensicsConfig()
        preprocessor = AudioPreprocessor(config)
        result = preprocessor.preprocess("path/to/audio.wav")
    """

    def __init__(self, config: Optional[AudioForensicsConfig] = None):
        """Initialize preprocessor with configuration.

        Args:
            config: Audio forensics configuration. Uses default if None.
        """
        self.config = config or AudioForensicsConfig()
        self._validate_config()

    def _validate_config(self):
        """Validate configuration on initialization."""
        errors = self.config.validate()
        if errors:
            raise AudioPreprocessingError(
                f"Invalid configuration: {'; '.join(errors)}"
            )

    def preprocess(
        self, audio_path: str, working_dir: Optional[str] = None
    ) -> PreprocessingResult:
        """Preprocess audio file for analysis.

        Creates a working copy and applies normalization without
        modifying the original file.

        Args:
            audio_path: Path to audio file.
            working_dir: Optional directory for working copies.

        Returns:
            PreprocessingResult with processed audio and metadata.

        Raises:
            AudioValidationError: If audio file is invalid.
            AudioPreprocessingError: If preprocessing fails.
        """
        start_time = time.time()

        # Validate input
        self._validate_file(audio_path)

        # Calculate file hash (from original)
        file_hash = self._calculate_hash(audio_path)

        # Create working copy
        working_copy = self._create_working_copy(audio_path, working_dir)

        try:
            # Load audio
            audio, sample_rate = self._load_audio(working_copy)

            # Extract properties before modification
            properties = self._extract_properties(
                audio, sample_rate, audio_path
            )

            # Validate audio properties
            self._validate_audio_properties(properties)

            # Apply preprocessing
            processed_audio, warnings = self._apply_preprocessing(
                audio, sample_rate
            )

            # Update properties after processing
            properties = self._update_properties(
                properties, processed_audio, sample_rate
            )

            preprocessing_time = time.time() - start_time

            return PreprocessingResult(
                audio=processed_audio,
                sample_rate=sample_rate,
                properties=properties,
                original_path=audio_path,
                file_hash=file_hash,
                working_copy_path=working_copy,
                preprocessing_time=preprocessing_time,
                warnings=warnings,
            )

        except Exception as e:
            # Clean up working copy on error
            self._cleanup_working_copy(working_copy)
            raise AudioPreprocessingError(f"Preprocessing failed: {e}") from e

    def _validate_file(self, audio_path: str):
        """Validate audio file exists and has supported format."""
        if not os.path.exists(audio_path):
            raise AudioValidationError(f"File not found: {audio_path}")

        if not os.path.isfile(audio_path):
            raise AudioValidationError(f"Not a file: {audio_path}")

        # Check file extension
        ext = Path(audio_path).suffix.lower()
        if ext not in self.config.supported_formats:
            raise AudioValidationError(
                f"Unsupported format '{ext}'. "
                f"Supported: {', '.join(self.config.supported_formats)}"
            )

        # Check file size
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise AudioValidationError(
                f"File too large: {file_size_mb:.1f}MB "
                f"(max: {self.config.max_file_size_mb}MB)"
            )

    def _calculate_hash(self, audio_path: str) -> str:
        """Calculate SHA-256 hash of audio file."""
        sha256_hash = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _create_working_copy(
        self, audio_path: str, working_dir: Optional[str] = None
    ) -> str:
        """Create a temporary working copy of the audio file."""
        try:
            # Create temporary file with same extension
            ext = Path(audio_path).suffix
            temp_dir = working_dir or tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)

            # Create unique temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix=ext,
                dir=temp_dir,
                delete=False,
                prefix="audio_forensics_",
            )
            temp_path = temp_file.name
            temp_file.close()

            # Copy original to working copy
            import shutil
            shutil.copy2(audio_path, temp_path)

            logger.debug(f"Created working copy: {temp_path}")
            return temp_path

        except Exception as e:
            raise AudioPreprocessingError(
                f"Failed to create working copy: {e}"
            ) from e

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file using available library.

        Tries soundfile first, falls back to scipy, then wave.
        """
        # Try soundfile first (most reliable)
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path, dtype="float32")
            logger.debug(f"Loaded audio with soundfile: sr={sr}")
            return audio, sr
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"soundfile failed: {e}")

        # Try scipy
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(audio_path)
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            logger.debug(f"Loaded audio with scipy: sr={sr}")
            return audio, sr
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"scipy failed: {e}")

        # Try wave module (built-in)
        try:
            import wave
            with wave.open(audio_path, "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

                # Convert to numpy array
                if sampwidth == 2:
                    audio = np.frombuffer(raw_data, dtype=np.int16)
                elif sampwidth == 4:
                    audio = np.frombuffer(raw_data, dtype=np.int32)
                else:
                    raise AudioPreprocessingError(
                        f"Unsupported sample width: {sampwidth}"
                    )

                # Convert to float32
                audio = audio.astype(np.float32) / np.iinfo(
                    audio.dtype
                ).max

                # Handle stereo
                if n_channels > 1:
                    audio = audio.reshape(-1, n_channels).mean(axis=1)

                logger.debug(f"Loaded audio with wave: sr={sr}")
                return audio, sr

        except Exception as e:
            raise AudioPreprocessingError(
                f"Failed to load audio with any method: {e}"
            ) from e

    def _extract_properties(
        self, audio: np.ndarray, sample_rate: int, file_path: str
    ) -> AudioProperties:
        """Extract audio properties without modification."""
        duration = len(audio) / sample_rate

        # Calculate levels
        rms_level = float(np.sqrt(np.mean(audio ** 2)))
        peak_level = float(np.max(np.abs(audio)))

        # Estimate SNR (simple estimate)
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(audio[:int(sample_rate * 0.1)] ** 2) if len(audio) > sample_rate * 0.1 else 1e-10
        snr_estimate = 10 * np.log10(signal_power / (noise_power + 1e-10))

        # Check for clipping
        has_clipping = float(np.max(np.abs(audio))) > 0.99

        # Check for silence
        has_silence = rms_level < self.config.silence_threshold_db / 100

        return AudioProperties(
            duration=duration,
            sample_rate=sample_rate,
            channels=1,  # After mono conversion
            format=Path(file_path).suffix.lstrip(".").upper(),
            file_size_bytes=os.path.getsize(file_path),
            rms_level=rms_level,
            peak_level=peak_level,
            snr_estimate=snr_estimate,
            has_clipping=has_clipping,
            has_silence=has_silence,
        )

    def _validate_audio_properties(self, properties: AudioProperties):
        """Validate audio properties against configuration."""
        if properties.duration < self.config.min_duration_seconds:
            raise AudioValidationError(
                f"Audio too short: {properties.duration:.2f}s "
                f"(min: {self.config.min_duration_seconds}s)"
            )

        if properties.duration > self.config.max_duration_seconds:
            raise AudioValidationError(
                f"Audio too long: {properties.duration:.2f}s "
                f"(max: {self.config.max_duration_seconds}s)"
            )

    def _apply_preprocessing(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, list]:
        """Apply preprocessing steps to audio."""
        warnings = []
        processed = audio.copy()

        # Resample if needed
        if sample_rate != self.config.target_sample_rate:
            processed, warnings_resample = self._resample(
                processed, sample_rate, self.config.target_sample_rate
            )
            warnings.extend(warnings_resample)
            sample_rate = self.config.target_sample_rate

        # Normalize if enabled
        if self.config.normalize_audio:
            processed = self._normalize(processed, self.config.target_db_level)

        # Remove silence if enabled
        if self.config.remove_silence:
            processed, removed = self._remove_silence(
                processed, sample_rate, self.config.silence_threshold_db
            )
            if removed > 0:
                warnings.append(f"Removed {removed:.1f}s of silence")

        # Ensure float32
        processed = processed.astype(np.float32)

        return processed, warnings

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> Tuple[np.ndarray, list]:
        """Resample audio to target sample rate."""
        warnings = []

        try:
            import librosa
            resampled = librosa.resample(
                audio, orig_sr=orig_sr, target_sr=target_sr
            )
            return resampled, warnings
        except ImportError:
            warnings.append(
                "librosa not available, using scipy resampling"
            )

        # Fallback to scipy
        try:
            from scipy.signal import resample
            num_samples = int(len(audio) * target_sr / orig_sr)
            resampled = resample(audio, num_samples)
            return resampled, warnings
        except ImportError:
            warnings.append(
                "scipy not available, skipping resampling"
            )
            return audio, warnings

    def _normalize(self, audio: np.ndarray, target_db: float) -> np.ndarray:
        """Normalize audio to target dB level."""
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < 1e-10:
            return audio

        # Calculate target RMS
        target_rms = 10 ** (target_db / 20)

        # Apply normalization
        normalized = audio * (target_rms / rms)

        # Clip to prevent overflow
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized

    def _remove_silence(
        self, audio: np.ndarray, sample_rate: int, threshold_db: float
    ) -> Tuple[np.ndarray, int]:
        """Remove silence from audio."""
        # Convert threshold to amplitude
        threshold = 10 ** (threshold_db / 20)

        # Find non-silent regions
        is_speech = np.abs(audio) > threshold

        # Smooth to avoid removing short pauses
        window_size = int(sample_rate * 0.02)  # 20ms window
        if window_size > 0:
            kernel = np.ones(window_size) / window_size
            is_speech_smooth = np.convolve(is_speech.astype(float), kernel, mode="same") > 0.5
        else:
            is_speech_smooth = is_speech

        # Extract non-silent segments
        speech_indices = np.where(is_speech_smooth)[0]

        if len(speech_indices) == 0:
            return audio, 0

        # Calculate removed duration
        removed_duration = int(
            (len(audio) - len(speech_indices)) / sample_rate
        )

        return audio[speech_indices], removed_duration

    def _update_properties(
        self,
        original: AudioProperties,
        processed: np.ndarray,
        sample_rate: int,
    ) -> AudioProperties:
        """Update properties after processing."""
        original.duration = len(processed) / sample_rate
        original.sample_rate = sample_rate
        original.rms_level = float(np.sqrt(np.mean(processed ** 2)))
        original.peak_level = float(np.max(np.abs(processed)))
        return original

    def _cleanup_working_copy(self, working_copy: Optional[str]):
        """Clean up temporary working copy."""
        if working_copy and os.path.exists(working_copy):
            try:
                os.unlink(working_copy)
                logger.debug(f"Cleaned up working copy: {working_copy}")
            except Exception as e:
                logger.warning(f"Failed to clean up working copy: {e}")


def get_preprocessor(
    config: Optional[AudioForensicsConfig] = None,
) -> AudioPreprocessor:
    """Factory function to get audio preprocessor."""
    return AudioPreprocessor(config)
