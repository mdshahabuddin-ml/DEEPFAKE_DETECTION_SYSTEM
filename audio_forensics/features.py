"""Audio feature extraction for deepfake detection.

This module extracts various audio features for analysis, including:
- Log-Mel spectrograms
- MFCC (Mel-Frequency Cepstral Coefficients)
- Spectral characteristics
- Pitch-related statistics
- Energy characteristics
- Frequency-domain features
- Speech quality indicators

The architecture is extensible to support future pretrained speech
representation models.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import AudioForensicsConfig

logger = logging.getLogger(__name__)


class FeatureExtractionError(Exception):
    """Custom exception for feature extraction errors."""
    pass


@dataclass
class FeatureSet:
    """Container for extracted audio features."""
    # Spectral features
    log_mel: Optional[np.ndarray] = None
    mfcc: Optional[np.ndarray] = None
    mfcc_delta: Optional[np.ndarray] = None
    mfcc_delta2: Optional[np.ndarray] = None
    chroma: Optional[np.ndarray] = None
    spectral_centroid: Optional[np.ndarray] = None
    spectral_bandwidth: Optional[np.ndarray] = None
    spectral_rolloff: Optional[np.ndarray] = None
    spectral_contrast: Optional[np.ndarray] = None

    # Temporal features
    zero_crossing_rate: Optional[np.ndarray] = None
    rms_energy: Optional[np.ndarray] = None
    tempo: Optional[float] = None

    # Pitch features
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    pitch_range: Optional[float] = None
    pitch_contour: Optional[np.ndarray] = None

    # Statistics
    duration: float = 0.0
    sample_rate: int = 0
    n_frames: int = 0
    feature_names: List[str] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """Flatten all features to a 1D array for simple models."""
        features = []
        for name in self.feature_names:
            feat = getattr(self, name)
            if feat is not None:
                if isinstance(feat, np.ndarray):
                    # Use statistics for high-dimensional features
                    if feat.size > 100:
                        features.extend([
                            float(np.mean(feat)),
                            float(np.std(feat)),
                            float(np.min(feat)),
                            float(np.max(feat)),
                            float(np.median(feat)),
                        ])
                    else:
                        features.extend(feat.flatten().tolist())
                elif isinstance(feat, (int, float)):
                    features.append(float(feat))
        return np.array(features, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for name in self.feature_names:
            feat = getattr(self, name)
            if feat is not None:
                if isinstance(feat, np.ndarray):
                    result[name] = feat.tolist() if feat.size < 1000 else {
                        "mean": float(np.mean(feat)),
                        "std": float(np.std(feat)),
                        "shape": list(feat.shape),
                    }
                else:
                    result[name] = feat
        return result


class AudioFeatureExtractor:
    """Extract audio features for deepfake detection.

    This class provides an extensible feature extraction pipeline
    that can work with various audio analysis methods.

    Usage:
        config = AudioForensicsConfig()
        extractor = AudioFeatureExtractor(config)
        features = extractor.extract(audio_array, sample_rate)
    """

    def __init__(self, config: Optional[AudioForensicsConfig] = None):
        """Initialize feature extractor.

        Args:
            config: Audio forensics configuration.
        """
        self.config = config or AudioForensicsConfig()
        self._check_dependencies()

    def _check_dependencies(self):
        """Check for available feature extraction libraries."""
        self._has_librosa = False
        self._has_scipy = False
        self._has_torchaudio = False

        try:
            import librosa
            self._has_librosa = True
            logger.debug("librosa available for feature extraction")
        except ImportError:
            logger.warning("librosa not available, using fallback methods")

        try:
            import scipy.signal
            import scipy.stats
            self._has_scipy = True
            logger.debug("scipy available for feature extraction")
        except ImportError:
            logger.warning("scipy not available for feature extraction")

        try:
            import torchaudio
            self._has_torchaudio = True
            logger.debug("torchaudio available for feature extraction")
        except ImportError:
            pass

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int,
        feature_names: Optional[List[str]] = None,
    ) -> FeatureSet:
        """Extract features from audio signal.

        Args:
            audio: Audio signal as numpy array.
            sample_rate: Audio sample rate in Hz.
            feature_names: Optional list of specific features to extract.
                          If None, extracts all available features.

        Returns:
            FeatureSet containing extracted features.

        Raises:
            FeatureExtractionError: If feature extraction fails.
        """
        if audio is None or len(audio) == 0:
            raise FeatureExtractionError("Empty audio signal")

        if sample_rate <= 0:
            raise FeatureExtractionError(f"Invalid sample rate: {sample_rate}")

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Ensure float32
        audio = audio.astype(np.float32)

        # Normalize if needed
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        features = FeatureSet(
            duration=len(audio) / sample_rate,
            sample_rate=sample_rate,
        )

        try:
            # Determine which features to extract
            if feature_names is None:
                feature_names = self._get_default_features()

            # Extract each feature
            for name in feature_names:
                try:
                    method = getattr(self, f"_extract_{name}", None)
                    if method:
                        value = method(audio, sample_rate)
                        setattr(features, name, value)
                        features.feature_names.append(name)
                except Exception as e:
                    logger.warning(f"Failed to extract {name}: {e}")

            # Calculate frame count
            if features.log_mel is not None:
                features.n_frames = features.log_mel.shape[1]
            elif features.mfcc is not None:
                features.n_frames = features.mfcc.shape[1]

            return features

        except Exception as e:
            raise FeatureExtractionError(f"Feature extraction failed: {e}") from e

    def _get_default_features(self) -> List[str]:
        """Get list of default features to extract."""
        return [
            "log_mel",
            "mfcc",
            "mfcc_delta",
            "mfcc_delta2",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "zero_crossing_rate",
            "rms_energy",
        ]

    def _extract_log_mel(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract Log-Mel spectrogram."""
        if self._has_librosa:
            import librosa
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                fmin=self.config.f_min,
                fmax=self.config.f_max,
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            return log_mel
        else:
            return self._extract_log_mel_fallback(audio, sample_rate)

    def _extract_log_mel_fallback(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Fallback Log-Mel extraction using scipy."""
        if not self._has_scipy:
            raise FeatureExtractionError("No library available for Log-Mel extraction")

        from scipy.signal import stft

        # Compute STFT
        f, t, Zxx = stft(
            audio,
            fs=sample_rate,
            nperseg=self.config.n_fft,
            noverlap=self.config.n_fft - self.config.hop_length,
        )

        # Compute power spectrogram
        power_spec = np.abs(Zxx) ** 2

        # Create mel filterbank
        mel_fb = self._mel_filterbank(
            sample_rate, self.config.n_fft // 2 + 1,
            self.config.n_mels,
            self.config.f_min, self.config.f_max
        )

        # Apply mel filterbank
        mel_spec = mel_fb @ power_spec

        # Log scale
        log_mel = np.log(mel_spec + 1e-10)

        return log_mel

    def _mel_filterbank(
        self,
        sample_rate: int,
        n_freqs: int,
        n_mels: int,
        f_min: float,
        f_max: float,
    ) -> np.ndarray:
        """Create mel filterbank matrix."""
        # Convert Hz to mel
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)

        # Create evenly spaced mel points
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Convert to FFT bin indices. n_freqs == n_fft // 2 + 1, so the bin
        # width in Hz is sample_rate / n_fft == sample_rate / (2 * (n_freqs - 1)).
        n_fft = 2 * (n_freqs - 1)
        bin_points = np.round(hz_points / (sample_rate / n_fft)).astype(int)
        np.clip(bin_points, 0, n_freqs - 1, out=bin_points)

        # Create filterbank
        filterbank = np.zeros((n_mels, n_freqs))
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising slope
            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)

            # Falling slope
            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def _extract_mfcc(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract MFCC features."""
        if self._has_librosa:
            import librosa
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=self.config.n_mfcc,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            )
            return mfcc
        else:
            return self._extract_mfcc_fallback(audio, sample_rate)

    def _extract_mfcc_fallback(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Fallback MFCC extraction using DCT of Log-Mel."""
        log_mel = self._extract_log_mel(audio, sample_rate)

        # Apply DCT to get MFCC
        n_mfcc = self.config.n_mfcc
        n_mels = log_mel.shape[0]

        # Create DCT matrix
        dct_matrix = np.zeros((n_mfcc, n_mels))
        for i in range(n_mfcc):
            for j in range(n_mels):
                dct_matrix[i, j] = np.cos(np.pi * i * (j + 0.5) / n_mels)

        # Apply DCT
        mfcc = dct_matrix @ log_mel

        return mfcc

    def _extract_mfcc_delta(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract MFCC delta (first derivative)."""
        mfcc = self._extract_mfcc(audio, sample_rate)

        if self._has_librosa:
            import librosa
            return librosa.feature.delta(mfcc)
        else:
            return self._delta_fallback(mfcc)

    def _extract_mfcc_delta2(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract MFCC delta-delta (second derivative)."""
        mfcc = self._extract_mfcc(audio, sample_rate)

        if self._has_librosa:
            import librosa
            return librosa.feature.delta(mfcc, order=2)
        else:
            delta = self._delta_fallback(mfcc)
            return self._delta_fallback(delta)

    def _delta_fallback(self, feature: np.ndarray) -> np.ndarray:
        """Fallback delta calculation."""
        # Simple finite difference (guard narrow feature matrices)
        if feature.shape[1] < 2:
            return np.zeros_like(feature)
        delta = np.zeros_like(feature)
        delta[:, 1:-1] = feature[:, 2:] - feature[:, :-2]
        delta[:, 0] = delta[:, 1]
        delta[:, -1] = delta[:, -2]
        return delta / 2.0

    def _extract_spectral_centroid(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract spectral centroid."""
        if self._has_librosa:
            import librosa
            return librosa.feature.spectral_centroid(
                y=audio, sr=sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            )[0]
        else:
            return self._spectral_centroid_fallback(audio, sample_rate)

    def _spectral_centroid_fallback(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Fallback spectral centroid calculation."""
        if not self._has_scipy:
            return np.array([0.0])

        from scipy.signal import stft

        f, t, Zxx = stft(
            audio, fs=sample_rate,
            nperseg=self.config.n_fft,
            noverlap=self.config.n_fft - self.config.hop_length,
        )

        magnitude = np.abs(Zxx)
        freqs = f.reshape(-1, 1)

        # Weighted average of frequencies
        centroid = np.sum(freqs * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-10)
        return centroid

    def _extract_spectral_bandwidth(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract spectral bandwidth."""
        if self._has_librosa:
            import librosa
            return librosa.feature.spectral_bandwidth(
                y=audio, sr=sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            )[0]
        else:
            return self._spectral_bandwidth_fallback(audio, sample_rate)

    def _spectral_bandwidth_fallback(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Fallback spectral bandwidth calculation."""
        if not self._has_scipy:
            return np.array([0.0])

        from scipy.signal import stft

        f, t, Zxx = stft(
            audio, fs=sample_rate,
            nperseg=self.config.n_fft,
            noverlap=self.config.n_fft - self.config.hop_length,
        )

        magnitude = np.abs(Zxx)
        centroid = self._spectral_centroid_fallback(audio, sample_rate)
        freqs = f.reshape(-1, 1)

        # Weighted standard deviation
        bandwidth = np.sqrt(
            np.sum(((freqs - centroid) ** 2) * magnitude, axis=0) /
            (np.sum(magnitude, axis=0) + 1e-10)
        )
        return bandwidth

    def _extract_spectral_rolloff(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract spectral rolloff."""
        if self._has_librosa:
            import librosa
            return librosa.feature.spectral_rolloff(
                y=audio, sr=sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            )[0]
        else:
            return np.array([0.0])

    def _extract_spectral_contrast(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract spectral contrast."""
        if self._has_librosa:
            import librosa
            return librosa.feature.spectral_contrast(
                y=audio, sr=sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            )
        else:
            return np.array([])

    def _extract_chroma(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract chroma features."""
        if self._has_librosa:
            import librosa
            return librosa.feature.chroma_stft(
                y=audio, sr=sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            )
        else:
            return np.array([])

    def _extract_zero_crossing_rate(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract zero crossing rate."""
        if self._has_librosa:
            import librosa
            return librosa.feature.zero_crossing_rate(
                audio, frame_length=self.config.n_fft,
                hop_length=self.config.hop_length,
            )[0]
        else:
            return self._zcr_fallback(audio, sample_rate)

    def _zcr_fallback(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Fallback ZCR calculation."""
        frame_length = self.config.n_fft
        hop_length = self.config.hop_length

        # Number of frames
        n_frames = max(1, 1 + (len(audio) - frame_length) // hop_length)
        zcr = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            # Count sign changes
            zcr[i] = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / frame_length

        return zcr

    def _extract_rms_energy(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract RMS energy."""
        if self._has_librosa:
            import librosa
            return librosa.feature.rms(
                y=audio, frame_length=self.config.n_fft,
                hop_length=self.config.hop_length,
            )[0]
        else:
            return self._rms_fallback(audio, sample_rate)

    def _rms_fallback(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Fallback RMS calculation."""
        frame_length = self.config.n_fft
        hop_length = self.config.hop_length

        n_frames = max(1, 1 + (len(audio) - frame_length) // hop_length)
        rms = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            rms[i] = np.sqrt(np.mean(frame ** 2))

        return rms

    def _extract_tempo(
        self, audio: np.ndarray, sample_rate: int
    ) -> float:
        """Extract tempo (BPM)."""
        if self._has_librosa:
            import librosa
            onset_env = librosa.onset.onset_strength(
                y=audio, sr=sample_rate,
                hop_length=self.config.hop_length,
            )
            tempo = librosa.beat.tempo(
                onset_envelope=onset_env,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )
            return float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            return 0.0

    def _extract_pitch_contour(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract pitch contour."""
        if self._has_librosa:
            import librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio, sr=sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            )
            # Get pitch with highest magnitude at each frame
            pitch_contour = np.zeros(pitches.shape[1])
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                pitch_contour[t] = pitches[idx, t]
            return pitch_contour
        else:
            return np.array([])

    def _extract_pitch_stats(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[float, float, float]:
        """Extract pitch statistics (mean, std, range)."""
        pitch_contour = self._extract_pitch_contour(audio, sample_rate)

        if len(pitch_contour) == 0:
            return 0.0, 0.0, 0.0

        # Remove zero pitches (unvoiced)
        voiced = pitch_contour[pitch_contour > 0]

        if len(voiced) == 0:
            return 0.0, 0.0, 0.0

        return (
            float(np.mean(voiced)),
            float(np.std(voiced)),
            float(np.max(voiced) - np.min(voiced)),
        )

    def extract_batch(
        self,
        audio_batch: List[np.ndarray],
        sample_rate: int,
        feature_names: Optional[List[str]] = None,
    ) -> List[FeatureSet]:
        """Extract features from a batch of audio signals.

        Args:
            audio_batch: List of audio signals.
            sample_rate: Sample rate.
            feature_names: Optional specific features to extract.

        Returns:
            List of FeatureSet objects.
        """
        results = []
        for audio in audio_batch:
            try:
                features = self.extract(audio, sample_rate, feature_names)
                results.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features: {e}")
                results.append(FeatureSet())
        return results


def get_feature_extractor(
    config: Optional[AudioForensicsConfig] = None,
) -> AudioFeatureExtractor:
    """Factory function to get feature extractor."""
    return AudioFeatureExtractor(config)
