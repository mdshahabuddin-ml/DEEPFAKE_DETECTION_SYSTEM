"""
src/audio_analysis.py
Advanced audio analysis for deepfake detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AudioFeatures:
    duration: float
    sample_rate: int
    rms_energy: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    mfcc_mean: np.ndarray
    mfcc_std: np.ndarray
    pitch_mean: float
    pitch_std: float
    voicing_fraction: float
    snr_estimate: float
    clipping_ratio: float


@dataclass
class AudioAnalysisResult:
    file_path: str
    features: AudioFeatures
    fake_probability: float
    confidence: float
    anomaly_scores: Dict[str, float]
    quality_assessment: Dict[str, Any]
    analysis_time: float = 0.0


class SpectralAnalyzer:
    def __init__(self, sr: int = 16000):
        self.sr = sr

    def extract_features(self, y: np.ndarray) -> Dict[str, Any]:
        if not LIBROSA_AVAILABLE:
            return {}

        features = {}
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spectral_centroid_std"] = float(np.std(spectral_centroid))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        features["spectral_contrast_mean"] = np.mean(spectral_contrast, axis=1)

        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))

        return features

    def compute_spectrogram_anomalies(
        self, y: np.ndarray, n_segments: int = 10
    ) -> Dict[str, float]:
        if not LIBROSA_AVAILABLE:
            return {}

        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        anomalies = {}
        if S_db.shape[1] > n_segments:
            segments = np.array_split(S_db, n_segments, axis=1)
            segment_means = [np.mean(seg) for seg in segments]
            anomalies["spectral_variance"] = float(np.var(segment_means))
            diffs = np.abs(np.diff(segment_means))
            anomalies["max_spectral_change"] = float(np.max(diffs))
            anomalies["mean_spectral_change"] = float(np.mean(diffs))

        return anomalies


class VoiceActivityDetector:
    def __init__(self, sr: int = 16000, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length

    def detect(self, y: np.ndarray) -> Dict[str, Any]:
        if not LIBROSA_AVAILABLE:
            return {"voicing_fraction": 0.5, "voice_segments": []}

        rms = librosa.feature.rms(
            y=y, frame_length=2048, hop_length=self.hop_length
        )[0]

        energy_threshold = np.mean(rms) * 0.3
        voice_mask = rms > energy_threshold
        voicing_fraction = float(np.mean(voice_mask))

        voice_segments = []
        in_segment = False
        start = 0

        for i, is_voice in enumerate(voice_mask):
            if is_voice and not in_segment:
                start = i
                in_segment = True
            elif not is_voice and in_segment:
                voice_segments.append({
                    "start": start * self.hop_length / self.sr,
                    "end": i * self.hop_length / self.sr,
                })
                in_segment = False

        if in_segment:
            voice_segments.append({
                "start": start * self.hop_length / self.sr,
                "end": len(voice_mask) * self.hop_length / self.sr,
            })

        return {
            "voicing_fraction": voicing_fraction,
            "num_segments": len(voice_segments),
            "voice_segments": voice_segments,
            "rms_energy_mean": float(np.mean(rms)),
            "rms_energy_std": float(np.std(rms)),
        }

    def compute_energy_contour(self, y: np.ndarray) -> np.ndarray:
        if not LIBROSA_AVAILABLE:
            return np.array([])

        rms = librosa.feature.rms(
            y=y, frame_length=2048, hop_length=self.hop_length
        )[0]
        return rms


class AudioQualityAnalyzer:
    def __init__(self, sr: int = 16000):
        self.sr = sr

    def analyze(self, y: np.ndarray) -> Dict[str, Any]:
        quality = {}

        quality["duration"] = len(y) / self.sr
        quality["rms_energy"] = float(np.sqrt(np.mean(y**2)))
        quality["peak_amplitude"] = float(np.max(np.abs(y)))

        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) >= clipping_threshold)
        quality["clipping_ratio"] = float(clipped_samples / len(y))
        quality["has_clipping"] = quality["clipping_ratio"] > 0.001

        snr_estimate = self._estimate_snr(y)
        quality["snr_estimate"] = snr_estimate
        quality["is_noisy"] = snr_estimate < 10.0

        quality["dynamic_range"] = float(
            20 * np.log10(
                (np.max(np.abs(y)) + 1e-10) /
                (np.sqrt(np.mean(y**2)) + 1e-10)
            )
        )

        return quality

    def _estimate_snr(self, y: np.ndarray) -> float:
        energy = y ** 2
        noise_floor = np.percentile(energy, 10)
        signal_power = np.mean(energy)
        snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))
        return float(snr)


class AudioDeepfakeAnalyzer:
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.spectral_analyzer = SpectralAnalyzer(sr)
        self.vad = VoiceActivityDetector(sr)
        self.quality_analyzer = AudioQualityAnalyzer(sr)

    def analyze(self, y: np.ndarray, file_path: str = "") -> AudioAnalysisResult:
        start_time = time.time()

        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for audio analysis")

        y = y.astype(np.float32)
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))

        spectral_features = self.spectral_analyzer.extract_features(y)
        vad_result = self.vad.detect(y)
        quality_result = self.quality_analyzer.analyze(y)
        anomaly_scores = self.spectral_analyzer.compute_spectrogram_anomalies(y)

        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            if magnitudes[idx, t] > 0:
                pitch_values.append(pitches[idx, t])
        pitch_mean = float(np.mean(pitch_values)) if pitch_values else 0.0
        pitch_std = float(np.std(pitch_values)) if pitch_values else 0.0

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        features = AudioFeatures(
            duration=len(y) / self.sr,
            sample_rate=self.sr,
            rms_energy=quality_result["rms_energy"],
            zero_crossing_rate=zcr,
            spectral_centroid=spectral_features.get("spectral_centroid_mean", 0.0),
            spectral_bandwidth=spectral_features.get("spectral_bandwidth_mean", 0.0),
            spectral_rolloff=spectral_features.get("spectral_rolloff_mean", 0.0),
            mfcc_mean=mfcc_mean,
            mfcc_std=mfcc_std,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            voicing_fraction=vad_result["voicing_fraction"],
            snr_estimate=quality_result["snr_estimate"],
            clipping_ratio=quality_result["clipping_ratio"],
        )

        fake_probability = self._compute_fake_score(
            spectral_features, vad_result, quality_result, anomaly_scores
        )

        confidence = abs(fake_probability - 0.5) * 2

        analysis_time = time.time() - start_time

        logger.info(
            "Audio analysis complete: fake_prob=%.4f (%.3f seconds)",
            fake_probability, analysis_time
        )

        return AudioAnalysisResult(
            file_path=file_path,
            features=features,
            fake_probability=fake_probability,
            confidence=confidence,
            anomaly_scores=anomaly_scores,
            quality_assessment=quality_result,
            analysis_time=analysis_time,
        )

    def _compute_fake_score(
        self,
        spectral: Dict[str, Any],
        vad: Dict[str, Any],
        quality: Dict[str, Any],
        anomalies: Dict[str, float],
    ) -> float:
        score = 0.0
        weights_total = 0.0

        flatness = spectral.get("spectral_flatness_mean", 0.0)
        if flatness > 0.1:
            score += 0.2 * min(1.0, flatness / 0.3)
        weights_total += 0.2

        spectral_var = anomalies.get("spectral_variance", 0.0)
        if spectral_var > 10.0:
            score += 0.3 * min(1.0, spectral_var / 50.0)
        weights_total += 0.3

        if quality.get("is_noisy", False):
            score += 0.15
        weights_total += 0.15

        if quality.get("has_clipping", False):
            score += 0.1
        weights_total += 0.1

        voicing = vad.get("voicing_fraction", 0.5)
        if voicing < 0.2 or voicing > 0.95:
            score += 0.15
        weights_total += 0.15

        if weights_total > 0:
            score = score / weights_total

        return float(np.clip(score, 0.0, 1.0))
