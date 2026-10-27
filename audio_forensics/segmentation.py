"""Audio segmentation for deepfake detection.

This module handles Voice Activity Detection (VAD) and speech segmentation
to divide audio into analyzable segments for deepfake detection.

Architecture:
    Full Audio -> VAD -> Speech Segments -> Segment Analysis

Each segment receives independent analysis for:
- Start time
- End time
- Speech type
- Duration
- Quality metrics
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import AudioForensicsConfig
from .schemas import SegmentResult, SpeechType, AudioStatus

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Custom exception for segmentation errors."""
    pass


@dataclass
class SpeechSegment:
    """A segment of detected speech."""
    start_time: float = 0.0
    end_time: float = 0.0
    start_sample: int = 0
    end_sample: int = 0
    duration: float = 0.0
    energy: float = 0.0
    speech_type: SpeechType = SpeechType.SPEECH
    confidence: float = 0.0


@dataclass
class SegmentationResult:
    """Result of audio segmentation."""
    segments: List[SpeechSegment] = field(default_factory=list)
    total_segments: int = 0
    total_speech_duration: float = 0.0
    total_silence_duration: float = 0.0
    speech_ratio: float = 0.0
    vad_threshold: float = 0.0
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AudioSegmenter:
    """Segment audio into speech regions for analysis.

    This class provides Voice Activity Detection (VAD) and segmentation
    capabilities for audio deepfake detection.

    Usage:
        config = AudioForensicsConfig()
        segmenter = AudioSegmenter(config)
        result = segmenter.segment(audio, sample_rate)
    """

    def __init__(self, config: Optional[AudioForensicsConfig] = None):
        """Initialize segmenter.

        Args:
            config: Audio forensics configuration.
        """
        self.config = config or AudioForensicsConfig()
        self._has_librosa = False
        self._has_scipy = False

        try:
            import librosa
            self._has_librosa = True
        except ImportError:
            pass

        try:
            import scipy.signal
            import scipy.ndimage
            self._has_scipy = True
        except ImportError:
            pass

    def segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        method: str = "energy",
    ) -> SegmentationResult:
        """Segment audio into speech regions.

        Args:
            audio: Audio signal as numpy array.
            sample_rate: Audio sample rate in Hz.
            method: Segmentation method ('energy', 'zcr', 'combined', 'librosa').

        Returns:
            SegmentationResult with detected segments.

        Raises:
            SegmentationError: If segmentation fails.
        """
        if audio is None or len(audio) == 0:
            raise SegmentationError("Empty audio signal")

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Ensure float32
        audio = audio.astype(np.float32)

        try:
            if method == "librosa":
                if self._has_librosa:
                    return self._segment_librosa(audio, sample_rate)
                else:
                    logger.warning("librosa not available, falling back to energy")
                    return self._segment_energy(audio, sample_rate)
            elif method == "zcr":
                return self._segment_zcr(audio, sample_rate)
            elif method == "combined":
                return self._segment_combined(audio, sample_rate)
            else:
                return self._segment_energy(audio, sample_rate)

        except Exception as e:
            raise SegmentationError(f"Segmentation failed: {e}") from e

    def _segment_energy(
        self, audio: np.ndarray, sample_rate: int
    ) -> SegmentationResult:
        """Segment based on energy levels."""
        frame_length = self.config.n_fft
        hop_length = self.config.hop_length

        # Calculate frame energies (guard: audio shorter than one frame)
        n_frames = max(1, 1 + (len(audio) - frame_length) // hop_length)
        energies = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            energies[i] = np.sqrt(np.mean(frame ** 2))

        # Apply threshold
        threshold = self.config.vad_energy_threshold
        is_speech = energies > threshold

        # Smooth to remove short gaps
        if self._has_scipy:
            from scipy.ndimage import binary_dilation, binary_erosion
            struct = np.ones(3)
            is_speech = binary_dilation(is_speech, struct)
            is_speech = binary_erosion(is_speech, struct)

        # Extract segments
        segments = self._extract_segments(
            is_speech, sample_rate, hop_length, energies
        )

        # Calculate statistics
        total_duration = len(audio) / sample_rate
        speech_duration = sum(s.duration for s in segments)
        silence_duration = total_duration - speech_duration

        return SegmentationResult(
            segments=segments,
            total_segments=len(segments),
            total_speech_duration=speech_duration,
            total_silence_duration=silence_duration,
            speech_ratio=speech_duration / total_duration if total_duration > 0 else 0,
            vad_threshold=threshold,
            method="energy",
            metadata={
                "frame_length": frame_length,
                "hop_length": hop_length,
                "n_frames": n_frames,
            },
        )

    def _segment_zcr(
        self, audio: np.ndarray, sample_rate: int
    ) -> SegmentationResult:
        """Segment based on zero crossing rate."""
        frame_length = self.config.n_fft
        hop_length = self.config.hop_length

        # Calculate ZCR
        n_frames = max(1, 1 + (len(audio) - frame_length) // hop_length)
        zcr = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            zcr[i] = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / frame_length

        # Speech typically has moderate ZCR (not too high, not too low)
        # Use adaptive threshold
        mean_zcr = np.mean(zcr)
        std_zcr = np.std(zcr)
        threshold_low = mean_zcr - std_zcr
        threshold_high = mean_zcr + std_zcr

        is_speech = (zcr > threshold_low) & (zcr < threshold_high)

        # Extract segments
        segments = self._extract_segments(
            is_speech, sample_rate, hop_length, zcr
        )

        total_duration = len(audio) / sample_rate
        speech_duration = sum(s.duration for s in segments)

        return SegmentationResult(
            segments=segments,
            total_segments=len(segments),
            total_speech_duration=speech_duration,
            total_silence_duration=total_duration - speech_duration,
            speech_ratio=speech_duration / total_duration if total_duration > 0 else 0,
            vad_threshold=mean_zcr,
            method="zcr",
        )

    def _segment_combined(
        self, audio: np.ndarray, sample_rate: int
    ) -> SegmentationResult:
        """Segment using combined energy and ZCR."""
        frame_length = self.config.n_fft
        hop_length = self.config.hop_length

        n_frames = max(1, 1 + (len(audio) - frame_length) // hop_length)
        energies = np.zeros(n_frames)
        zcr = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            energies[i] = np.sqrt(np.mean(frame ** 2))
            zcr[i] = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / frame_length

        # Normalize both features
        energies_norm = (energies - np.mean(energies)) / (np.std(energies) + 1e-10)
        zcr_norm = (zcr - np.mean(zcr)) / (np.std(zcr) + 1e-10)

        # Combine: speech has moderate energy and moderate ZCR
        energy_score = np.clip(energies_norm, 0, 1)
        zcr_score = 1 - np.abs(zcr_norm)

        combined_score = 0.7 * energy_score + 0.3 * zcr_score
        is_speech = combined_score > 0.3

        segments = self._extract_segments(
            is_speech, sample_rate, hop_length, combined_score
        )

        total_duration = len(audio) / sample_rate
        speech_duration = sum(s.duration for s in segments)

        return SegmentationResult(
            segments=segments,
            total_segments=len(segments),
            total_speech_duration=speech_duration,
            total_silence_duration=total_duration - speech_duration,
            speech_ratio=speech_duration / total_duration if total_duration > 0 else 0,
            vad_threshold=0.3,
            method="combined",
        )

    def _segment_librosa(
        self, audio: np.ndarray, sample_rate: int
    ) -> SegmentationResult:
        """Segment using librosa's onset detection."""
        import librosa

        # Get onset times
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sample_rate,
            hop_length=self.config.hop_length,
        )
        onset_times = librosa.frames_to_time(
            onset_frames, sr=sample_rate,
            hop_length=self.config.hop_length,
        )

        # Get RMS energy
        rms = librosa.feature.rms(
            y=audio, frame_length=self.config.n_fft,
            hop_length=self.config.hop_length,
        )[0]

        # Create segments from onset times
        segments = []
        for i, onset_time in enumerate(onset_times):
            if i < len(onset_times) - 1:
                end_time = onset_times[i + 1]
            else:
                end_time = len(audio) / sample_rate

            # Get average RMS for this segment
            start_frame = librosa.time_to_frames(
                onset_time, sr=sample_rate,
                hop_length=self.config.hop_length,
            )
            end_frame = librosa.time_to_frames(
                end_time, sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            if start_frame < len(rms) and end_frame <= len(rms):
                avg_rms = float(np.mean(rms[start_frame:end_frame]))
            else:
                avg_rms = 0.0

            segment = SpeechSegment(
                start_time=float(onset_time),
                end_time=float(end_time),
                start_sample=int(onset_time * sample_rate),
                end_sample=int(end_time * sample_rate),
                duration=float(end_time - onset_time),
                energy=avg_rms,
                speech_type=SpeechType.SPEECH if avg_rms > self.config.vad_energy_threshold else SpeechType.SILENCE,
                confidence=min(1.0, avg_rms * 10),
            )
            segments.append(segment)

        total_duration = len(audio) / sample_rate
        speech_duration = sum(s.duration for s in segments if s.speech_type == SpeechType.SPEECH)

        return SegmentationResult(
            segments=segments,
            total_segments=len(segments),
            total_speech_duration=speech_duration,
            total_silence_duration=total_duration - speech_duration,
            speech_ratio=speech_duration / total_duration if total_duration > 0 else 0,
            vad_threshold=self.config.vad_energy_threshold,
            method="librosa",
        )

    def _extract_segments(
        self,
        is_speech: np.ndarray,
        sample_rate: int,
        hop_length: int,
        feature_values: np.ndarray,
    ) -> List[SpeechSegment]:
        """Extract segments from binary speech indicator."""
        segments = []

        # Find contiguous speech regions
        in_segment = False
        start_frame = 0

        for i in range(len(is_speech)):
            if is_speech[i] and not in_segment:
                # Start of new segment
                in_segment = True
                start_frame = i
            elif not is_speech[i] and in_segment:
                # End of segment
                in_segment = False
                segment = self._create_segment(
                    start_frame, i, sample_rate, hop_length, feature_values
                )
                if segment.duration >= self.config.min_speech_duration:
                    segments.append(segment)

        # Handle last segment
        if in_segment:
            segment = self._create_segment(
                start_frame, len(is_speech), sample_rate, hop_length, feature_values
            )
            if segment.duration >= self.config.min_speech_duration:
                segments.append(segment)

        return segments

    def _create_segment(
        self,
        start_frame: int,
        end_frame: int,
        sample_rate: int,
        hop_length: int,
        feature_values: np.ndarray,
    ) -> SpeechSegment:
        """Create a SpeechSegment from frame indices."""
        start_time = start_frame * hop_length / sample_rate
        end_time = end_frame * hop_length / sample_rate
        duration = end_time - start_time

        # Average feature value in segment
        avg_energy = float(np.mean(feature_values[start_frame:end_frame]))

        # Determine speech type based on energy
        if avg_energy > self.config.vad_energy_threshold * 2:
            speech_type = SpeechType.SPEECH
        elif avg_energy > self.config.vad_energy_threshold:
            speech_type = SpeechType.SPEECH  # Could be soft speech
        elif avg_energy > self.config.vad_energy_threshold * 0.5:
            speech_type = SpeechType.NOISE
        else:
            speech_type = SpeechType.SILENCE

        # Confidence based on how far from threshold
        confidence = min(1.0, avg_energy / (self.config.vad_energy_threshold + 1e-10))

        return SpeechSegment(
            start_time=start_time,
            end_time=end_time,
            start_sample=int(start_time * sample_rate),
            end_sample=int(end_time * sample_rate),
            duration=duration,
            energy=avg_energy,
            speech_type=speech_type,
            confidence=confidence,
        )

    def segment_to_analyzable_chunks(
        self,
        audio: np.ndarray,
        sample_rate: int,
        max_chunk_duration: float = 10.0,
        overlap: float = 0.5,
    ) -> List[Tuple[np.ndarray, float, float]]:
        """Split audio into fixed-size chunks for analysis.

        Args:
            audio: Audio signal.
            sample_rate: Sample rate.
            max_chunk_duration: Maximum chunk duration in seconds.
            overlap: Overlap between chunks (0.0 to 1.0).

        Returns:
            List of (chunk, start_time, end_time) tuples.
        """
        chunk_samples = int(max_chunk_duration * sample_rate)
        # overlap=1.0 would give hop=0 (infinite loop) - clamp to 1 sample
        hop_samples = max(1, int(chunk_samples * (1 - overlap)))

        chunks = []
        start = 0

        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            # Skip very short chunks
            if len(chunk) < sample_rate * 0.5:  # Less than 0.5s
                break

            start_time = start / sample_rate
            end_time = end / sample_rate

            chunks.append((chunk, start_time, end_time))
            start += hop_samples

        return chunks

    def get_segment_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segment: SpeechSegment,
    ) -> np.ndarray:
        """Extract features for a specific segment."""
        start_sample = segment.start_sample
        end_sample = segment.end_sample

        # Ensure bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        segment_audio = audio[start_sample:end_sample]

        # Basic features
        features = [
            segment.duration,
            segment.energy,
            float(np.mean(np.abs(segment_audio))),
            float(np.std(segment_audio)),
            float(np.max(np.abs(segment_audio))),
            float(np.min(segment_audio)),
        ]

        return np.array(features, dtype=np.float32)


def get_segmenter(
    config: Optional[AudioForensicsConfig] = None,
) -> AudioSegmenter:
    """Factory function to get audio segmenter."""
    return AudioSegmenter(config)
