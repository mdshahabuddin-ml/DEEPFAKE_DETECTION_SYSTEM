"""
src/audio_video_sync.py
Audio-video synchronization analysis for lip-sync deepfake detection.

Provides:
- Lip movement extraction and tracking
- Audio-visual correlation analysis
- Cross-modal consistency scoring
- Lip-sync anomaly detection
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LipRegion:
    """Detected lip region in a frame."""
    frame_idx: int
    bbox: Tuple[int, int, int, int]
    landmarks: Optional[np.ndarray] = None
    mouth_openness: float = 0.0
    width: int = 0
    height: int = 0


@dataclass
class SyncFrame:
    """Synchronization data for a single frame."""
    frame_idx: int
    timestamp: float
    lip_region: Optional[LipRegion] = None
    audio_energy: float = 0.0
    visual_energy: float = 0.0
    correlation: float = 0.0


@dataclass
class SyncAnalysisResult:
    """Complete audio-video synchronization analysis result."""
    video_path: str
    total_frames: int
    analyzed_frames: int
    sync_score: float
    sync_timeline: List[SyncFrame]
    lip_sync_anomalies: List[Dict[str, Any]]
    overall_consistency: float
    audio_energy_profile: np.ndarray
    visual_energy_profile: np.ndarray
    analysis_time: float = 0.0


class LipDetector:
    """Detect and track lips in video frames."""

    def __init__(self):
        """Initialize lip detector using OpenCV Haar cascades."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.mouth_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

    def detect_lips(
        self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[LipRegion]:
        """
        Detect lips in a frame.

        Args:
            frame: Input frame (BGR)
            face_bbox: Optional face bounding box

        Returns:
            LipRegion if detected, None otherwise
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        if face_bbox is None:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) == 0:
                return None
            face_bbox = max(
                faces, key=lambda f: f[2] * f[3]
            )

        x, y, fw, fh = face_bbox
        mouth_region = gray[y + fh // 2 : y + fh, x : x + fw]

        if mouth_region.size == 0:
            return None

        mouth_resized = cv2.resize(mouth_region, (100, 50))

        edges = cv2.Canny(mouth_resized, 50, 150)
        edge_density = np.mean(edges) / 255.0

        mouth_openness = min(1.0, edge_density * 3)

        lx = x + fw // 4
        ly = y + int(fh * 0.65)
        lw = fw // 2
        lh = fh // 4

        return LipRegion(
            frame_idx=0,
            bbox=(lx, ly, lx + lw, ly + lh),
            mouth_openness=mouth_openness,
            width=lw,
            height=lh,
        )


class AudioVisualCorrelator:
    """Correlate audio and visual signals."""

    def __init__(self, sr: int = 16000, fps: float = 30.0):
        """
        Initialize correlator.

        Args:
            sr: Audio sample rate
            fps: Video frame rate
        """
        self.sr = sr
        self.fps = fps

    def compute_audio_energy(
        self, audio: np.ndarray, n_frames: int
    ) -> np.ndarray:
        """
        Compute audio energy profile aligned with video frames.

        Args:
            audio: Audio waveform
            n_frames: Number of video frames

        Returns:
            Audio energy per frame
        """
        samples_per_frame = int(self.sr / self.fps)
        energies = []

        for i in range(n_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            if end > len(audio):
                break
            frame_audio = audio[start:end]
            energy = float(np.sqrt(np.mean(frame_audio ** 2)))
            energies.append(energy)

        if not energies:
            return np.array([])

        energies = np.array(energies)
        if np.max(energies) > 0:
            energies = energies / np.max(energies)

        return energies

    def compute_visual_energy(
        self, lip_regions: List[Optional[LipRegion]]
    ) -> np.ndarray:
        """
        Compute visual energy profile from lip movements.

        Args:
            lip_regions: List of detected lip regions per frame

        Returns:
            Visual energy per frame
        """
        energies = []
        for lip in lip_regions:
            if lip is not None:
                energies.append(lip.mouth_openness)
            else:
                energies.append(0.0)

        energies = np.array(energies)
        if np.max(energies) > 0:
            energies = energies / np.max(energies)

        return energies

    def compute_correlation(
        self, audio_energy: np.ndarray, visual_energy: np.ndarray
    ) -> float:
        """
        Compute correlation between audio and visual signals.

        Args:
            audio_energy: Audio energy profile
            visual_energy: Visual energy profile

        Returns:
            Correlation score between 0 and 1
        """
        min_len = min(len(audio_energy), len(visual_energy))
        if min_len < 2:
            return 0.0

        audio = audio_energy[:min_len]
        visual = visual_energy[:min_len]

        if np.std(audio) < 1e-6 or np.std(visual) < 1e-6:
            return 0.0

        correlation = np.corrcoef(audio, visual)[0, 1]
        correlation = (correlation + 1) / 2

        return float(correlation)

    def detect_sync_anomalies(
        self,
        audio_energy: np.ndarray,
        visual_energy: np.ndarray,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Detect synchronization anomalies.

        Args:
            audio_energy: Audio energy profile
            visual_energy: Visual energy profile
            threshold: Anomaly threshold

        Returns:
            List of detected anomalies
        """
        anomalies = []
        min_len = min(len(audio_energy), len(visual_energy))

        if min_len < 3:
            return anomalies

        audio = audio_energy[:min_len]
        visual = visual_energy[:min_len]

        window_size = max(3, min_len // 10)

        for i in range(0, min_len - window_size):
            audio_window = audio[i : i + window_size]
            visual_window = visual[i : i + window_size]

            if np.std(audio_window) > 0.1 and np.std(visual_window) > 0.1:
                corr = np.corrcoef(audio_window, visual_window)[0, 1]

                if corr < -0.3:
                    anomalies.append({
                        "frame_start": i,
                        "frame_end": i + window_size,
                        "correlation": float(corr),
                        "type": "desynchronization",
                    })

            audio_energy_sum = np.sum(audio_window)
            visual_energy_sum = np.sum(visual_window)

            if audio_energy_sum > 0.5 and visual_energy_sum < 0.1:
                anomalies.append({
                    "frame_start": i,
                    "frame_end": i + window_size,
                    "type": "audio_without_visual",
                    "audio_energy": float(audio_energy_sum),
                    "visual_energy": float(visual_energy_sum),
                })
            elif visual_energy_sum > 0.5 and audio_energy_sum < 0.1:
                anomalies.append({
                    "frame_start": i,
                    "frame_end": i + window_size,
                    "type": "visual_without_audio",
                    "audio_energy": float(audio_energy_sum),
                    "visual_energy": float(visual_energy_sum),
                })

        return anomalies


class LipSyncAnalyzer:
    """
    Comprehensive lip-sync analysis for deepfake detection.
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.lip_detector = LipDetector()
        self.correlator = AudioVisualCorrelator(sr)

    def analyze_video(
        self,
        video_path: str,
        audio: np.ndarray,
        max_frames: int = 100,
        fps_override: Optional[float] = None,
    ) -> SyncAnalysisResult:
        """
        Perform lip-sync analysis on a video.

        Args:
            video_path: Path to the video file
            audio: Audio waveform (mono, at self.sr)
            max_frames: Maximum frames to analyze
            fps_override: Optional FPS override

        Returns:
            SyncAnalysisResult
        """
        start_time = time.time()
        logger.info("Starting lip-sync analysis: %s", video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.correlator.fps = fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        lip_regions = []
        sync_frames = []
        frame_count = 0

        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_count / fps
                lip_region = self.lip_detector.detect_lips(frame)
                lip_regions.append(lip_region)

                sync_frame = SyncFrame(
                    frame_idx=frame_count,
                    timestamp=timestamp,
                    lip_region=lip_region,
                )
                sync_frames.append(sync_frame)

                frame_count += 1
        finally:
            cap.release()

        audio_energy = self.correlator.compute_audio_energy(audio, len(lip_regions))
        visual_energy = self.correlator.compute_visual_energy(lip_regions)

        min_len = min(len(audio_energy), len(visual_energy))
        overall_correlation = 0.0
        if min_len >= 2:
            overall_correlation = self.correlator.compute_correlation(
                audio_energy[:min_len], visual_energy[:min_len]
            )

        lip_sync_anomalies = self.correlator.detect_sync_anomalies(
            audio_energy, visual_energy
        )

        for sf in sync_frames[:min_len]:
            sf.audio_energy = float(audio_energy[sf.frame_idx]) if sf.frame_idx < len(audio_energy) else 0.0
            sf.visual_energy = float(visual_energy[sf.frame_idx]) if sf.frame_idx < len(visual_energy) else 0.0

        anomaly_ratio = len(lip_sync_anomalies) / max(1, len(sync_frames))
        sync_score = overall_correlation * (1.0 - min(1.0, anomaly_ratio * 2))

        analysis_time = time.time() - start_time

        result = SyncAnalysisResult(
            video_path=video_path,
            total_frames=total_frames,
            analyzed_frames=len(sync_frames),
            sync_score=sync_score,
            sync_timeline=sync_frames,
            lip_sync_anomalies=lip_sync_anomalies,
            overall_consistency=overall_correlation,
            audio_energy_profile=audio_energy,
            visual_energy_profile=visual_energy,
            analysis_time=analysis_time,
        )

        logger.info(
            "Lip-sync analysis complete: score=%.4f, anomalies=%d (%.3f seconds)",
            sync_score, len(lip_sync_anomalies), analysis_time
        )

        return result


def analyze_lip_sync(
    video_path: str,
    audio: np.ndarray,
    sr: int = 16000,
    max_frames: int = 100,
) -> SyncAnalysisResult:
    """Convenience function for lip-sync analysis."""
    analyzer = LipSyncAnalyzer(sr=sr)
    return analyzer.analyze_video(video_path, audio, max_frames)
