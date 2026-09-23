"""Intelligent frame sampling for video analysis."""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .config import TemporalConfig

logger = logging.getLogger(__name__)


class FrameSampler:
    """Sample frames from video for temporal analysis."""

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()

    def sample_frames(
        self, video_path: str, method: str = "uniform"
    ) -> List[np.ndarray]:
        """Sample frames from video.

        Args:
            video_path: Path to video file.
            method: Sampling method ('uniform', 'keyframe', 'adaptive').

        Returns:
            List of sampled frames as numpy arrays.
        """
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not available")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        # Calculate frame indices to sample
        indices = self._calculate_sample_indices(
            total_frames, fps, method
        )

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize
                frame = cv2.resize(frame, self.config.frame_size)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        logger.info(f"Sampled {len(frames)} frames from {video_path}")
        return frames

    def sample_from_array(
        self,
        video_frames: np.ndarray,
        method: str = "uniform",
    ) -> List[np.ndarray]:
        """Sample frames from numpy array.

        Args:
            video_frames: Video frames array (N, H, W, C).
            method: Sampling method.

        Returns:
            List of sampled frames.
        """
        n_frames = len(video_frames)
        indices = self._calculate_sample_indices(n_frames, 30.0, method)
        return [video_frames[i] for i in indices if i < n_frames]

    def _calculate_sample_indices(
        self, total_frames: int, fps: float, method: str
    ) -> List[int]:
        """Calculate frame indices to sample."""
        if method == "uniform":
            return self._uniform_sampling(total_frames)
        elif method == "keyframe":
            return self._keyframe_sampling(total_frames)
        elif method == "adaptive":
            return self._adaptive_sampling(total_frames)
        else:
            return self._uniform_sampling(total_frames)

    def _uniform_sampling(self, total_frames: int) -> List[int]:
        """Uniform sampling across video."""
        n_samples = min(self.config.max_frames, total_frames)
        n_samples = max(n_samples, self.config.min_frames)

        indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
        return indices.tolist()

    def _keyframe_sampling(self, total_frames: int) -> List[int]:
        """Sample at regular intervals."""
        interval = max(1, int(fps / self.config.target_fps))
        indices = list(range(0, total_frames, interval))
        return indices[:self.config.max_frames]

    def _adaptive_sampling(self, total_frames: int) -> List[int]:
        """Adaptive sampling based on scene changes."""
        # Simple adaptive: sample more densely at start and end
        n_samples = min(self.config.max_frames, total_frames)
        first_quarter = n_samples // 4
        last_quarter = n_samples // 4
        middle = n_samples - first_quarter - last_quarter

        indices = []
        if first_quarter > 0:
            indices.extend(
                np.linspace(0, total_frames // 4, first_quarter, dtype=int).tolist()
            )
        if middle > 0:
            indices.extend(
                np.linspace(total_frames // 4, 3 * total_frames // 4, middle, dtype=int).tolist()
            )
        if last_quarter > 0:
            indices.extend(
                np.linspace(3 * total_frames // 4, total_frames - 1, last_quarter, dtype=int).tolist()
            )

        return sorted(set(indices))[:self.config.max_frames]
