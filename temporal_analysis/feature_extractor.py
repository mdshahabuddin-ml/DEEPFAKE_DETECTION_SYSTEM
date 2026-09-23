"""Feature extraction for temporal video analysis."""

import logging
from typing import List, Optional

import numpy as np

from .config import TemporalConfig

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor:
    """Extract features from video frames for temporal analysis."""

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        self._model = None

    def extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract features from frames.

        Args:
            frames: List of video frames.

        Returns:
            Feature array of shape (n_frames, n_features).
        """
        if not frames:
            return np.array([])

        # Try model-based extraction
        try:
            return self._extract_with_model(frames)
        except Exception as e:
            logger.warning(f"Model extraction failed: {e}")
            return self._extract_simple_features(frames)

    def _extract_with_model(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract features using neural network model."""
        try:
            import torch
            import torchvision.transforms as transforms
            import torchvision.models as models

            # Load pretrained model
            if self._model is None:
                model = models.efficientnet_b0(pretrained=True)
                model.eval()
                # Remove classifier
                self._model = torch.nn.Sequential(*list(model.children())[:-1])

            # Preprocess
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.config.frame_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            features_list = []
            for frame in frames:
                tensor = transform(frame).unsqueeze(0)
                with torch.no_grad():
                    feat = self._model(tensor)
                features_list.append(feat.squeeze().numpy())

            return np.array(features_list)

        except ImportError:
            raise RuntimeError("PyTorch not available")

    def _extract_simple_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract simple features without neural network."""
        features = []
        for frame in frames:
            feat = self._compute_frame_features(frame)
            features.append(feat)
        return np.array(features)

    def _compute_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """Compute simple features for a single frame."""
        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame

        # Basic statistics
        features = [
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray),
            np.median(gray),
        ]

        # Histogram features (8 bins)
        hist, _ = np.histogram(gray, bins=8, range=(0, 255))
        hist = hist.astype(float) / (hist.sum() + 1e-10)
        features.extend(hist.tolist())

        # Spatial features
        h, w = gray.shape
        if h > 0 and w > 0:
            # Quarter statistics
            q1 = np.mean(gray[:h//2, :w//2])
            q2 = np.mean(gray[:h//2, w//2:])
            q3 = np.mean(gray[h//2:, :w//2])
            q4 = np.mean(gray[h//2:, w//2:])
            features.extend([q1, q2, q3, q4])

        return np.array(features, dtype=np.float32)

    def extract_differences(
        self, features: np.ndarray
    ) -> np.ndarray:
        """Extract frame-to-frame differences.

        Args:
            features: Feature array (n_frames, n_features).

        Returns:
            Differences array (n_frames-1, n_features).
        """
        if len(features) < 2:
            return np.array([])

        return np.diff(features, axis=0)
