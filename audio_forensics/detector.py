"""Audio deepfake detection model architecture.

This module provides modular audio deepfake detection models that are
completely independent from the existing visual detection system.

Architecture:
    Audio Features -> CNN/CRNN -> Synthetic/Manipulated Probability

The architecture is designed to be extensible:
- CNN: Simple convolutional classifier
- CRNN: Convolutional Recurrent Neural Network
- Transformer: Attention-based model (future)
- Pretrained: Use pretrained speech encoder (future)

Important:
    - Models output probabilities, not absolute decisions
    - Use terminology like "potentially manipulated" not "definitely fake"
    - Always include confidence and uncertainty estimates
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import AudioForensicsConfig
from .schemas import AudioStatus

logger = logging.getLogger(__name__)


class ModelArchitecture(Enum):
    """Supported model architectures."""
    CNN = "cnn"
    CRNN = "crnn"
    TRANSFORMER = "transformer"
    PRETRAINED = "pretrained"


class DetectorError(Exception):
    """Custom exception for detector errors."""
    pass


class ModelNotAvailableError(DetectorError):
    """Exception when model is not available."""
    pass


@dataclass
class DetectionResult:
    """Result of audio deepfake detection."""
    probability: float = 0.0
    prediction: AudioStatus = AudioStatus.UNKNOWN
    confidence: float = 0.0
    features_used: List[str] = None
    model_name: str = ""
    model_version: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.features_used is None:
            self.features_used = []
        if self.metadata is None:
            self.metadata = {}


class AudioDetector:
    """Modular audio deepfake detector.

    This class provides a flexible detection pipeline that can work
    with various model architectures and feature sets.

    Usage:
        config = AudioForensicsConfig()
        detector = AudioDetector(config)

        # Load model if available
        detector.load_model()

        # Detect
        result = detector.detect(features)
    """

    def __init__(self, config: Optional[AudioForensicsConfig] = None):
        """Initialize detector.

        Args:
            config: Audio forensics configuration.
        """
        self.config = config or AudioForensicsConfig()
        self._model = None
        self._model_loaded = False
        self._architecture = ModelArchitecture.CNN
        self._device = self._get_device()

    def _get_device(self) -> str:
        """Determine compute device."""
        if self.config.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return self.config.device

    @property
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    @property
    def architecture(self) -> ModelArchitecture:
        """Get current model architecture."""
        return self._architecture

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        architecture: ModelArchitecture = ModelArchitecture.CNN,
    ) -> bool:
        """Load detection model.

        Args:
            checkpoint_path: Path to model checkpoint.
            architecture: Model architecture to use.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        self._architecture = architecture

        # Use config checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = self.config.model_checkpoint

        if checkpoint_path is None:
            logger.warning("No model checkpoint specified")
            return False

        try:
            import torch

            if architecture == ModelArchitecture.CNN:
                self._model = self._build_cnn_model()
            elif architecture == ModelArchitecture.CRNN:
                self._model = self._build_crnn_model()
            else:
                logger.warning(f"Architecture {architecture} not yet implemented")
                return False

            # Load weights
            try:
                state_dict = torch.load(checkpoint_path, map_location=self._device)
                self._model.load_state_dict(state_dict)
                self._model.to(self._device)
                self._model.eval()
                self._model_loaded = True
                logger.info(f"Model loaded from {checkpoint_path}")
                return True
            except FileNotFoundError:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False

        except ImportError:
            logger.warning("PyTorch not available")
            return False

    def _build_cnn_model(self):
        """Build CNN-based audio classifier."""
        try:
            import torch
            import torch.nn as nn

            model = AudioCNNModel(
                n_mels=self.config.n_mels,
                n_classes=2,  # bonafide vs fake
            )
            return model

        except ImportError:
            raise DetectorError("PyTorch not available")

    def _build_crnn_model(self):
        """Build CRNN-based audio classifier."""
        try:
            import torch
            import torch.nn as nn

            model = AudioCRNNModel(
                n_mels=self.config.n_mels,
                n_classes=2,
            )
            return model

        except ImportError:
            raise DetectorError("PyTorch not available")

    def detect(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> DetectionResult:
        """Detect audio manipulation.

        Args:
            features: Extracted features (2D array: n_features x n_frames).
            feature_names: Names of features used.

        Returns:
            DetectionResult with prediction and confidence.
        """
        if features is None or features.size == 0:
            return DetectionResult(
                probability=0.0,
                prediction=AudioStatus.UNKNOWN,
                confidence=0.0,
                model_name=self._architecture.value,
            )

        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        try:
            if self._model_loaded and self._model is not None:
                return self._detect_with_model(features, feature_names)
            else:
                return self._detect_with_rules(features, feature_names)

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return DetectionResult(
                probability=0.0,
                prediction=AudioStatus.REQUIRES_REVIEW,
                confidence=0.0,
                features_used=feature_names or [],
                metadata={"error": str(e)},
            )

    def _detect_with_model(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> DetectionResult:
        """Detection using loaded neural network model."""
        try:
            import torch

            # Prepare input tensor
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            input_tensor = input_tensor.to(self._device)

            # Forward pass
            with torch.no_grad():
                output = self._model(input_tensor)
                probabilities = torch.softmax(output, dim=1)

            # Get fake probability
            fake_prob = float(probabilities[0, 1])

            # Determine prediction
            prediction = self._probability_to_prediction(fake_prob)

            # Calculate confidence
            confidence = abs(fake_prob - 0.5) * 2  # Distance from uncertainty

            return DetectionResult(
                probability=fake_prob,
                prediction=prediction,
                confidence=confidence,
                features_used=feature_names or [],
                model_name=self._architecture.value,
                metadata={"method": "neural_network"},
            )

        except Exception as e:
            logger.warning(f"Model inference failed: {e}")
            return self._detect_with_rules(features, feature_names)

    def _detect_with_rules(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> DetectionResult:
        """Rule-based detection when no model is available.

        This provides a baseline using statistical analysis of features.
        """
        if feature_names is None:
            feature_names = []

        # Simple statistical analysis
        # This is a baseline - real detection requires trained models

        # Calculate various anomaly scores
        scores = []

        # Check for unusual spectral characteristics
        if features.ndim == 2 and features.shape[0] > 10:
            # Log-Mel or similar spectrogram
            mean_val = np.mean(features)
            std_val = np.std(features)
            max_val = np.max(features)

            # Unusual energy distribution
            if std_val > mean_val * 2:
                scores.append(0.3)

            # Unusually high values
            if max_val > mean_val + 3 * std_val:
                scores.append(0.2)

        # Check for temporal anomalies
        if features.ndim == 2:
            # Compute frame-to-frame variation
            diffs = np.diff(features, axis=1)
            variation = np.mean(np.abs(diffs))

            if variation > 0.5:  # High variation
                scores.append(0.2)

        # Combine scores
        if scores:
            probability = np.mean(scores)
        else:
            probability = 0.0  # No signal

        # Determine prediction
        prediction = self._probability_to_prediction(probability)

        # Lower confidence for rule-based
        confidence = 0.3

        return DetectionResult(
            probability=probability,
            prediction=prediction,
            confidence=confidence,
            features_used=feature_names,
            model_name="rule_based",
            metadata={"method": "statistical_rules", "n_signals": len(scores)},
        )

    def _probability_to_prediction(self, probability: float) -> AudioStatus:
        """Convert probability to prediction status."""
        if probability >= self.config.high_confidence_threshold:
            return AudioStatus.POTENTIAL_MANIPULATION
        elif probability >= self.config.fake_threshold:
            return AudioStatus.SUSPICIOUS
        elif probability >= self.config.suspicious_threshold:
            return AudioStatus.LOW_SUSPICION
        else:
            return AudioStatus.NORMAL

    def predict_batch(
        self,
        features_batch: List[np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> List[DetectionResult]:
        """Predict on a batch of features.

        Args:
            features_batch: List of feature arrays.
            feature_names: Feature names.

        Returns:
            List of DetectionResult objects.
        """
        results = []
        for features in features_batch:
            result = self.detect(features, feature_names)
            results.append(result)
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "architecture": self._architecture.value,
            "is_loaded": self._model_loaded,
            "device": self._device,
            "n_mels": self.config.n_mels,
            "thresholds": {
                "fake": self.config.fake_threshold,
                "suspicious": self.config.suspicious_threshold,
                "high_confidence": self.config.high_confidence_threshold,
            },
        }


class AudioCNNModel:
    """Simple CNN model for audio deepfake detection.

    Architecture:
        Log-Mel -> Conv2d -> BN -> ReLU -> Pool ->
        Conv2d -> BN -> ReLU -> Pool ->
        Conv2d -> BN -> ReLU -> Pool ->
        FC -> ReLU -> Dropout -> FC -> Output
    """

    def __init__(self, n_mels: int = 128, n_classes: int = 2):
        """Initialize CNN model.

        Args:
            n_mels: Number of mel frequency bands.
            n_classes: Number of output classes.
        """
        try:
            import torch.nn as nn

            self.n_mels = n_mels
            self.n_classes = n_classes

            # Build model as sequential
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, n_classes),
            )

        except ImportError:
            raise DetectorError("PyTorch not available")

    def forward(self, x):
        """Forward pass."""
        import torch

        # Ensure correct input shape: (batch, 1, n_mels, n_frames)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = self.classifier(x)
        return x


class AudioCRNNModel:
    """CRNN model for audio deepfake detection.

    Architecture:
        Log-Mel -> Conv2d -> ReLU -> Pool ->
        RNN (GRU) -> FC -> Output
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_classes: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
    ):
        """Initialize CRNN model.

        Args:
            n_mels: Number of mel frequency bands.
            n_classes: Number of output classes.
            hidden_size: RNN hidden size.
            num_layers: Number of RNN layers.
        """
        try:
            import torch.nn as nn

            self.n_mels = n_mels
            self.n_classes = n_classes

            # CNN feature extractor
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

            # RNN
            cnn_output_size = 64 * (n_mels // 4)
            self.rnn = nn.GRU(
                input_size=cnn_output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, n_classes),
            )

        except ImportError:
            raise DetectorError("PyTorch not available")

    def forward(self, x):
        """Forward pass."""
        import torch

        # Ensure correct input shape
        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size = x.size(0)

        # CNN features
        x = self.cnn(x)

        # Reshape for RNN: (batch, channels, height, width) -> (batch, width, channels*height)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)

        # RNN
        x, _ = self.rnn(x)

        # Take last time step
        x = x[:, -1, :]

        # Classify
        x = self.classifier(x)

        return x


def get_detector(
    config: Optional[AudioForensicsConfig] = None,
) -> AudioDetector:
    """Factory function to get audio detector."""
    return AudioDetector(config)
