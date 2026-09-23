"""
src/simple_audio_pipeline.py
Simple 5-step audio deepfake detection pipeline.

Flow: Upload -> Preprocess -> Spectrogram -> Classify -> REAL / SYNTHETIC

This is a lightweight, self-contained pipeline that wraps existing components
without modifying any existing files.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

try:
    from src.audio_infer import (
        load_audio_detector, predict_from_array,
        DEFAULT_AUDIO_MODEL, DEFAULT_AUDIO_CHECKPOINT,
    )
    from src.audio_dataset import (
        SAMPLE_RATE, DURATION, waveform_to_input,
        load_waveform, fix_waveform_length,
    )
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.audio_infer import (
        load_audio_detector, predict_from_array,
        DEFAULT_AUDIO_MODEL, DEFAULT_AUDIO_CHECKPOINT,
    )
    from src.audio_dataset import (
        SAMPLE_RATE, DURATION, waveform_to_input,
        load_waveform, fix_waveform_length,
    )


class SimpleAudioPipeline:
    """Simple 5-step audio deepfake detection pipeline.

    Flow: Upload -> Preprocess -> Spectrogram -> Classify -> REAL / SYNTHETIC

    Usage:
        pipeline = SimpleAudioPipeline()
        result = pipeline.analyze("path/to/audio.wav")
        print(result["prediction"])  # "REAL" or "SYNTHETIC"
    """

    def __init__(
        self,
        model_name: str = DEFAULT_AUDIO_MODEL,
        checkpoint_path: str = DEFAULT_AUDIO_CHECKPOINT,
        sample_rate: int = SAMPLE_RATE,
        duration: float = DURATION,
        fake_threshold: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.fake_threshold = fake_threshold
        self._model_name = model_name
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._device = None

    def _ensure_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model, self._device = load_audio_detector(
                self._model_name, self._checkpoint_path
            )

    def analyze(self, audio_input: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Process audio through the complete 5-step pipeline.

        Args:
            audio_input: File path (str/Path) or numpy waveform array.

        Returns:
            Dict with prediction, confidence, fake_probability,
            processing_time, spectrogram, duration, sample_rate.
        """
        start_time = time.time()

        # Step 1: Handle audio input (upload)
        if isinstance(audio_input, (str, Path)):
            audio_array = self._load_audio(audio_input)
            file_path = str(audio_input)
        else:
            audio_array = audio_input.astype(np.float32)
            file_path = "<array>"

        audio_duration = len(audio_array) / self.sample_rate

        # Step 2: Audio preprocessing
        preprocessed = self._preprocess(audio_array)

        # Step 3: Spectrogram generation
        spectrogram = self._generate_spectrogram(preprocessed)

        # Step 4: Basic audio classifier
        self._ensure_model()
        fake_prob = self._classify(spectrogram)

        # Step 5: REAL / SYNTHETIC decision
        prediction = "SYNTHETIC" if fake_prob >= self.fake_threshold else "REAL"
        confidence = abs(fake_prob - 0.5) * 2.0

        processing_time = time.time() - start_time

        return {
            "prediction": prediction,
            "confidence": float(min(confidence, 1.0)),
            "fake_probability": float(fake_prob),
            "processing_time": float(processing_time),
            "spectrogram": spectrogram,
            "duration": float(audio_duration),
            "sample_rate": self.sample_rate,
            "file_path": file_path,
        }

    def _load_audio(self, file_path: Union[str, Path]) -> np.ndarray:
        """Step 1: Load audio file."""
        return load_waveform(str(file_path), sr=self.sample_rate)

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """Step 2: Normalize and fix length."""
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        return fix_waveform_length(audio, sr=self.sample_rate, duration=self.duration)

    def _generate_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Step 3: Generate log-mel spectrogram."""
        return waveform_to_input(audio, sr=self.sample_rate)

    def _classify(self, spectrogram: np.ndarray) -> float:
        """Step 4: Run through audio classifier."""
        import torch
        tensor = torch.from_numpy(spectrogram).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logit = self._model(tensor)
            fake_prob = torch.sigmoid(logit).item()
        return fake_prob

    def get_info(self) -> Dict[str, Any]:
        """Return pipeline configuration info."""
        return {
            "model_name": self._model_name,
            "checkpoint_path": self._checkpoint_path,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "fake_threshold": self.fake_threshold,
            "model_loaded": self._model is not None,
        }
