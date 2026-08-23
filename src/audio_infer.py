"""
src/audio_infer.py
Shared inference helpers for audio deepfake detection.

Used by app.py (Audio tab) and src/realtime_audio.py so they behave identically.
"""

from pathlib import Path

import numpy as np
import torch

try:
    from src.audio_model import build_audio_model, load_checkpoint
    from src.audio_dataset import (
        SAMPLE_RATE, load_waveform, waveform_to_input,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.audio_model import build_audio_model, load_checkpoint
    from src.audio_dataset import (
        SAMPLE_RATE, load_waveform, waveform_to_input,
    )

# Keep in sync with app.py / audio_train.py defaults
DEFAULT_AUDIO_MODEL = "efficientnet"
DEFAULT_AUDIO_CHECKPOINT = "models/best_audio_efficientnet.pth"


def load_audio_detector(model_name=DEFAULT_AUDIO_MODEL,
                        checkpoint_path=DEFAULT_AUDIO_CHECKPOINT, device=None):
    """Build the audio model and load trained weights, ready for inference."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_audio_model(model_name, pretrained=False, device=device)
    model = load_checkpoint(model, str(checkpoint_path), device=device)
    model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def predict_from_array(y: np.ndarray, model, device, sr: int = SAMPLE_RATE) -> float:
    """Return the fake probability in [0, 1] for a raw waveform array."""
    arr = waveform_to_input(y.astype(np.float32), sr=sr)          # (1, N_MELS, N_FRAMES)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)        # add batch dim
    return torch.sigmoid(model(tensor)).item()


@torch.no_grad()
def predict_from_file(path, model, device, sr: int = SAMPLE_RATE) -> float:
    """Return the fake probability in [0, 1] for an audio file on disk."""
    y = load_waveform(path, sr=sr)
    return predict_from_array(y, model, device, sr=sr)
