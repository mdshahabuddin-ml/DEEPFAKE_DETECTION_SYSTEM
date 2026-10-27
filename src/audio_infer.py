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

    path = Path(checkpoint_path)
    if not path.is_absolute():
        from src.config import PROJECT_ROOT
        path = PROJECT_ROOT / path

    from src.checkpoint import CheckpointDownloadError, ensure_audio_checkpoint
    try:
        resolved = ensure_audio_checkpoint(path)
    except CheckpointDownloadError as exc:
        raise FileNotFoundError(
            f"Audio checkpoint download failed:\n{exc}\n\n"
            f"Expected file: {path}\n"
            "Fix DEEPFAKE_AUDIO_CHECKPOINT_URL / DEEPFAKE_AUDIO_HF_REPO "
            "(env or Streamlit secrets), or place the .pth manually."
        ) from exc
    if resolved is None:
        raise FileNotFoundError(
            f"Audio checkpoint not found:\n{path}\n\n"
            "Place a trained .pth at that path, or configure an auto-download "
            "source (DEEPFAKE_AUDIO_CHECKPOINT_URL or DEEPFAKE_AUDIO_HF_REPO). "
            "On Streamlit Community Cloud, add it under Settings > Secrets."
        )

    model = build_audio_model(model_name, pretrained=False, device=device)
    model = load_checkpoint(model, str(resolved), device=device)
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
