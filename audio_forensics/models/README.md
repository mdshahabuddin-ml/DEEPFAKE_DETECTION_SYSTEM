# Audio Forensics Models

This directory stores audio deepfake detection model checkpoints.

## Supported Models

- **Audio-EfficientNet**: CNN-based audio classifier
- **Audio-CRNN**: Convolutional Recurrent Neural Network
- **Speech-Encoder**: Pretrained speech representation model

## Model Format

Models should be saved as PyTorch `.pth` or `.pt` files.

## Usage

```python
from audio_forensics.config import AudioForensicsConfig

config = AudioForensicsConfig(model_checkpoint="audio_forensics/models/best_model.pth")
```

## Adding New Models

1. Train or download the model
2. Save checkpoint to this directory
3. Update `config.py` with the model path
4. Add model loading code in `detector.py`
