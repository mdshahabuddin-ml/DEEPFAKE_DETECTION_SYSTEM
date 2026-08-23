"""
src/audio_model.py
Model architectures for audio deepfake (spoof) detection.

Operates on single-channel log-mel spectrograms (1, N_MELS, N_FRAMES) and
outputs a single logit (fake = high). Mirrors the interface of src/model.py:
build_audio_model() + load_checkpoint().
"""

import torch
import torch.nn as nn
import timm


class SpectrogramCNN(nn.Module):
    """A lightweight from-scratch CNN — no pretrained download needed."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128),
            block(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class TimmAudioDetector(nn.Module):
    """A timm CNN backbone adapted to 1-channel spectrogram input."""

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained,
            num_classes=0, global_pool="avg", in_chans=1,
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


AUDIO_MODELS = {
    "cnn": SpectrogramCNN,
    "efficientnet": TimmAudioDetector,   # efficientnet_b0 backbone
}


def build_audio_model(model_name: str = "efficientnet", pretrained: bool = True,
                      dropout: float = 0.3, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name not in AUDIO_MODELS:
        raise ValueError(
            f"Unknown audio model: {model_name}. Choose from {list(AUDIO_MODELS)}"
        )
    if model_name == "cnn":
        model = SpectrogramCNN(dropout=dropout)
    else:
        model = TimmAudioDetector(pretrained=pretrained, dropout=dropout)
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Audio model: {model_name} | Params: {total:,} | Device: {device}")
    return model


def load_checkpoint(model, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded audio checkpoint: {checkpoint_path}")
    return model
