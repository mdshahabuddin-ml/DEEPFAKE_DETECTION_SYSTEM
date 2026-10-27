"""
src/model.py
Model architectures for deepfake detection.
XceptionNet (Week 3), EfficientNet-B4 (Week 6), ViT (Week 6)
"""

import torch
import torch.nn as nn
import timm
from typing import Optional

from src.logger import get_logger

logger = get_logger(__name__)


class XceptionDetector(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model(
            "xception", pretrained=pretrained,
            num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )
        self._dropout_rate = dropout

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def get_features(self, x):
        return self.backbone(x)
    
    def enable_dropout(self):
        """Enable dropout layers for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        logger.debug("Dropout enabled for MC Dropout inference")
    
    def disable_dropout(self):
        """Disable dropout layers (set to eval mode)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        logger.debug("Dropout disabled")


class EfficientNetDetector(nn.Module):
    def __init__(self, pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained,
            num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )
        self._dropout_rate = dropout

    def forward(self, x):
        return self.classifier(self.backbone(x))
    
    def enable_dropout(self):
        """Enable dropout layers for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        logger.debug("Dropout enabled for MC Dropout inference")
    
    def disable_dropout(self):
        """Disable dropout layers (set to eval mode)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        logger.debug("Dropout disabled")


class ViTDetector(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained,
            num_classes=0, global_pool="token"
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )
        self._dropout_rate = dropout

    def forward(self, x):
        return self.classifier(self.backbone(x))
    
    def enable_dropout(self):
        """Enable dropout layers for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        logger.debug("Dropout enabled for MC Dropout inference")
    
    def disable_dropout(self):
        """Disable dropout layers (set to eval mode)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        logger.debug("Dropout disabled")


class EnsembleDetector(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights

    def forward(self, x):
        probs = []
        for model, w in zip(self.models, self.weights):
            prob = torch.sigmoid(model(x))
            probs.append(prob * w)
        return sum(probs)


MODELS = {
    "xception":     XceptionDetector,
    "efficientnet": EfficientNetDetector,
    "vit":          ViTDetector,
}


def build_model(model_name="xception", pretrained=True, dropout=0.5, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}")
    model = MODELS[model_name](pretrained=pretrained, dropout=dropout)
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Params: {total:,} | Device: {device}")
    return model


def load_checkpoint(model, checkpoint_path, device="cpu"):
    from src.checkpoint import safe_torch_load
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model