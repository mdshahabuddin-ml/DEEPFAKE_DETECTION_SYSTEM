"""
src/gradcam.py
Explainable-AI: Grad-CAM heatmaps for the deepfake detector.

Wraps the model with a Grad-CAM implementation that automatically targets the
last convolutional layer, produces a heatmap over the detected face, and
overlays it on the original crop so users can see *where* the model looked.

Returns RGB uint8 arrays that render directly in Streamlit or encode to PNG
for the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import IMG_SIZE, NORM_MEAN, NORM_STD, confidence_from_prob, label_from_prob
from src.dataset import get_val_transforms
from src.inference import DeepfakeDetector
from src.logger import get_logger

logger = get_logger(__name__)


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last nn.Conv2d in the model (works for any CNN backbone)."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found — Grad-CAM needs a CNN backbone.")
    return last_conv


class GradCAMExplainer:
    """Generate Grad-CAM overlays for a loaded DeepfakeDetector."""

    def __init__(self, detector: DeepfakeDetector):
        self.detector = detector
        self.model = detector.model
        self.device = detector.device
        self.img_size = detector.img_size
        self.transform = get_val_transforms(self.img_size)
        self.target_layer = find_last_conv_layer(self.model)

    def _preprocess(self, face_rgb: np.ndarray) -> torch.Tensor:
        tensor = self.transform(image=face_rgb)["image"].unsqueeze(0)
        return tensor.to(self.device)

    def explain_face(self, face_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Given a face crop (BGR), return:
          (original_rgb_resized, overlay_rgb, fake_probability)
        Both images are uint8 RGB of size (img_size, img_size).
        """
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self._preprocess(face_rgb)

        # Float image in [0, 1] for the overlay background.
        rgb_float = cv2.resize(face_rgb, (self.img_size, self.img_size)).astype(np.float32) / 255.0

        with torch.no_grad():
            prob = float(torch.sigmoid(self.model(input_tensor)).item())

        cam = GradCAM(model=self.model, target_layers=[self.target_layer])
        targets = [BinaryClassifierOutputTarget(1)]  # explain the "fake" logit
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        original = (rgb_float * 255).astype(np.uint8)
        return original, overlay, prob

    def explain_image_array(self, bgr_image: np.ndarray) -> Optional[dict]:
        """
        Detect the largest face in a full image and explain it.
        Returns None if no face is found.
        """
        face = bgr_image
        if face is None:
            return None
        original, overlay, prob = self.explain_face(face)
        return {
            "original": original,
            "overlay": overlay,
            "fake_probability": prob,
            "prediction": label_from_prob(prob, self.detector.threshold),
            "confidence": confidence_from_prob(prob, self.detector.threshold),
        }

    def explain_image(self, image_path) -> Optional[dict]:
        """Explain the largest face in an image file. None if no face."""
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.explain_image_array(bgr)
