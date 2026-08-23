"""
Grad-CAM utilities for the deepfake detection model.

Save this file as: src/gradcam_utils.py

Works with any CNN backbone (including timm's xception) by automatically
finding the last Conv2d layer in the model, so it doesn't need to know
your exact model.py internals.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    Walks the model and returns the last nn.Conv2d module found.
    This works regardless of how build_model() wraps/names things internally.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model — cannot run Grad-CAM.")
    return last_conv


def generate_gradcam_overlay(model, input_tensor, rgb_face_float, device):
    """
    Args:
        model: your trained torch model (already .eval() and on `device`)
        input_tensor: preprocessed tensor, shape (1, C, H, W), already on `device`
        rgb_face_float: numpy array HxWx3, RGB, float32, values in [0, 1]
                         (same size as input_tensor's spatial dims, for overlay)
        device: "cpu" or "cuda"

    Returns:
        overlay: numpy uint8 image (HxWx3) with heatmap blended on top of the face
        raw_prob: sigmoid probability output for this input (float)
    """
    target_layer = find_last_conv_layer(model)

    # Model outputs a single logit (binary). Grad-CAM needs a scalar target.
    with torch.no_grad():
        raw_logit = model(input_tensor)
        raw_prob = torch.sigmoid(raw_logit).item()

    targets = [BinaryClassifierOutputTarget(1)]  # class "1" = fake, matches training label convention

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # first (only) image in batch

    # resize heatmap to match the face crop if needed
    if grayscale_cam.shape[:2] != rgb_face_float.shape[:2]:
        grayscale_cam = cv2.resize(
            grayscale_cam, (rgb_face_float.shape[1], rgb_face_float.shape[0])
        )

    overlay = show_cam_on_image(rgb_face_float, grayscale_cam, use_rgb=True)
    return overlay, raw_prob