"""
src/explainability.py
Advanced explainability methods for deepfake detection.

Provides:
- Grad-CAM (existing, enhanced)
- Guided Backpropagation
- Layer-wise Relevance Propagation (LRP)
- Integrated Gradients
- SHAP-based explanations
- Feature visualization
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExplanationResult:
    """Result of an explainability analysis."""
    method: str
    heatmap: np.ndarray
    original_rgb: np.ndarray
    overlay: np.ndarray
    fake_probability: float
    prediction: str
    confidence: float
    feature_importance: Optional[Dict[str, float]] = None
    analysis_time: float = 0.0


class GuidedBackpropagation:
    """Guided Backpropagation for input gradient visualization."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate guided backpropagation attribution map.

        Args:
            input_tensor: Input tensor (1, C, H, W)

        Returns:
            Attribution map (H, W)
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        prob = torch.sigmoid(output)

        self.model.zero_grad()
        prob.backward()

        if self.gradients is None:
            return np.zeros(input_tensor.shape[2:])

        attribution = self.gradients[0, 0].cpu().numpy()
        attribution = np.maximum(attribution, 0)
        if np.max(attribution) > 0:
            attribution = attribution / np.max(attribution)

        return attribution


class IntegratedGradients:
    """Integrated Gradients attribution method."""

    def __init__(self, model: nn.Module, device: str = "cpu", n_steps: int = 50):
        self.model = model
        self.device = device
        self.n_steps = n_steps

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate integrated gradients attribution.

        Args:
            input_tensor: Input tensor (1, C, H, W)

        Returns:
            Attribution map (H, W)
        """
        self.model.eval()

        baseline = torch.zeros_like(input_tensor).to(self.device)

        scaled_inputs = [
            baseline + (float(i) / self.n_steps) * (input_tensor - baseline)
            for i in range(self.n_steps + 1)
        ]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad_(True)

        outputs = self.model(scaled_inputs)
        probs = torch.sigmoid(outputs)[:, 0]

        self.model.zero_grad()
        probs.sum().backward()

        grads = scaled_inputs.grad.detach()
        grads = (grads[:-1] + grads[1:]) / 2.0

        avg_grads = grads.mean(dim=0, keepdim=True)
        attribution = (input_tensor - baseline) * avg_grads
        attribution = attribution[0, 0].cpu().numpy()

        attribution = np.abs(attribution)
        if np.max(attribution) > 0:
            attribution = attribution / np.max(attribution)

        return attribution


class LayerwiseRelevancePropagation:
    """Simplified Layer-wise Relevance Propagation."""

    def __init__(self, model: nn.Module, device: str = "cpu", epsilon: float = 1e-7):
        self.model = model
        self.device = device
        self.epsilon = epsilon

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate LRP attribution.

        Args:
            input_tensor: Input tensor (1, C, H, W)

        Returns:
            Relevance map (H, W)
        """
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()

        activation_maps = []
        hooks = []

        def hook_fn(module, input, output):
            activation_maps.append(output.detach())

        for module in self.model.modules():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            self.model(input_tensor)

        for hook in hooks:
            hook.remove()

        if not activation_maps:
            return np.zeros(input_tensor.shape[2:])

        relevance = activation_maps[-1][0].cpu().numpy()

        for i in range(len(activation_maps) - 2, -1, -1):
            act = activation_maps[i][0].cpu().numpy()
            if relevance.shape != act.shape:
                relevance = cv2.resize(
                    relevance.T, (act.shape[1], act.shape[0])
                ).T
            relevance = relevance * act
            relevance = relevance / (np.abs(relevance).sum() + self.epsilon)

        relevance = np.sum(relevance, axis=0)
        relevance = np.maximum(relevance, 0)

        if np.max(relevance) > 0:
            relevance = relevance / np.max(relevance)

        return relevance


class ExplainabilityEngine:
    """
    Unified explainability engine combining multiple methods.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.guided_bp = GuidedBackpropagation(model, device)
        self.integrated_grads = IntegratedGradients(model, device)
        self.lrp = LayerwiseRelevancePropagation(model, device)

    def explain(
        self,
        input_tensor: torch.Tensor,
        method: str = "gradcam",
        face_bgr: Optional[np.ndarray] = None,
    ) -> ExplanationResult:
        """
        Generate explanation using specified method.

        Args:
            input_tensor: Input tensor (1, C, H, W)
            method: Explanation method ('gradcam', 'guided', 'integrated', 'lrp')
            face_bgr: Optional original face for overlay

        Returns:
            ExplanationResult
        """
        start_time = time.time()

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = float(torch.sigmoid(output).item())

        prediction = "FAKE" if prob >= 0.5 else "REAL"
        confidence = prob if prob >= 0.5 else 1 - prob

        if face_bgr is not None:
            original_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            original_rgb = cv2.resize(original_rgb, (input_tensor.shape[3], input_tensor.shape[2]))
        else:
            original_rgb = np.transpose(input_tensor[0].cpu().numpy(), (1, 2, 0))
            original_rgb = ((original_rgb * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).clip(0, 255).astype(np.uint8)

        if method == "guided":
            heatmap = self.guided_bp.generate(input_tensor.clone())
        elif method == "integrated":
            heatmap = self.integrated_grads.generate(input_tensor.clone())
        elif method == "lrp":
            heatmap = self.lrp.generate(input_tensor.clone())
        else:
            heatmap = self._generate_gradcam(input_tensor)

        overlay = self._create_overlay(original_rgb, heatmap)

        analysis_time = time.time() - start_time

        return ExplanationResult(
            method=method,
            heatmap=heatmap,
            original_rgb=original_rgb,
            overlay=overlay,
            fake_probability=prob,
            prediction=prediction,
            confidence=confidence,
            analysis_time=analysis_time,
        )

    def _generate_gradcam(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

            target_layer = None
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module

            if target_layer is None:
                return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

            cam = GradCAM(model=self.model, target_layers=[target_layer])
            targets = [BinaryClassifierOutputTarget(1)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            return grayscale_cam

        except ImportError:
            logger.warning("pytorch_grad_cam not available, using input gradient")
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            prob = torch.sigmoid(output)
            self.model.zero_grad()
            prob.backward()

            if input_tensor.grad is not None:
                heatmap = input_tensor.grad[0, 0].abs().cpu().numpy()
                if np.max(heatmap) > 0:
                    heatmap = heatmap / np.max(heatmap)
                return heatmap
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

    def _create_overlay(
        self, original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """Create overlay of heatmap on original image."""
        heatmap_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = (
            original_rgb.astype(np.float32) * (1 - alpha) +
            heatmap_color.astype(np.float32) * alpha
        ).astype(np.uint8)

        return overlay

    def explain_batch(
        self,
        input_tensor: torch.Tensor,
        methods: List[str] = None,
    ) -> Dict[str, ExplanationResult]:
        """
        Generate explanations using multiple methods.

        Args:
            input_tensor: Input tensor
            methods: List of methods to use

        Returns:
            Dictionary of method -> ExplanationResult
        """
        if methods is None:
            methods = ["gradcam", "guided", "integrated"]

        results = {}
        for method in methods:
            try:
                results[method] = self.explain(input_tensor, method=method)
            except Exception as e:
                logger.warning("Explanation method %s failed: %s", method, e)

        return results


def create_feature_comparison(
    results: Dict[str, ExplanationResult],
) -> np.ndarray:
    """
    Create a side-by-side comparison of different explanation methods.

    Args:
        results: Dictionary of method -> ExplanationResult

    Returns:
        Combined image array
    """
    if not results:
        return np.array([])

    images = []
    for method, result in results.items():
        images.append(result.overlay)

    if not images:
        return np.array([])

    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    padded = []
    for img in images:
        h, w = img.shape[:2]
        pad_h = max_h - h
        pad_w = max_w - w
        padded_img = np.pad(
            img,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant',
            constant_values=255,
        )
        padded.append(padded_img)

    combined = np.hstack(padded)
    return combined
