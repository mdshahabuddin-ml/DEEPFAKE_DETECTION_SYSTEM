"""
src/adversarial.py
Adversarial attack methods for robustness testing.

Provides:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- DeepFool
- Adversarial patch generation
- Defense evaluation
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AdversarialResult:
    """Result of an adversarial attack."""
    original_image: np.ndarray
    adversarial_image: np.ndarray
    perturbation: np.ndarray
    original_prediction: float
    adversarial_prediction: float
    prediction_shift: float
    attack_success: bool
    l2_distance: float
    linf_distance: float
    attack_method: str
    attack_time: float = 0.0


class FGSMAttack:
    """Fast Gradient Sign Method attack."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    def attack(
        self,
        image: np.ndarray,
        epsilon: float = 0.03,
        target_label: Optional[int] = None,
    ) -> AdversarialResult:
        """
        Apply FGSM attack.

        Args:
            image: Input image (BGR uint8)
            epsilon: Perturbation magnitude
            target_label: Optional target label (None for untargeted)

        Returns:
            AdversarialResult
        """
        start_time = time.time()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (299, 299))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        tensor.requires_grad_(True)

        self.model.eval()
        output = self.model(tensor)
        original_prob = torch.sigmoid(output).item()

        if target_label is None:
            target = 1 if original_prob < 0.5 else 0
        else:
            target = target_label

        target_tensor = torch.tensor([target], dtype=torch.float32).to(self.device)
        loss = F.binary_cross_entropy_with_logits(output, target_tensor)

        self.model.zero_grad()
        loss.backward()

        perturbation = epsilon * tensor.grad.sign()
        adversarial_tensor = tensor + perturbation

        with torch.no_grad():
            adv_output = self.model(adversarial_tensor)
            adversarial_prob = torch.sigmoid(adv_output).item()

        adv_image_numpy = adversarial_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
        adv_image_numpy = adv_image_numpy * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        adv_image_numpy = np.clip(adv_image_numpy * 255, 0, 255).astype(np.uint8)
        adv_image_bgr = cv2.cvtColor(adv_image_numpy, cv2.COLOR_RGB2BGR)

        perturbation_numpy = perturbation.detach().cpu().numpy()[0].transpose(1, 2, 0)
        perturbation_visual = (perturbation_numpy * 255 / epsilon).astype(np.uint8)

        attack_success = (original_prob >= 0.5) != (adversarial_prob >= 0.5)

        l2_dist = float(np.linalg.norm((adversarial_tensor - tensor).detach().cpu().numpy()))
        linf_dist = float(np.max(np.abs((adversarial_tensor - tensor).detach().cpu().numpy())))

        attack_time = time.time() - start_time

        return AdversarialResult(
            original_image=image,
            adversarial_image=adv_image_bgr,
            perturbation=perturbation_visual,
            original_prediction=original_prob,
            adversarial_prediction=adversarial_prob,
            prediction_shift=abs(adversarial_prob - original_prob),
            attack_success=attack_success,
            l2_distance=l2_dist,
            linf_distance=linf_dist,
            attack_method="FGSM",
            attack_time=attack_time,
        )


class PGDAttack:
    """Projected Gradient Descent attack."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    def attack(
        self,
        image: np.ndarray,
        epsilon: float = 0.03,
        alpha: float = 0.007,
        num_iter: int = 10,
        random_start: bool = True,
    ) -> AdversarialResult:
        """
        Apply PGD attack.

        Args:
            image: Input image (BGR uint8)
            epsilon: Maximum perturbation
            alpha: Step size per iteration
            num_iter: Number of iterations
            random_start: Whether to start from random point

        Returns:
            AdversarialResult
        """
        start_time = time.time()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (299, 299))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        original_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            output = self.model(original_tensor)
            original_prob = torch.sigmoid(output).item()

        target = 1 if original_prob < 0.5 else 0

        adv_tensor = original_tensor.clone().detach()

        if random_start:
            noise = torch.empty_like(adv_tensor).uniform_(-epsilon, epsilon)
            adv_tensor = adv_tensor + noise

        for _ in range(num_iter):
            adv_tensor.requires_grad_(True)

            output = self.model(adv_tensor)
            target_tensor = torch.tensor([target], dtype=torch.float32).to(self.device)
            loss = F.binary_cross_entropy_with_logits(output, target_tensor)

            self.model.zero_grad()
            loss.backward()

            adv_tensor = adv_tensor + alpha * adv_tensor.grad.sign()
            delta = torch.clamp(adv_tensor - original_tensor, -epsilon, epsilon)
            adv_tensor = torch.clamp(original_tensor + delta, -1, 1).detach()

        with torch.no_grad():
            adv_output = self.model(adv_tensor)
            adversarial_prob = torch.sigmoid(adv_output).item()

        adv_image_numpy = adv_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
        adv_image_numpy = adv_image_numpy * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        adv_image_numpy = np.clip(adv_image_numpy * 255, 0, 255).astype(np.uint8)
        adv_image_bgr = cv2.cvtColor(adv_image_numpy, cv2.COLOR_RGB2BGR)

        perturbation = (adv_tensor - original_tensor).detach().cpu().numpy()[0].transpose(1, 2, 0)
        perturbation_visual = ((perturbation + epsilon) / (2 * epsilon) * 255).astype(np.uint8)

        attack_success = (original_prob >= 0.5) != (adversarial_prob >= 0.5)
        l2_dist = float(np.linalg.norm((adv_tensor - original_tensor).detach().cpu().numpy()))
        linf_dist = float(np.max(np.abs((adv_tensor - original_tensor).detach().cpu().numpy())))

        attack_time = time.time() - start_time

        return AdversarialResult(
            original_image=image,
            adversarial_image=adv_image_bgr,
            perturbation=perturbation_visual,
            original_prediction=original_prob,
            adversarial_prediction=adversarial_prob,
            prediction_shift=abs(adversarial_prob - original_prob),
            attack_success=attack_success,
            l2_distance=l2_dist,
            linf_distance=linf_dist,
            attack_method="PGD",
            attack_time=attack_time,
        )


class DeepFoolAttack:
    """DeepFool attack for minimal perturbation."""

    def __init__(self, model: nn.Module, device: str = "cpu", max_iter: int = 50):
        self.model = model
        self.device = device
        self.max_iter = max_iter

    def attack(self, image: np.ndarray) -> AdversarialResult:
        """
        Apply DeepFool attack.

        Args:
            image: Input image (BGR uint8)

        Returns:
            AdversarialResult
        """
        start_time = time.time()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (299, 299))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        self.model.eval()

        with torch.no_grad():
            output = self.model(tensor)
            original_prob = torch.sigmoid(output).item()

        original_label = 1 if original_prob >= 0.5 else 0

        perturbation = torch.zeros_like(tensor)
        x = tensor.clone().detach()

        for _ in range(self.max_iter):
            x.requires_grad_(True)
            output = self.model(x)
            prob = torch.sigmoid(output)

            if original_label == 1:
                loss = prob
            else:
                loss = 1 - prob

            self.model.zero_grad()
            loss.backward()

            grad = x.grad
            if grad is None:
                break

            if original_label == 1:
                pert = (prob.item() + 1e-8) * grad / (grad.norm() + 1e-8)
            else:
                pert = (1 - prob.item() + 1e-8) * grad / (grad.norm() + 1e-8)

            perturbation = perturbation + pert

            x = tensor + (1 + 0.02) * perturbation
            x = torch.clamp(x, -3, 3).detach()

            with torch.no_grad():
                new_output = self.model(x)
                new_prob = torch.sigmoid(new_output).item()
                new_label = 1 if new_prob >= 0.5 else 0

                if new_label != original_label:
                    break

        with torch.no_grad():
            adv_output = self.model(x)
            adversarial_prob = torch.sigmoid(adv_output).item()

        adv_image_numpy = x.detach().cpu().numpy()[0].transpose(1, 2, 0)
        adv_image_numpy = adv_image_numpy * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        adv_image_numpy = np.clip(adv_image_numpy * 255, 0, 255).astype(np.uint8)
        adv_image_bgr = cv2.cvtColor(adv_image_numpy, cv2.COLOR_RGB2BGR)

        pert_numpy = perturbation.detach().cpu().numpy()[0].transpose(1, 2, 0)
        pert_visual = ((pert_numpy - pert_numpy.min()) / (pert_numpy.max() - pert_numpy.min() + 1e-8) * 255).astype(np.uint8)

        attack_success = (original_prob >= 0.5) != (adversarial_prob >= 0.5)
        l2_dist = float(np.linalg.norm(perturbation.detach().cpu().numpy()))
        linf_dist = float(np.max(np.abs(perturbation.detach().cpu().numpy())))

        attack_time = time.time() - start_time

        return AdversarialResult(
            original_image=image,
            adversarial_image=adv_image_bgr,
            perturbation=pert_visual,
            original_prediction=original_prob,
            adversarial_prediction=adversarial_prob,
            prediction_shift=abs(adversarial_prob - original_prob),
            attack_success=attack_success,
            l2_distance=l2_dist,
            linf_distance=linf_dist,
            attack_method="DeepFool",
            attack_time=attack_time,
        )


class AdversarialRobustnessEvaluator:
    """Evaluate model robustness against adversarial attacks."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.fgsm = FGSMAttack(model, device)
        self.pgd = PGDAttack(model, device)
        self.deepfool = DeepFoolAttack(model, device)

    def evaluate_single(
        self,
        image: np.ndarray,
        epsilon: float = 0.03,
    ) -> dict:
        """
        Evaluate robustness for a single image.

        Args:
            image: Input image (BGR uint8)
            epsilon: Perturbation magnitude

        Returns:
            Dictionary of attack results
        """
        results = {}

        try:
            fgsm_result = self.fgsm.attack(image, epsilon=epsilon)
            results["fgsm"] = fgsm_result
        except Exception as e:
            logger.warning("FGSM attack failed: %s", e)

        try:
            pgd_result = self.pgd.attack(image, epsilon=epsilon)
            results["pgd"] = pgd_result
        except Exception as e:
            logger.warning("PGD attack failed: %s", e)

        try:
            deepfool_result = self.deepfool.attack(image)
            results["deepfool"] = deepfool_result
        except Exception as e:
            logger.warning("DeepFool attack failed: %s", e)

        return results

    def evaluate_dataset(
        self,
        images: List[np.ndarray],
        epsilon: float = 0.03,
    ) -> dict:
        """
        Evaluate robustness on a dataset.

        Args:
            images: List of input images
            epsilon: Perturbation magnitude

        Returns:
            Aggregate robustness metrics
        """
        all_results = {
            "fgsm": {"successes": 0, "total": 0, "mean_shift": []},
            "pgd": {"successes": 0, "total": 0, "mean_shift": []},
            "deepfool": {"successes": 0, "total": 0, "mean_shift": []},
        }

        for i, image in enumerate(images):
            logger.info("Evaluating image %d/%d", i + 1, len(images))

            results = self.evaluate_single(image, epsilon)

            for method, result in results.items():
                all_results[method]["successes"] += int(result.attack_success)
                all_results[method]["total"] += 1
                all_results[method]["mean_shift"].append(result.prediction_shift)

        summary = {}
        for method, stats in all_results.items():
            if stats["total"] > 0:
                summary[method] = {
                    "attack_success_rate": stats["successes"] / stats["total"],
                    "mean_prediction_shift": float(np.mean(stats["mean_shift"])) if stats["mean_shift"] else 0.0,
                    "std_prediction_shift": float(np.std(stats["mean_shift"])) if stats["mean_shift"] else 0.0,
                }
            else:
                summary[method] = {
                    "attack_success_rate": 0.0,
                    "mean_prediction_shift": 0.0,
                    "std_prediction_shift": 0.0,
                }

        return summary
