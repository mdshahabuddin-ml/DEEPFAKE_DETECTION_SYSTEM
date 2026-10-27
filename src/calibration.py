"""
src/calibration.py
Prediction calibration and uncertainty estimation for the Deepfake Detection System.

Implements:
- Monte Carlo Dropout for uncertainty estimation
- Temperature scaling for confidence calibration
- Platt scaling for probability calibration
- Expected Calibration Error (ECE) computation
- Reliability diagram data generation
"""

from __future__ import annotations

import threading

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.special import expit

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationResult:
    """Results from prediction calibration."""
    original_prob: float
    calibrated_prob: float
    uncertainty: float
    confidence_interval: Tuple[float, float]
    n_forward_passes: int
    temperature: Optional[float] = None
    ece: Optional[float] = None


@dataclass
class UncertaintyResult:
    """Results from uncertainty estimation."""
    mean_prediction: float
    std_prediction: float
    uncertainty: float
    confidence_interval: Tuple[float, float]
    predictive_entropy: float
    mutual_information: float
    n_forward_passes: int


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Uses multiple forward passes with dropout enabled to estimate
    predictive uncertainty.
    """
    
    def __init__(self, n_forward_passes: int = 30, dropout_rate: float = 0.5):
        """
        Initialize MC Dropout uncertainty estimator.

        Args:
            n_forward_passes: Number of stochastic forward passes
            dropout_rate: Dropout rate to use during inference
        """
        self.n_forward_passes = n_forward_passes
        self.dropout_rate = dropout_rate
        # Serializes model.train()/eval() toggling — concurrent calls on a
        # shared model would otherwise race on the dropout-layer mode flags.
        self._lock = threading.Lock()
    
    def _enable_dropout(self, model: nn.Module) -> None:
        """Enable dropout layers during inference."""
        for module in model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.train()
    
    def _disable_dropout(self, model: nn.Module) -> None:
        """Disable dropout layers (set to eval mode)."""
        for module in model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.eval()
    
    def estimate_uncertainty(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        device: str = "cpu"
    ) -> UncertaintyResult:
        """
        Estimate prediction uncertainty using MC Dropout.
        
        Args:
            model: Neural network model
            input_tensor: Input tensor for prediction
            device: Device to run inference on
            
        Returns:
            UncertaintyResult with uncertainty metrics
        """
        model.eval()
        predictions = []

        with self._lock:
            self._enable_dropout(model)
            try:
                with torch.no_grad():
                    for _ in range(self.n_forward_passes):
                        logits = model(input_tensor)
                        prob = torch.sigmoid(logits).item()
                        predictions.append(prob)
            finally:
                self._disable_dropout(model)
        
        predictions = np.array(predictions)
        
        mean_pred = float(np.mean(predictions))
        std_pred = float(np.std(predictions))
        
        # Predictive uncertainty (total uncertainty)
        uncertainty = std_pred
        
        # Confidence interval (95%)
        ci_lower = float(np.percentile(predictions, 2.5))
        ci_upper = float(np.percentile(predictions, 97.5))
        
        # Predictive entropy
        p_mean = mean_pred
        entropy = -(p_mean * np.log(p_mean + 1e-10) + 
                   (1 - p_mean) * np.log(1 - p_mean + 1e-10))
        
        # Mutual information (epistemic uncertainty)
        expected_entropy = -np.mean(
            predictions * np.log(predictions + 1e-10) +
            (1 - predictions) * np.log(1 - predictions + 1e-10)
        )
        mutual_info = entropy - expected_entropy
        
        logger.debug(
            "MC Dropout uncertainty: mean=%.4f, std=%.4f, entropy=%.4f",
            mean_pred, std_pred, entropy
        )
        
        return UncertaintyResult(
            mean_prediction=mean_pred,
            std_prediction=std_pred,
            uncertainty=uncertainty,
            confidence_interval=(ci_lower, ci_upper),
            predictive_entropy=float(entropy),
            mutual_information=float(mutual_info),
            n_forward_passes=self.n_forward_passes,
        )


class TemperatureScaling:
    """
    Temperature scaling for confidence calibration.
    
    Learns a single temperature parameter to scale logits
    before softmax/sigmoid.
    """
    
    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted: bool = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """
        Fit temperature scaling on validation data.
        
        Args:
            logits: Model logits (not probabilities)
            labels: Binary labels (0 or 1)
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations
            
        Returns:
            Learned temperature parameter
        """
        def loss_fn(temp):
            scaled_logits = logits / temp
            probs = expit(scaled_logits)
            # Negative log-likelihood
            nll = -np.mean(
                labels * np.log(probs + 1e-10) +
                (1 - labels) * np.log(1 - probs + 1e-10)
            )
            return nll
        
        result = minimize(
            loss_fn,
            x0=1.0,
            method='L-BFGS-B',
            bounds=[(0.1, 10.0)],
            options={'maxiter': max_iter, 'lr': lr}
        )
        
        self.temperature = float(result.x[0])
        self.is_fitted = True
        
        logger.info("Temperature scaling fitted: T=%.4f", self.temperature)
        return self.temperature
    
    def scale(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model logits
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Temperature not fitted, using T=1.0")
        
        scaled_logits = logits / self.temperature
        return expit(scaled_logits)


class PlattScaling:
    """
    Platt scaling for binary classification calibration.
    
    Fits a logistic regression on model outputs to calibrate
    probabilities.
    """
    
    def __init__(self):
        self.a: float = 1.0
        self.b: float = 0.0
        self.is_fitted: bool = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> Tuple[float, float]:
        """
        Fit Platt scaling on validation data.
        
        Args:
            logits: Model logits
            labels: Binary labels (0 or 1)
            lr: Learning rate
            max_iter: Maximum iterations
            
        Returns:
            Tuple of (a, b) parameters
        """
        def loss_fn(params):
            a, b = params
            scaled = a * logits + b
            probs = expit(scaled)
            nll = -np.mean(
                labels * np.log(probs + 1e-10) +
                (1 - labels) * np.log(1 - probs + 1e-10)
            )
            return nll
        
        result = minimize(
            loss_fn,
            x0=[1.0, 0.0],
            method='L-BFGS-B',
            options={'maxiter': max_iter}
        )
        
        self.a, self.b = result.x
        self.is_fitted = True
        
        logger.info("Platt scaling fitted: a=%.4f, b=%.4f", self.a, self.b)
        return self.a, self.b
    
    def scale(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to logits.
        
        Args:
            logits: Raw model logits
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Platt scaling not fitted, using identity")
        
        scaled = self.a * logits + self.b
        return expit(scaled)


def compute_ece(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        predictions: Model predictions (probabilities)
        labels: True binary labels
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
        bin_size = np.sum(mask)
        
        if bin_size > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(predictions[mask])
            ece += bin_size * np.abs(bin_acc - bin_conf)
    
    ece /= len(predictions)
    
    logger.debug("ECE computed: %.4f (n_bins=%d)", ece, n_bins)
    return float(ece)


def compute_reliability_diagram_data(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, List[float]]:
    """
    Generate data for reliability diagram.
    
    Args:
        predictions: Model predictions
        labels: True labels
        n_bins: Number of bins
        
    Returns:
        Dictionary with bin_centers, accuracies, confidences, counts
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    accuracies = []
    confidences = []
    counts = []
    
    for i in range(n_bins):
        mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
        bin_size = np.sum(mask)
        
        if bin_size > 0:
            bin_center = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
            bin_centers.append(float(bin_center))
            accuracies.append(float(np.mean(labels[mask])))
            confidences.append(float(np.mean(predictions[mask])))
            counts.append(int(bin_size))
    
    return {
        "bin_centers": bin_centers,
        "accuracies": accuracies,
        "confidences": confidences,
        "counts": counts,
    }


class PredictionCalibrator:
    """
    Combined prediction calibrator with multiple methods.
    
    Provides a unified interface for calibrating predictions
    and estimating uncertainty.
    """
    
    def __init__(
        self,
        method: str = "temperature",
        n_mc_passes: int = 30
    ):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('temperature', 'platt', 'mc_dropout')
            n_mc_passes: Number of MC Dropout forward passes
        """
        self.method = method
        self.mc_dropout = MCDropoutUncertainty(n_mc_passes)
        self.temperature_scaling = TemperatureScaling()
        self.platt_scaling = PlattScaling()
        
        logger.info("PredictionCalibrator initialized: method=%s", method)
    
    def calibrate_single(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        device: str = "cpu"
    ) -> CalibrationResult:
        """
        Calibrate a single prediction with uncertainty.
        
        Args:
            model: Neural network model
            input_tensor: Input tensor
            device: Device for inference
            
        Returns:
            CalibrationResult with calibrated probability and uncertainty
        """
        # Get base prediction
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            original_prob = torch.sigmoid(logits).item()
        
        # Get uncertainty estimate
        uncertainty_result = self.mc_dropout.estimate_uncertainty(
            model, input_tensor, device
        )
        
        # Apply calibration based on method
        if self.method == "temperature" and self.temperature_scaling.is_fitted:
            logit = np.log(original_prob / (1 - original_prob + 1e-10) + 1e-10)
            calibrated_prob = float(self.temperature_scaling.scale(np.array([logit]))[0])
        elif self.method == "platt" and self.platt_scaling.is_fitted:
            logit = np.log(original_prob / (1 - original_prob + 1e-10) + 1e-10)
            calibrated_prob = float(self.platt_scaling.scale(np.array([logit]))[0])
        else:
            calibrated_prob = uncertainty_result.mean_prediction
        
        # Clamp to valid range
        calibrated_prob = float(np.clip(calibrated_prob, 0.0, 1.0))
        
        return CalibrationResult(
            original_prob=original_prob,
            calibrated_prob=calibrated_prob,
            uncertainty=uncertainty_result.uncertainty,
            confidence_interval=uncertainty_result.confidence_interval,
            n_forward_passes=uncertainty_result.n_forward_passes,
            temperature=self.temperature_scaling.temperature if self.temperature_scaling.is_fitted else None,
        )
    
    def calibrate_batch(
        self,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Calibrate a batch of predictions.
        
        Args:
            predictions: Array of model predictions
            labels: Optional true labels for fitting calibration
            
        Returns:
            Tuple of (calibrated_predictions, ece_score)
        """
        if labels is not None and not self.temperature_scaling.is_fitted:
            # Convert probabilities to logits for fitting
            logits = np.log(predictions / (1 - predictions + 1e-10) + 1e-10)
            self.temperature_scaling.fit(logits, labels)
        
        if self.temperature_scaling.is_fitted:
            logits = np.log(predictions / (1 - predictions + 1e-10) + 1e-10)
            calibrated = self.temperature_scaling.scale(logits)
        else:
            calibrated = predictions
        
        # Compute ECE if labels provided
        ece = None
        if labels is not None:
            ece = compute_ece(calibrated, labels)
        
        return calibrated, ece
