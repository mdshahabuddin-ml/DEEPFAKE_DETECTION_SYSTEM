"""
src/benchmarking.py
Comprehensive benchmarking and evaluation for deepfake detection.

Provides:
- Standard metrics (AUC, accuracy, F1, precision, recall)
- ROC curve and precision-recall curve computation
- Cross-dataset evaluation
- Model comparison
- Confidence calibration metrics
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from src.config import RESULTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationMetrics:
    """Standard classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_samples: int


@dataclass
class CalibrationMetrics:
    """Prediction calibration metrics."""
    ece: float
    mce: float
    brier_score: float
    reliability_diagram_data: Dict[str, List[float]]


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    model_name: str
    dataset_name: str
    classification_metrics: ClassificationMetrics
    calibration_metrics: Optional[CalibrationMetrics]
    inference_stats: Dict[str, float]
    timestamp: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCalculator:
    """Compute standard classification metrics."""

    def __init__(self):
        pass

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> ClassificationMetrics:
        """
        Compute all classification metrics.

        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_prob: Predicted probabilities
            threshold: Classification threshold

        Returns:
            ClassificationMetrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_prob = np.asarray(y_prob)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        auc_roc = self._compute_auc_roc(y_true, y_prob)
        auc_pr = self._compute_auc_pr(y_true, y_prob)

        return ClassificationMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total_samples=len(y_true),
        )

    def _compute_auc_roc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute AUC-ROC using trapezoidal rule."""
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, y_prob))
        except ImportError:
            pass

        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_list = [0.0]
        fpr_list = [0.0]

        tp = 0
        fp = 0

        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        tpr_list.append(1.0)
        fpr_list.append(1.0)

        auc = np.trapz(tpr_list, fpr_list)
        return float(abs(auc))

    def _compute_auc_pr(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute AUC for precision-recall curve."""
        try:
            from sklearn.metrics import average_precision_score
            return float(average_precision_score(y_true, y_prob))
        except ImportError:
            pass

        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)

        if n_pos == 0:
            return 0.0

        tp = 0
        fp = 0
        precisions = []
        recalls = []

        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / n_pos

            precisions.append(precision)
            recalls.append(recall)

        precisions = np.array(precisions)
        recalls = np.array(recalls)

        sorted_idx = np.argsort(recalls)
        recalls = recalls[sorted_idx]
        precisions = precisions[sorted_idx]

        auc = np.trapz(precisions, recalls)
        return float(abs(auc))


class CalibrationMetricsCalculator:
    """Compute calibration metrics."""

    def compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """
        Compute calibration metrics.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for ECE

        Returns:
            CalibrationMetrics
        """
        ece = self._compute_ece(y_true, y_prob, n_bins)
        mce = self._compute_mce(y_true, y_prob, n_bins)
        brier = self._compute_brier_score(y_true, y_prob)
        reliability_data = self._compute_reliability_data(y_true, y_prob, n_bins)

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            reliability_diagram_data=reliability_data,
        )

    def _compute_ece(self, y_true, y_prob, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            bin_size = np.sum(mask)

            if bin_size > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                ece += bin_size * np.abs(bin_acc - bin_conf)

        return float(ece / len(y_true)) if len(y_true) > 0 else 0.0

    def _compute_mce(self, y_true, y_prob, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        max_calibration_error = 0.0

        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            bin_size = np.sum(mask)

            if bin_size > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                calibration_error = np.abs(bin_acc - bin_conf)
                max_calibration_error = max(max_calibration_error, calibration_error)

        return float(max_calibration_error)

    def _compute_brier_score(self, y_true, y_prob):
        return float(np.mean((y_true - y_prob) ** 2))

    def _compute_reliability_data(self, y_true, y_prob, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        accuracies = []
        confidences = []
        counts = []

        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            bin_size = np.sum(mask)

            if bin_size > 0:
                bin_center = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
                bin_centers.append(float(bin_center))
                accuracies.append(float(np.mean(y_true[mask])))
                confidences.append(float(np.mean(y_prob[mask])))
                counts.append(int(bin_size))

        return {
            "bin_centers": bin_centers,
            "accuracies": accuracies,
            "confidences": confidences,
            "counts": counts,
        }


class BenchmarkRunner:
    """Run comprehensive benchmarks on deepfake detection models."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else RESULTS_DIR / "benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_calc = MetricsCalculator()
        self.calibration_calc = CalibrationMetricsCalculator()

    def run_benchmark(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "unknown",
        dataset_name: str = "unknown",
        inference_times: Optional[List[float]] = None,
    ) -> BenchmarkResult:
        """
        Run a complete benchmark.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            dataset_name: Name of the dataset
            inference_times: Optional list of inference times

        Returns:
            BenchmarkResult
        """
        logger.info("Running benchmark: model=%s, dataset=%s, samples=%d",
                    model_name, dataset_name, len(y_true))

        threshold = 0.5
        y_pred = (y_prob >= threshold).astype(int)

        classification_metrics = self.metrics_calc.compute_metrics(
            y_true, y_pred, y_prob, threshold
        )

        calibration_metrics = self.calibration_calc.compute_calibration_metrics(
            y_true, y_prob
        )

        inference_stats = {}
        if inference_times:
            inference_times = np.array(inference_times)
            inference_stats = {
                "mean_inference_time": float(np.mean(inference_times)),
                "std_inference_time": float(np.std(inference_times)),
                "median_inference_time": float(np.median(inference_times)),
                "min_inference_time": float(np.min(inference_times)),
                "max_inference_time": float(np.max(inference_times)),
                "throughput_fps": float(1.0 / np.mean(inference_times)) if np.mean(inference_times) > 0 else 0.0,
            }

        result = BenchmarkResult(
            model_name=model_name,
            dataset_name=dataset_name,
            classification_metrics=classification_metrics,
            calibration_metrics=calibration_metrics,
            inference_stats=inference_stats,
            timestamp=datetime.now().isoformat(),
        )

        self._save_result(result)

        logger.info(
            "Benchmark complete: AUC=%.4f, Acc=%.4f, F1=%.4f, ECE=%.4f",
            classification_metrics.auc_roc,
            classification_metrics.accuracy,
            classification_metrics.f1_score,
            calibration_metrics.ece,
        )

        return result

    def compare_models(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, Any]:
        """
        Compare multiple model benchmark results.

        Args:
            results: List of BenchmarkResult objects

        Returns:
            Comparison summary
        """
        comparison = {
            "models": [],
            "best_auc": None,
            "best_accuracy": None,
            "best_f1": None,
        }

        best_auc = -1
        best_acc = -1
        best_f1 = -1

        for result in results:
            model_info = {
                "model_name": result.model_name,
                "dataset_name": result.dataset_name,
                "auc_roc": result.classification_metrics.auc_roc,
                "accuracy": result.classification_metrics.accuracy,
                "f1_score": result.classification_metrics.f1_score,
                "ece": result.calibration_metrics.ece if result.calibration_metrics else None,
            }
            comparison["models"].append(model_info)

            if result.classification_metrics.auc_roc > best_auc:
                best_auc = result.classification_metrics.auc_roc
                comparison["best_auc"] = result.model_name

            if result.classification_metrics.accuracy > best_acc:
                best_acc = result.classification_metrics.accuracy
                comparison["best_accuracy"] = result.model_name

            if result.classification_metrics.f1_score > best_f1:
                best_f1 = result.classification_metrics.f1_score
                comparison["best_f1"] = result.model_name

        return comparison

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{result.model_name}_{result.dataset_name}_{timestamp}.json"
        output_path = self.output_dir / filename

        data = {
            "model_name": result.model_name,
            "dataset_name": result.dataset_name,
            "classification_metrics": asdict(result.classification_metrics),
            "calibration_metrics": asdict(result.calibration_metrics) if result.calibration_metrics else None,
            "inference_stats": result.inference_stats,
            "timestamp": result.timestamp,
            "details": result.details,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Benchmark result saved: %s", output_path)
