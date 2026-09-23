"""
src/model_evaluation.py
Comprehensive model evaluation: Image, Video, Audio, Fusion models.
Metrics: ROC-AUC, PR-AUC, F1, EER, Calibration, Robustness.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, roc_curve,
        f1_score, precision_score, recall_score, accuracy_score,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.config import RESULTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    eer: float
    eer_threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


@dataclass
class CalibrationMetrics:
    ece: float
    mce: float
    brier_score: float
    reliability_data: Dict[str, List[float]]


@dataclass
class RobustnessMetrics:
    clean_accuracy: float
    corrupted_accuracies: Dict[str, float]
    mean_corruption_accuracy: float
    accuracy_drop: float
    adversarial_success_rate: float
    mean_prediction_shift: float


@dataclass
class ModelEvaluationResult:
    model_name: str
    model_type: str
    dataset_name: str
    classification: ClassificationMetrics
    calibration: CalibrationMetrics
    robustness: Optional[RobustnessMetrics]
    inference_time_mean: float
    inference_time_std: float
    throughput_fps: float
    evaluation_timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparativeEvaluation:
    models: List[ModelEvaluationResult]
    best_by_roc_auc: str
    best_by_f1: str
    best_by_eer: str
    best_by_calibration: str
    summary_table: List[Dict[str, Any]]
    evaluation_timestamp: str


class MetricsCalculator:

    @staticmethod
    def compute(y_true, y_pred, threshold=0.5):
        if not SKLEARN_AVAILABLE:
            return MetricsCalculator._manual(y_true, y_pred, threshold)
        y_bin = (y_pred >= threshold).astype(int)
        acc = float(accuracy_score(y_true, y_bin))
        prec = float(precision_score(y_true, y_bin, zero_division=0))
        rec = float(recall_score(y_true, y_bin, zero_division=0))
        f1 = float(f1_score(y_true, y_bin, zero_division=0))
        try:
            roc = float(roc_auc_score(y_true, y_pred))
        except ValueError:
            roc = 0.5
        try:
            pr = float(average_precision_score(y_true, y_pred))
        except ValueError:
            pr = 0.5
        eer, ert = MetricsCalculator._eer(y_true, y_pred)
        cm = confusion_matrix(y_true, y_bin)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        return ClassificationMetrics(
            accuracy=acc, precision=prec, recall=rec, f1=f1,
            roc_auc=roc, pr_auc=pr, eer=eer, eer_threshold=ert,
            true_positives=int(tp), false_positives=int(fp),
            true_negatives=int(tn), false_negatives=int(fn),
        )

    @staticmethod
    def _eer(y_true, y_pred):
        if not SKLEARN_AVAILABLE:
            return MetricsCalculator._eer_manual(y_true, y_pred)
        fpr, tpr, thr = roc_curve(y_true, y_pred)
        fnr = 1 - tpr
        i = np.nanargmin(np.abs(fpr - fnr))
        return float(fpr[i]), float(thr[i])

    @staticmethod
    def _eer_manual(y_true, y_pred):
        best, eer, ert = 1.0, 0.5, 0.5
        for t in np.linspace(0, 1, 100):
            yb = (y_pred >= t).astype(int)
            tp = int(np.sum((yb == 1) & (y_true == 1)))
            fp = int(np.sum((yb == 1) & (y_true == 0)))
            fn = int(np.sum((yb == 0) & (y_true == 1)))
            tn = int(np.sum((yb == 0) & (y_true == 0)))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            d = abs(fpr - fnr)
            if d < best:
                best, eer, ert = d, (fpr + fnr) / 2, t
        return float(eer), float(ert)

    @staticmethod
    def _manual(y_true, y_pred, threshold):
        yb = (y_pred >= threshold).astype(int)
        tp = int(np.sum((yb == 1) & (y_true == 1)))
        fp = int(np.sum((yb == 1) & (y_true == 0)))
        fn = int(np.sum((yb == 0) & (y_true == 1)))
        tn = int(np.sum((yb == 0) & (y_true == 0)))
        n = tp + fp + fn + tn
        acc = (tp + tn) / n if n else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1v = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        eer, ert = MetricsCalculator._eer_manual(y_true, y_pred)
        return ClassificationMetrics(
            accuracy=acc, precision=prec, recall=rec, f1=f1v,
            roc_auc=0.5, pr_auc=0.45, eer=eer, eer_threshold=ert,
            true_positives=tp, false_positives=fp,
            true_negatives=tn, false_negatives=fn,
        )


class CalibrationCalculator:

    @staticmethod
    def compute(y_true, y_pred, n_bins=10):
        ece = CalibrationCalculator._ece(y_true, y_pred, n_bins)
        mce = CalibrationCalculator._mce(y_true, y_pred, n_bins)
        brier = float(np.mean((y_pred - y_true) ** 2))
        rel = CalibrationCalculator._reliability(y_true, y_pred, n_bins)
        return CalibrationMetrics(ece=ece, mce=mce, brier_score=brier, reliability_data=rel)

    @staticmethod
    def _ece(y_true, y_pred, n_bins):
        edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_pred > edges[i]) & (y_pred <= edges[i + 1])
            prop = float(np.mean(mask))
            if prop > 0:
                ece += prop * abs(float(np.mean(y_true[mask])) - float(np.mean(y_pred[mask])))
        return ece

    @staticmethod
    def _mce(y_true, y_pred, n_bins):
        edges = np.linspace(0, 1, n_bins + 1)
        mce = 0.0
        for i in range(n_bins):
            mask = (y_pred > edges[i]) & (y_pred <= edges[i + 1])
            if float(np.mean(mask)) > 0:
                mce = max(mce, abs(float(np.mean(y_true[mask])) - float(np.mean(y_pred[mask]))))
        return mce

    @staticmethod
    def _reliability(y_true, y_pred, n_bins):
        edges = np.linspace(0, 1, n_bins + 1)
        centers, accs, confs, counts = [], [], [], []
        for i in range(n_bins):
            mask = (y_pred > edges[i]) & (y_pred <= edges[i + 1])
            centers.append(float((edges[i] + edges[i + 1]) / 2))
            if float(np.mean(mask)) > 0:
                accs.append(float(np.mean(y_true[mask])))
                confs.append(float(np.mean(y_pred[mask])))
                counts.append(int(np.sum(mask)))
            else:
                accs.append(0.0)
                confs.append(0.0)
                counts.append(0)
        return {"bin_centers": centers, "bin_accuracies": accs, "bin_confidences": confs, "bin_counts": counts}


class ModelEvaluator:

    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else RESULTS_DIR / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(self, model_name, model_type, dataset_name, y_true, y_pred,
                       inference_times=None, robustness_data=None):
        cls_metrics = MetricsCalculator.compute(y_true, y_pred)
        cal_metrics = CalibrationCalculator.compute(y_true, y_pred)
        rob_metrics = self._robustness(robustness_data) if robustness_data else None
        if inference_times is not None and len(inference_times) > 0:
            itm = float(np.mean(inference_times))
            its = float(np.std(inference_times))
            tps = 1.0 / itm if itm > 0 else 0
        else:
            itm, its, tps = 0.0, 0.0, 0.0
        return ModelEvaluationResult(
            model_name=model_name, model_type=model_type, dataset_name=dataset_name,
            classification=cls_metrics, calibration=cal_metrics, robustness=rob_metrics,
            inference_time_mean=itm, inference_time_std=its, throughput_fps=tps,
            evaluation_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def evaluate_all_models(self, models_data, dataset_name="combined"):
        results = []
        for md in models_data:
            r = self.evaluate_model(
                md["name"], md["type"], dataset_name,
                np.array(md["y_true"]), np.array(md["y_pred"]),
                md.get("inference_times"), md.get("robustness_data"),
            )
            results.append(r)
        return self._compare(results)

    def _robustness(self, data):
        return RobustnessMetrics(
            clean_accuracy=data.get("clean_accuracy", 0),
            corrupted_accuracies=data.get("corrupted_accuracies", {}),
            mean_corruption_accuracy=data.get("mean_corruption_accuracy", 0),
            accuracy_drop=data.get("accuracy_drop", 0),
            adversarial_success_rate=data.get("adversarial_success_rate", 0),
            mean_prediction_shift=data.get("mean_prediction_shift", 0),
        )

    def _compare(self, results):
        def best(key, reverse):
            return max(results, key=lambda r: getattr(r.classification, key)).model_name if results else ""
        summary = []
        for r in results:
            summary.append({
                "model": r.model_name, "type": r.model_type,
                "roc_auc": r.classification.roc_auc, "pr_auc": r.classification.pr_auc,
                "f1": r.classification.f1, "eer": r.classification.eer,
                "ece": r.calibration.ece, "brier": r.calibration.brier_score,
                "accuracy": r.classification.accuracy,
            })
        return ComparativeEvaluation(
            models=results,
            best_by_roc_auc=best("roc_auc", True),
            best_by_f1=best("f1", True),
            best_by_eer=best("eer", False),
            best_by_calibration=min(results, key=lambda r: r.calibration.ece).model_name if results else "",
            summary_table=summary,
            evaluation_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def save_results(self, comp, output_path=None):
        if output_path is None:
            output_path = self.output_dir / "evaluation_results.json"
        data = {
            "best_by_roc_auc": comp.best_by_roc_auc,
            "best_by_f1": comp.best_by_f1,
            "best_by_eer": comp.best_by_eer,
            "best_by_calibration": comp.best_by_calibration,
            "summary_table": comp.summary_table,
            "evaluation_timestamp": comp.evaluation_timestamp,
            "models": [],
        }
        for m in comp.models:
            data["models"].append({
                "model_name": m.model_name, "model_type": m.model_type,
                "dataset_name": m.dataset_name,
                "classification": asdict(m.classification),
                "calibration": {k: v for k, v in asdict(m.calibration).items() if k != "reliability_data"},
                "robustness": asdict(m.robustness) if m.robustness else None,
                "inference_time_mean": m.inference_time_mean,
                "inference_time_std": m.inference_time_std,
                "throughput_fps": m.throughput_fps,
            })
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Evaluation results saved to %s", output_path)
        return output_path

    def print_summary(self, comp):
        print("\n" + "=" * 80)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 80)
        header = f"{'Model':<25} {'Type':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1':<10} {'EER':<10} {'ECE':<10}"
        print(header)
        print("-" * 80)
        for row in comp.summary_table:
            print(f"{row['model']:<25} {row['type']:<10} {row['roc_auc']:<10.4f} {row['pr_auc']:<10.4f} {row['f1']:<10.4f} {row['eer']:<10.4f} {row['ece']:<10.4f}")
        print("-" * 80)
        print(f"Best ROC-AUC:     {comp.best_by_roc_auc}")
        print(f"Best F1:          {comp.best_by_f1}")
        print(f"Best EER:         {comp.best_by_eer}")
        print(f"Best Calibration: {comp.best_by_calibration}")
        print("=" * 80 + "\n")
