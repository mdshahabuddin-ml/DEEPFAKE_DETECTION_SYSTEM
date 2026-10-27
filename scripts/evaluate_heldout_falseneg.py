"""
scripts/evaluate_heldout_falseneg.py

False-negative audit on the COMPLETE held-out Indian Deepfake data
(test.csv + val.csv — both unseen by the deployed Celeb-DF checkpoint).

Reuses saved predictions (no re-inference, no downloads, no retraining):
  results/indian_heldout_audit/predictions_fullframe.csv  (test, full-frame)
  results/indian_heldout_audit/predictions_appcrop_detected.csv (test, app-crop)
  results/webcam_calibration/val_predictions.csv          (val, full-frame)

Verifies prediction-file integrity against the split CSVs, anchors the
reported example (FAKE @ ~39.8%), and reports Accuracy, Precision, Recall,
F1, confusion matrix, FNR and FPR per split and combined.

Usage:
    .venv/Scripts/python.exe scripts/evaluate_heldout_falseneg.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

THRESHOLD = 0.47
AUDIT_DIR = PROJECT_ROOT / "results" / "indian_heldout_audit"
CAL_DIR = PROJECT_ROOT / "results" / "webcam_calibration"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "indian-deepfake-face"


def load_preds(path, prob_col):
    df = pd.read_csv(path)
    return df["true_label"].to_numpy(), df[prob_col].to_numpy(), df


def check_integrity(pred_df, split_name, prob_col="fake_prob"):
    split = pd.read_csv(SPLIT_DIR / f"{split_name}.csv")
    assert len(pred_df) == len(split), (len(pred_df), len(split))
    assert set(pred_df["path"]) == set(split["path"]), "path mismatch"
    merged = pred_df.merge(split, on="path", suffixes=("_p", "_s"))
    assert (merged["true_label"] == merged["label"]).all(), "label mismatch"
    assert pred_df[prob_col].between(0, 1).all(), "prob out of range"
    print(f"integrity {split_name}: {len(pred_df)} rows, paths+labels match, "
          f"probs in [0,1]")


def report(name, y, p):
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    print(f"--- {name} @ {THRESHOLD} (n={len(y)}, "
          f"REAL={(y == 0).sum()}, FAKE={(y == 1).sum()}) ---")
    print(f"  Accuracy  : {accuracy_score(y, pred):.4f}")
    print(f"  Precision : {precision_score(y, pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y, pred, zero_division=0):.4f}")
    print(f"  F1-score  : {f1_score(y, pred, zero_division=0):.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  [[TN={tn} FP={fp}]")
    print(f"   [FN={fn} TP={tp}]]")
    print(f"  FNR={fn / max(fn + tp, 1):.4f}  "
          f"FPR={fp / max(fp + tn, 1):.4f}")
    return {"n": len(y), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "acc": accuracy_score(y, pred),
            "prec": precision_score(y, pred, zero_division=0),
            "rec": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "auc": auc}


def main():
    # Integrity of every prediction file against its split CSV.
    df_test = pd.read_csv(AUDIT_DIR / "predictions_fullframe.csv")
    check_integrity(df_test, "test")
    df_val = pd.read_csv(CAL_DIR / "val_predictions.csv")
    check_integrity(df_val, "val")
    df_crop = pd.read_csv(AUDIT_DIR / "predictions_appcrop_detected.csv")
    assert len(df_crop) == 1120, len(df_crop)
    assert set(df_crop["path"]) <= set(df_test["path"])
    print(f"integrity test/app-crop: {len(df_crop)} detected-subset rows, "
          f"subset of test paths")

    # Anchor the reported example: closest FAKE to 0.398.
    fakes = df_test[df_test["true_label"] == 1].copy()
    fakes["dist"] = (fakes["fake_prob"] - 0.398).abs()
    ex = fakes.sort_values("dist").iloc[0]
    print(f"\nExemplar for reported case (FAKE scored ~39.8%): "
          f"{Path(ex['path']).name} -> {ex['fake_prob']:.4f} "
          f"(=> REAL @ {THRESHOLD})")
    print(f"FAKE prob distribution: mean={fakes['fake_prob'].mean():.4f} "
          f"median={fakes['fake_prob'].median():.4f} "
          f"frac<0.47={(fakes['fake_prob'] < THRESHOLD).mean():.4f}")

    y_test, p_test, _ = load_preds(AUDIT_DIR / "predictions_fullframe.csv",
                                   "fake_prob")
    y_val, p_val, _ = load_preds(CAL_DIR / "val_predictions.csv",
                                 "fake_prob")
    yc = np.concatenate([y_test, y_val])
    pc = np.concatenate([p_test, p_val])

    print()
    r_test = report("TEST full-frame", y_test, p_test)
    print()
    r_val = report("VAL full-frame", y_val, p_val)
    print()
    r_all = report("COMPLETE HELD-OUT (test+val) full-frame", yc, pc)
    print()
    y_app, p_app, _ = load_preds(
        AUDIT_DIR / "predictions_appcrop_detected.csv", "fake_prob_appcrop")
    r_app = report("TEST app-crop (deployment path, detected subset)",
                   y_app, p_app)

    with open(AUDIT_DIR / "falseneg_metrics.txt", "w") as f:
        f.write(f"threshold={THRESHOLD}\n")
        for name, r in [("test_fullframe", r_test), ("val_fullframe", r_val),
                        ("complete_heldout_fullframe", r_all),
                        ("test_appcrop", r_app)]:
            f.write(f"{name}={r}\n")
        f.write(f"exemplar_fake_at_0398={ex['path']} prob={ex['fake_prob']}\n")
    print("\nSaved metrics to results/indian_heldout_audit/falseneg_metrics.txt")


if __name__ == "__main__":
    main()
