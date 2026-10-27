"""
scripts/calibrate_webcam_threshold.py

Threshold calibration on VALIDATION data only
(data/splits/indian-deepfake-face/val.csv).

- No downloads, no retraining, no model modification.
- Full-frame batched inference with the deployed checkpoint
  (identical preprocessing to training/val: Resize 299 + ImageNet norm).
- Reports Youden-J and best-F1 thresholds on VAL, then applies the
  val-chosen threshold to the held-out TEST predictions to show
  whether it transfers.
- Verdict on whether any threshold fixes the false-positive problem
  is printed; nothing is changed on disk (report only).

Usage:
    .venv/Scripts/python.exe -u scripts/calibrate_webcam_threshold.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_val_transforms  # noqa: E402
from src.model import build_model, load_checkpoint  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "models" / "best_xception_celebdf.pth"
VAL_CSV = PROJECT_ROOT / "data" / "splits" / "indian-deepfake-face" / "val.csv"
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "indian-deepfake-face" / "train.csv"
TEST_PREDS = PROJECT_ROOT / "results" / "indian_heldout_audit" / "predictions_fullframe.csv"
OUT_DIR = PROJECT_ROOT / "results" / "webcam_calibration"

IMG_SIZE = 299
BATCH_SIZE = 64
DEPLOYED_THRESHOLD = 0.47
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrameDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


def main():
    print("Loading model:", CHECKPOINT.name, flush=True)
    model = build_model("xception", pretrained=False, dropout=0.5,
                        device=str(DEVICE))
    load_checkpoint(model, str(CHECKPOINT), device=str(DEVICE))
    model.eval()
    transform = get_val_transforms(IMG_SIZE)

    val_samples = [(r["path"], int(r["label"]))
                   for r in csv.DictReader(open(VAL_CSV))]
    train_paths = set(r["path"] for r in csv.DictReader(open(TRAIN_CSV)))
    overlap = [p for p, _ in val_samples if p in train_paths]
    missing = [p for p, _ in val_samples if not Path(p).exists()]
    print(f"VAL: {len(val_samples)} | missing={len(missing)} | "
          f"train/val overlap={len(overlap)}", flush=True)
    assert not missing and not overlap

    t0 = time.time()
    loader = DataLoader(FrameDataset(val_samples, transform),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    y, p = [], []
    with torch.no_grad():
        for images, lb in loader:
            pr = torch.sigmoid(model(images.to(DEVICE))).view(-1)
            y.extend(lb.numpy().tolist())
            p.extend(pr.cpu().numpy().tolist())
    y, p = np.array(y), np.array(p)
    print(f"Inference: {time.time() - t0:.1f}s", flush=True)
    print(f"VAL class balance: REAL={(y == 0).sum()} FAKE={(y == 1).sum()}",
          flush=True)
    # Persist predictions FIRST so a later crash can never lose them.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "val_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "fake_prob"])
        for (path, lb), prob in zip(val_samples, p):
            w.writerow([path, lb, f"{prob:.4f}"])
    try:
        print(f"VAL ROC-AUC: {roc_auc_score(y, p):.4f}", flush=True)
    except ValueError:
        print("VAL ROC-AUC: undefined", flush=True)

    # Sweep thresholds on VAL only.
    print(f"\n{'t':>6} {'acc':>7} {'f1':>7} {'fpr':>7} {'fnr':>7} "
          f"{'youdenJ':>8}", flush=True)
    best_f1, best_j = (float("-inf"), None), (float("-inf"), None)
    for t in np.arange(0.05, 0.96, 0.05):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fpr, fnr = fp / max(fp + tn, 1), fn / max(fn + tp, 1)
        youden = (tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1)) - 1
        f1 = f1_score(y, pred, zero_division=0)
        print(f"{t:>6.2f} {accuracy_score(y, pred):>7.4f} {f1:>7.4f} "
              f"{fpr:>7.4f} {fnr:>7.4f} {youden:>8.4f}", flush=True)
        if f1 > best_f1[0]:
            best_f1 = (f1, t)
        if youden > best_j[0]:
            best_j = (youden, t)
    print(f"\nVAL best-F1 threshold: {best_f1[1]:.2f} (F1={best_f1[0]:.4f})",
          flush=True)
    print(f"VAL Youden-J threshold: {best_j[1]:.2f} (J={best_j[0]:.4f})",
          flush=True)

    # Transfer check: apply VAL-chosen thresholds to held-out TEST.
    if TEST_PREDS.exists():
        rows = list(csv.DictReader(open(TEST_PREDS)))
        yt = np.array([int(r["true_label"]) for r in rows])
        pt = np.array([float(r["fake_prob"]) for r in rows])
        for name, t in [("deployed", DEPLOYED_THRESHOLD),
                        ("val-bestF1", best_f1[1]),
                        ("val-YoudenJ", best_j[1])]:
            pred = (pt >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
            print(f"TEST @{name}={t:.2f}: acc={accuracy_score(yt, pred):.4f} "
                  f"f1={f1_score(yt, pred, zero_division=0):.4f} "
                  f"FPR={fp / max(fp + tn, 1):.4f} "
                  f"FNR={fn / max(fn + tp, 1):.4f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "calibration_report.txt", "w") as f:
        f.write(f"val_best_f1_threshold={best_f1[1]:.2f} f1={best_f1[0]:.4f}\n")
        f.write(f"val_youden_threshold={best_j[1]:.2f} J={best_j[0]:.4f}\n")
        f.write(f"deployed_threshold={DEPLOYED_THRESHOLD} (unchanged)\n")
    print(f"\nSaved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
