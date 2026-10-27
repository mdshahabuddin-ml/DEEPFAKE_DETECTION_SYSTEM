"""
scripts/evaluate_indian_heldout.py

Held-out audit of the DEPLOYED checkpoint (models/best_xception_celebdf.pth)
on the Indian Celebrity test split (data/splits/indian-deepfake-face/test.csv).

Two inference paths, same model / threshold:
  1. full-frame : Resize(299) + ImageNet norm — identical to training/val
                  preprocessing (src/dataset.get_val_transforms).
  2. app-crop   : Haar largest-face + 20% margin crop, BGR->RGB, then the
                  same val transforms — identical to app.py Image Detection
                  (extract_face -> predict).

Also runs:
  - leakage check (test paths must not appear in train.csv),
  - consistency check vs results/indian_audit/all_predictions.csv,
  - in-distribution sanity check on a few Celeb-DF v2 videos
    (proves the checkpoint itself is healthy).

Writes metrics + predictions to results/indian_heldout_audit/.

Usage:
    .venv/Scripts/python.exe scripts/evaluate_indian_heldout.py
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_val_transforms  # noqa: E402
from src.inference import FaceDetector  # noqa: E402
from src.model import build_model, load_checkpoint  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "models" / "best_xception_celebdf.pth"
TEST_CSV = PROJECT_ROOT / "data" / "splits" / "indian-deepfake-face" / "test.csv"
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "indian-deepfake-face" / "train.csv"
PRIOR_AUDIT = PROJECT_ROOT / "results" / "indian_audit" / "all_predictions.csv"
CELEBDF_ROOT = PROJECT_ROOT / "data" / "celeb-df-v2"
OUT_DIR = PROJECT_ROOT / "results" / "indian_heldout_audit"

IMG_SIZE = 299
THRESHOLD = 0.47
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrameDataset(Dataset):
    """Full frames, BGR->RGB + val transforms (training-consistent)."""

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


@torch.no_grad()
def infer_loader(model, loader):
    labels, probs = [], []
    for images, lb in loader:
        images = images.to(DEVICE)
        pr = torch.sigmoid(model(images)).view(-1)
        labels.extend(lb.numpy().tolist())
        probs.extend(pr.cpu().numpy().tolist())
    return np.array(labels), np.array(probs)


def report(name, y, p, threshold=THRESHOLD):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    print(f"--- {name} @ {threshold} ---")
    print(f"  n={len(y)} (REAL={int((y == 0).sum())}, FAKE={int((y == 1).sum())})")
    print(f"  Accuracy  : {accuracy_score(y, pred):.4f}")
    print(f"  Precision : {precision_score(y, pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y, pred, zero_division=0):.4f}")
    print(f"  F1-score  : {f1_score(y, pred, zero_division=0):.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  FPR={fp / max(fp + tn, 1):.4f}  FNR={fn / max(fn + tp, 1):.4f}")
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "auc": auc}


def threshold_sweep(y, p):
    print(f"\n{'t':>6} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} "
          f"{'fpr':>7} {'fnr':>7}")
    best = (0.0, None)
    for t in np.arange(0.05, 0.96, 0.05):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        a = accuracy_score(y, pred)
        f = f1_score(y, pred, zero_division=0)
        print(f"{t:>6.2f} {a:>7.4f} "
              f"{precision_score(y, pred, zero_division=0):>7.4f} "
              f"{recall_score(y, pred, zero_division=0):>7.4f} "
              f"{f:>7.4f} {fp / max(fp + tn, 1):>7.4f} "
              f"{fn / max(fn + tp, 1):>7.4f}")
        if f > best[0]:
            best = (f, t)
    print(f"\nBest F1 threshold (report only, not applied): "
          f"{best[1]:.2f} (F1={best[0]:.4f})")


def main():
    print("Loading model:", CHECKPOINT.name)
    model = build_model("xception", pretrained=False, dropout=0.5,
                        device=str(DEVICE))
    load_checkpoint(model, str(CHECKPOINT), device=str(DEVICE))
    model.eval()
    transform = get_val_transforms(IMG_SIZE)

    # ---- Leakage check ----
    test_rows = list(csv.DictReader(open(TEST_CSV)))
    train_paths = set(r["path"] for r in csv.DictReader(open(TRAIN_CSV)))
    test_samples = [(r["path"], int(r["label"])) for r in test_rows]
    missing = [p for p, _ in test_samples if not Path(p).exists()]
    overlap = [p for p, _ in test_samples if p in train_paths]
    print(f"Test samples: {len(test_samples)} | missing files: {len(missing)} | "
          f"train/test path overlap: {len(overlap)}")
    assert not missing, f"{len(missing)} test files missing"
    assert not overlap, "LEAKAGE: test paths present in train.csv"

    # ---- Path 1: full-frame (training-consistent) ----
    t0 = time.time()
    full_loader = DataLoader(FrameDataset(test_samples, transform),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    y_full, p_full = infer_loader(model, full_loader)
    print(f"Full-frame inference: {time.time() - t0:.1f}s")
    r_full = report("FULL-FRAME (training-consistent)", y_full, p_full)

    # ---- Consistency vs prior audit ----
    if PRIOR_AUDIT.exists():
        prior = {r["path"]: float(r["fake_prob"])
                 for r in csv.DictReader(open(PRIOR_AUDIT))}
        common = [(p_full[i], prior[test_samples[i][0]])
                  for i in range(len(test_samples))
                  if test_samples[i][0] in prior]
        if common:
            a = np.array([c[0] for c in common])
            b = np.array([c[1] for c in common])
            corr = float(np.corrcoef(a, b)[0, 1])
            print(f"Consistency vs results/indian_audit/all_predictions.csv: "
                  f"{len(common)} common, Pearson r={corr:.4f}, "
                  f"max|dp|={float(np.max(np.abs(a - b))):.5f}")

    # ---- Path 2: app-identical face crop ----
    print("\nApp-crop path: Haar largest-face + 20% margin (as in app.py)...")
    fd = FaceDetector(backend="auto")
    crops, crop_labels, faced = [], [], []
    no_face = 0
    t0 = time.time()
    for path, label in test_samples:
        bgr = cv2.imread(path)
        crop = fd.crop_largest(bgr) if bgr is not None else None
        if crop is None:
            no_face += 1
            faced.append(False)
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(transform(image=rgb)["image"])
        crop_labels.append(label)
        faced.append(True)
    print(f"Face detected: {len(crops)}/{len(test_samples)} "
          f"(no-face: {no_face}) in {time.time() - t0:.1f}s")
    t0 = time.time()
    y_crop, p_crop = [], []
    with torch.no_grad():
        for i in range(0, len(crops), BATCH_SIZE):
            batch = torch.stack(crops[i:i + BATCH_SIZE]).to(DEVICE)
            pr = torch.sigmoid(model(batch)).view(-1)
            p_crop.extend(pr.cpu().numpy().tolist())
            y_crop.extend(crop_labels[i:i + BATCH_SIZE])
    y_crop, p_crop = np.array(y_crop), np.array(p_crop)
    print(f"App-crop inference: {time.time() - t0:.1f}s")
    r_crop = report("APP-CROP (deployment-identical, face-detected subset)",
                    y_crop, p_crop)

    # ---- In-distribution sanity (Celeb-DF v2, few videos) ----
    print("\nIn-distribution sanity: Celeb-DF v2 sample frames...")
    sanity = []
    for folder, label, n_vids in [("Celeb-real", 0, 2), ("YouTube-real", 0, 2),
                                  ("Celeb-synthesis", 1, 4)]:
        vids = sorted((CELEBDF_ROOT / folder).glob("*.mp4"))[:n_vids]
        for vp in vids:
            cap = cv2.VideoCapture(str(vp))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            for fi in np.linspace(0, max(total - 1, 0), 10).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, fr = cap.read()
                if ok and fr is not None and fr.size > 0:
                    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    sanity.append((transform(image=rgb)["image"], label))
            cap.release()
    ys = np.array([lb for _, lb in sanity])
    ps = []
    with torch.no_grad():
        for i in range(0, len(sanity), BATCH_SIZE):
            batch = torch.stack([s[0] for s in sanity[i:i + BATCH_SIZE]]).to(DEVICE)
            ps.extend(torch.sigmoid(model(batch)).view(-1).cpu().numpy().tolist())
    ps = np.array(ps)
    print(f"  n={len(ys)} (REAL={(ys == 0).sum()}, FAKE={(ys == 1).sum()})")
    print(f"  mean P(FAKE|REAL)={ps[ys == 0].mean():.3f}  "
          f"mean P(FAKE|FAKE)={ps[ys == 1].mean():.3f}  "
          f"AUC={roc_auc_score(ys, ps):.4f}")

    # ---- Threshold sweep on app-crop path ----
    print("\nThreshold sweep — APP-CROP path (informational only):")
    threshold_sweep(y_crop, p_crop)

    # ---- Save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "predictions_fullframe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "fake_prob", "threshold"])
        for (p, lb), prob in zip(test_samples, p_full):
            w.writerow([p, lb, f"{prob:.4f}", THRESHOLD])
    # app-crop probs aligned to detected subset
    det_idx = [i for i, fd_ in enumerate(faced) if fd_]
    prob_by_index = {i: float(p_crop[j]) for j, i in enumerate(det_idx)}
    with open(OUT_DIR / "predictions_appcrop.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "fake_prob_appcrop", "face_detected",
                    "threshold"])
        for i, (p, lb) in enumerate(test_samples):
            w.writerow([p, lb,
                        f"{prob_by_index[i]:.4f}" if i in prob_by_index else "",
                        int(faced[i]), THRESHOLD])
    with open(OUT_DIR / "predictions_appcrop_detected.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "fake_prob_appcrop", "threshold"])
        for j, i in enumerate(det_idx):
            w.writerow([test_samples[i][0], test_samples[i][1],
                        f"{float(p_crop[j]):.4f}", THRESHOLD])
    with open(OUT_DIR / "metrics.txt", "w") as f:
        f.write(f"checkpoint={CHECKPOINT.name}\nthreshold={THRESHOLD}\n")
        f.write(f"full_frame={r_full}\napp_crop={r_crop}\n"
                f"no_face={no_face}\n")
    print(f"\nSaved predictions + metrics to {OUT_DIR}/")


if __name__ == "__main__":
    main()
