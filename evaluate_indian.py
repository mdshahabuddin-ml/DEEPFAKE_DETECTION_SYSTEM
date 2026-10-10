"""
Audit evaluation of the existing Xception deepfake model on the Indian Deepfake test set.
Batched DataLoader inference for CPU efficiency.
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

from src.model import build_model, load_checkpoint

# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = Path("models/best_xception_celebdf.pth")
TEST_CSV = Path("data/splits/indian-deepfake-face/test.csv")
IMG_SIZE = 299
THRESHOLD = 0.47
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ============================================================
# DATASET
# ============================================================

class EvalDataset(Dataset):
    def __init__(self, samples, transform=None):
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


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("#" * 60)
    print("#  INDIAN DEEPFAKE TEST SET AUDIT")
    print(f"#  Threshold: {THRESHOLD}  |  Checkpoint: {CHECKPOINT.name}")
    print("#" * 60)
    print()

    # --- Load model ---
    print("Loading model...")
    model = build_model("xception", pretrained=False, dropout=0.5, device=str(DEVICE))
    load_checkpoint(model, str(CHECKPOINT), device=str(DEVICE))
    model.eval()

    # --- Load test set ---
    print("Loading test set...")
    samples = []
    with open(TEST_CSV, "r") as f:
        for row in csv.DictReader(f):
            p = row["path"]
            if Path(p).exists():
                samples.append((p, int(row["label"])))

    real_count = sum(1 for _, l in samples if l == 0)
    fake_count = sum(1 for _, l in samples if l == 1)
    print(f"Test set: {len(samples)} samples  (REAL={real_count}, FAKE={fake_count})")

    # --- Batched inference ---
    dataset = EvalDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_labels = []
    all_probs = []
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).view(-1)
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"  Batch {batch_idx+1}/{len(loader)}")

    elapsed = time.time() - t0
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)
    preds_arr = (probs_arr >= THRESHOLD).astype(int)

    # --- Metrics ---
    acc = accuracy_score(labels_arr, preds_arr)
    prec = precision_score(labels_arr, preds_arr, zero_division=0)
    rec = recall_score(labels_arr, preds_arr, zero_division=0)
    f1 = f1_score(labels_arr, preds_arr, zero_division=0)
    try:
        auc = roc_auc_score(labels_arr, probs_arr)
    except ValueError:
        auc = 0.0
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print()
    print("=" * 60)
    print(f"  RESULTS @ threshold = {THRESHOLD}")
    print("=" * 60)
    print(f"  Dataset   : {len(labels_arr)} samples (REAL={real_count}, FAKE={fake_count})")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print()
    print(f"  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                REAL    FAKE")
    print(f"  Actual REAL  [{tn:>5}   {fp:>5}]")
    print(f"  Actual FAKE  [{fn:>5}   {tp:>5}]")
    print()
    print(f"  True Negatives  (REAL->REAL) : {tn}")
    print(f"  False Positives (REAL->FAKE) : {fp}")
    print(f"  False Negatives (FAKE->REAL) : {fn}")
    print(f"  True Positives  (FAKE->FAKE) : {tp}")
    print(f"  Inference time  : {elapsed:.1f}s ({elapsed/len(labels_arr)*1000:.1f}ms/img)")

    # --- Threshold sweep ---
    print()
    print("=" * 60)
    print("THRESHOLD SWEEP")
    print("=" * 60)
    print(f"{'Thresh':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7}  {'FP':>5} {'FN':>5}")
    print("-" * 72)

    best_f1 = 0.0
    best_t = 0.5

    for t in np.arange(0.05, 0.96, 0.05):
        p = (probs_arr >= t).astype(int)
        cm_t = confusion_matrix(labels_arr, p, labels=[0, 1])
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        a = accuracy_score(labels_arr, p)
        pr = precision_score(labels_arr, p, zero_division=0)
        rc = recall_score(labels_arr, p, zero_division=0)
        f = f1_score(labels_arr, p, zero_division=0)
        try:
            au = roc_auc_score(labels_arr, probs_arr)
        except ValueError:
            au = 0.0
        print(f"{t:>7.2f} {a:>7.4f} {pr:>7.4f} {rc:>7.4f} {f:>7.4f} {au:>7.4f}  {fp_t:>5} {fn_t:>5}")
        if f > best_f1:
            best_f1 = f
            best_t = t

    # Fine sweep
    print()
    print("  Fine-grained sweep (0.30 - 0.70):")
    print(f"  {'Thresh':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("  " + "-" * 42)
    for t in np.arange(0.30, 0.71, 0.01):
        p = (probs_arr >= t).astype(int)
        a = accuracy_score(labels_arr, p)
        pr = precision_score(labels_arr, p, zero_division=0)
        rc = recall_score(labels_arr, p, zero_division=0)
        f = f1_score(labels_arr, p, zero_division=0)
        if f >= best_f1 - 0.02:
            print(f"  {t:>7.2f} {a:>7.4f} {pr:>7.4f} {rc:>7.4f} {f:>7.4f}")
        if f > best_f1:
            best_f1 = f
            best_t = t

    print(f"\n  >>> Best threshold by F1: {best_t:.2f}  (F1={best_f1:.4f})")

    # --- Save error lists ---
    out_dir = Path("results/indian_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_list = [(samples[i][0], probs_arr[i]) for i in range(len(samples))
               if labels_arr[i] == 0 and preds_arr[i] == 1]
    fn_list = [(samples[i][0], probs_arr[i]) for i in range(len(samples))
               if labels_arr[i] == 1 and preds_arr[i] == 0]

    with open(out_dir / "false_positives.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "fake_prob"])
        for p, prob in fp_list:
            w.writerow([p, f"{prob:.4f}"])

    with open(out_dir / "false_negatives.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "fake_prob"])
        for p, prob in fn_list:
            w.writerow([p, f"{prob:.4f}"])

    with open(out_dir / "all_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "fake_prob", "prediction", "threshold"])
        for i in range(len(samples)):
            w.writerow([samples[i][0], labels_arr[i], f"{probs_arr[i]:.4f}",
                         preds_arr[i], THRESHOLD])

    print(f"\n  Error lists saved to {out_dir}/")

    # --- Summary ---
    print()
    print("#" * 60)
    print("#  AUDIT SUMMARY")
    print("#" * 60)
    print()
    print("  LABEL MAPPING:  REAL=0, FAKE=1  |  sigmoid = P(FAKE)")
    print(f"  THRESHOLD:      {THRESHOLD}  (prob >= {THRESHOLD} => FAKE)")
    print(f"  PREPROCESSING:  Resize {IMG_SIZE}x{IMG_SIZE} + ImageNet norm")
    print(f"  FACE DETECT:    Haar cascade, largest face, 20%% margin")
    print()
    print(f"  IMBALANCE:  REAL={real_count} ({real_count/len(samples)*100:.1f}%)  FAKE={fake_count} ({fake_count/len(samples)*100:.1f}%)  Ratio=1:{fake_count//max(real_count,1)}")
    print()
    print(f"  RECOMMENDED THRESHOLD: {best_t:.2f}  (F1={best_f1:.4f})")
    print()


if __name__ == "__main__":
    main()
