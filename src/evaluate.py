"""
src/evaluate.py
Evaluation: AUC, EER, ROC curve, confusion matrix, score distribution.

Usage:
    python src/evaluate.py --model xception \
        --checkpoint models/best_xception_c23.pth \
        --data_root data/FaceForensics --dataset ff
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)

sys.path.append(str(Path(__file__).parent.parent))
from src.dataset import DeepfakeDataset, get_val_transforms
from src.dataset import load_ff_samples, load_celebdf_samples, load_dfdc_samples, load_split_csv
from src.model import build_model, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="xception")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--data_root",   type=str, required=True)
    parser.add_argument("--dataset",     type=str, default="ff",
                        choices=["ff", "celebdf", "dfdc"])
    parser.add_argument("--compression", type=str, default="c23")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--img_size",    type=int, default=299)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split_csv",   type=str, default=None,
                        help="Path to split CSV with columns: path,label,split")
    parser.add_argument("--split_name",  type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate when --split_csv is used")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--no_plots",    action="store_true")
    return parser.parse_args()


def compute_eer(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2, thresholds[idx]


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        probs = torch.sigmoid(model(images)).cpu().numpy()
        all_probs.extend(probs.flatten())
        all_labels.extend(labels.numpy().flatten())
    return np.array(all_labels), np.array(all_probs)


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    eer, eer_thresh = compute_eer(labels, probs)
    return {
        "Threshold":     threshold,
        "AUC":           roc_auc_score(labels, probs),
        "Accuracy":      accuracy_score(labels, preds),
        "Precision":     precision_score(labels, preds, zero_division=0),
        "Recall":        recall_score(labels, preds, zero_division=0),
        "F1":            f1_score(labels, preds, zero_division=0),
        "EER":           eer,
        "EER_threshold": eer_thresh,
    }


def print_metrics(metrics, name=""):
    print(f"\n{'='*50}\nResults — {name}\n{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")
    print("="*50)


def plot_roc(labels, probs, name, path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}", lw=2, color="#534AB7")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC — {name}"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"ROC saved: {path}")


def plot_cm(labels, probs, name, path, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Real","Fake"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Confusion matrix saved: {path}")


def plot_dist(labels, probs, name, path):
    plt.figure(figsize=(9, 5))
    plt.hist(probs[labels==0], bins=50, alpha=0.6, label="Real",  color="#2ECC71", density=True)
    plt.hist(probs[labels==1], bins=50, alpha=0.6, label="Fake",  color="#E74C3C", density=True)
    plt.xlabel("Predicted Probability (Fake)")
    plt.ylabel("Density"); plt.title(f"Score Distribution — {name}")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Distribution saved: {path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.split_csv:
        split_to_samples = load_split_csv(args.split_csv, data_root=args.data_root)
        samples = split_to_samples[args.split_name]
        if not samples:
            raise ValueError(f"No samples found for split='{args.split_name}' in {args.split_csv}")
        name = f"{args.dataset.upper()} [{args.split_name}]"
    else:
        if args.dataset == "ff":
            samples = load_ff_samples(args.data_root, compression=args.compression)
            name = f"FaceForensics++ ({args.compression})"
        elif args.dataset == "celebdf":
            samples = load_celebdf_samples(args.data_root)
            name = "Celeb-DF v2"
        else:
            samples = load_dfdc_samples(args.data_root)
            name = "DFDC"

    loader = DataLoader(
        DeepfakeDataset(samples, transform=get_val_transforms(args.img_size)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, pretrained=False, device=device)
    model = load_checkpoint(model, args.checkpoint, device=device).to(device)

    labels, probs = run_inference(model, loader, device)
    metrics = compute_metrics(labels, probs, threshold=args.threshold)
    print_metrics(metrics, name)

    split_tag = f"_{args.split_name}" if args.split_csv else ""
    tag = f"{args.model}_{args.dataset}{split_tag}_{args.compression}"
    with open(results_dir / f"metrics_{tag}.txt", "w") as f:
        f.write(f"Model: {args.model}\nDataset: {name}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    if not args.no_plots:
        plot_roc(labels, probs, name, results_dir / f"roc_{tag}.png")
        plot_cm(labels, probs, name, results_dir / f"cm_{tag}.png", threshold=args.threshold)
        plot_dist(labels, probs, name, results_dir / f"dist_{tag}.png")


if __name__ == "__main__":
    main()
