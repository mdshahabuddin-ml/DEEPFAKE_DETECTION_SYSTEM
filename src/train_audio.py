"""
src/train_audio.py
Train the audio deepfake (AI voice) CNN on a labelled folder of real/fake
audio, with train/val/test split. Reports AUC, accuracy and EER the same way
src/evaluate.py does, and saves the best checkpoint to
``models/audio_detector.pth`` (the path ``src/audio_inference.py`` loads).

Folder layout (label 0 = REAL, 1 = FAKE)::

    data/audio/
        real/   # bonafide / genuine / human also accepted
            *.wav / *.flac / *.mp3 / *.m4a / *.ogg
        fake/   # spoof / synthetic / tts / vc also accepted
            ...

Or pass a split CSV (columns: path,label,split with split in
train/val/test)::

    python -m src.train_audio --split_csv data/audio_split.csv

Usage::

    python -m src.train_audio --data_root data/audio --epochs 15
    python -m src.train_audio --data_root data/audio --model cnn --epochs 20

Datasets to download
--------------------
1. ASVspoof 2019 LA (bonafide vs TTS/VC spoofs):
   https://www.asvspoof.org/index2019.html
   Use the LA train/dev/eval partitions; map bonafide -> real/,
   spoof -> fake/.
2. WaveFake (generated speech from several vocoders):
   https://github.com/RUB-SysSec/WaveFake
   Generated files -> fake/; add real speech (e.g. LJSpeech) -> real/.
3. In-the-Wild (real vs celebrity deepfake clips in the wild):
   https://github.com/MickJagger19/In-the-Wild-Audio-Deepfake-Data
   Copy its real clips -> real/, fake clips -> fake/.

Keep the three sources as separate sub-runs or merge them into one
real/ + fake/ tree; the loader only cares about the folder names.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(str(Path(__file__).parent.parent))

from src.audio_dataset import get_audio_dataloaders
from src.audio_model import build_audio_model

DEFAULT_SAVE = Path("models") / "audio_detector.pth"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the AI-voice CNN")
    p.add_argument("--model", default="efficientnet",
                   choices=["efficientnet", "cnn"])
    p.add_argument("--data_root", default="data/audio",
                   help="Root with real/ and fake/ subfolders")
    p.add_argument("--split_csv", default=None,
                   help="Split CSV (path,label,split); overrides --data_root")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save", default=str(DEFAULT_SAVE),
                   help="Where to save the best checkpoint")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_eer(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Equal error rate + threshold (same helper as src/evaluate.py)."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])


def compute_metrics(
    labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> dict:
    preds = (np.asarray(probs) >= threshold).astype(int)
    eer, eer_thresh = compute_eer(np.asarray(labels), np.asarray(probs))
    return {
        "threshold": threshold,
        "AUC": float(roc_auc_score(labels, probs)),
        "Accuracy": float(accuracy_score(labels, preds)),
        "EER": float(eer),
        "EER_threshold": float(eer_thresh),
    }


def print_metrics(metrics: dict, name: str = "") -> None:
    print(f"\n{'=' * 50}\nResults - {name}\n{'=' * 50}")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("=" * 50)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for specs, labels in loader:
            specs = specs.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            all_probs.extend(
                torch.sigmoid(logits).detach().cpu().numpy().flatten()
            )
            all_labels.extend(labels.detach().cpu().numpy().flatten())
    labels_np = np.asarray(all_labels, dtype=int)
    probs_np = np.asarray(all_probs, dtype=float)
    try:
        auc = float(roc_auc_score(labels_np, probs_np))
    except Exception:
        auc = 0.0
    return total_loss / max(len(loader), 1), auc, labels_np, probs_np


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_audio_dataloaders(
        data_root=args.data_root,
        split_csv=args.split_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    model = build_audio_model(
        args.model, pretrained=not args.no_pretrained,
        dropout=args.dropout, device=device,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        t = time.time()
        train_loss, train_auc, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_auc, val_labels, val_probs = run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()
        print(f"\nEpoch {epoch}/{args.epochs} ({time.time() - t:.1f}s)")
        print(f"  Train -> Loss: {train_loss:.4f} | AUC: {train_auc:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}   | AUC: {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {"epoch": epoch, "model_name": args.model,
                 "model_state_dict": model.state_dict(),
                 "val_auc": val_auc, "args": vars(args)},
                save_path,
            )
            print(f"  Best model saved -> {save_path} (Val AUC {val_auc:.4f})")

    _, _, test_labels, test_probs = run_epoch(
        model, test_loader, criterion, device
    )
    metrics = compute_metrics(test_labels, test_probs,
                              threshold=args.threshold)
    print_metrics(metrics, f"audio test ({args.model})")
    with open(save_path.with_suffix(".metrics.txt"), "w") as f:
        f.write(f"Model: {args.model}\nCheckpoint: {save_path}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Best Val AUC: {best_val_auc:.4f} | Saved: {save_path}")


if __name__ == "__main__":
    main()
