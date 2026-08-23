"""
src/audio_train.py
Training loop for audio deepfake (spoof) detection.

Usage:
    # Folder layout with real/ and fake/ subfolders:
    python -m src.audio_train --data_root data/audio --epochs 15

    # Or from a precomputed split CSV (columns: path,label,split):
    python -m src.audio_train --split_csv data/audio_split.csv --model cnn
"""

import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).parent.parent))
from src.audio_dataset import get_audio_dataloaders
from src.audio_model import build_audio_model


def parse_args():
    p = argparse.ArgumentParser(description="Train an audio deepfake detector")
    p.add_argument("--model", default="efficientnet", choices=["efficientnet", "cnn"])
    p.add_argument("--data_root", default="data/audio",
                   help="Root with real/ and fake/ subfolders")
    p.add_argument("--split_csv", default=None,
                   help="Precomputed split CSV (path,label,split); overrides --data_root")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_dir", default="models")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, epoch=0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for i, (specs, labels) in enumerate(loader):
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
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())
            if is_train and i % 50 == 0:
                print(f"  Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.4f}")

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return total_loss / max(len(loader), 1), auc


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_audio_dataloaders(
        data_root=args.data_root,
        split_csv=args.split_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_audio_model(args.model, pretrained=not args.no_pretrained,
                              dropout=args.dropout, device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"best_audio_{args.model}.pth"
    best_val_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        t = time.time()
        train_loss, train_auc = run_epoch(model, train_loader, criterion, device,
                                          optimizer=optimizer, epoch=epoch)
        val_loss, val_auc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"\nEpoch {epoch}/{args.epochs} ({time.time()-t:.1f}s)")
        print(f"  Train -> Loss: {train_loss:.4f} | AUC: {train_auc:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}   | AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"epoch": epoch, "model_name": args.model,
                        "model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "args": vars(args)}, best_path)
            print(f"  Best model saved - Val AUC: {val_auc:.4f}")

    print("\n" + "=" * 50)
    test_loss, test_auc = run_epoch(model, test_loader, criterion, device)
    print(f"Test AUC: {test_auc:.4f} | Test Loss: {test_loss:.4f}")
    if best_path.exists():
        print(f"Best Val AUC: {best_val_auc:.4f} | Saved: {best_path}")


if __name__ == "__main__":
    main()
