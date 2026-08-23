"""
src/train.py
Training loop for deepfake detection with W&B logging.

Usage:
    python src/train.py --model xception --data_root data/FaceForensics --epochs 10
    python src/train.py --dataset celebdf --data_root data/processed/celeb-df-v2 --epochs 10
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
from src.dataset import (
    get_ff_dataloaders, get_celebdf_dataloaders,
    get_dataloaders_from_split_csv, get_dataloaders_from_split_dir,
)
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,   default="xception",
                        choices=["xception", "efficientnet", "vit"])
    parser.add_argument("--dataset",     type=str,   default="ff",
                        choices=["ff", "celebdf", "dfdc"])
    parser.add_argument("--data_root",   type=str,   default="data/FaceForensics")
    parser.add_argument("--compression", type=str,   default="c23")
    parser.add_argument("--max_frames_per_video", type=int, default=30)
    parser.add_argument("--max_videos_per_class", type=int, default=None)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--img_size",    type=int,   default=299)
    parser.add_argument("--dropout",     type=float, default=0.5)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train from random weights instead of downloading pretrained weights")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--save_dir",    type=str,   default="models")
    parser.add_argument("--split_csv",   type=str,   default=None,
                        help="Path to precomputed split CSV with columns: path,label,split")
    parser.add_argument("--split_dir",   type=str,   default=None,
                        help="Directory with separate train.csv/val.csv/test.csv (path,label)")
    parser.add_argument("--wandb_project", type=str, default="deepfake-detection")
    parser.add_argument("--no_wandb",    action="store_true")
    parser.add_argument("--balanced_train", action="store_true",
                        help="Use weighted sampling to balance real and fake examples during training")
    parser.add_argument("--seed",        type=int,   default=42)
    return parser.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        if i % 50 == 0:
            print(f"  Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.4f}")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return total_loss / len(loader), auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return total_loss / len(loader), auc


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not args.no_wandb:
        import wandb
        run_tag = args.compression if args.dataset == "ff" else args.dataset
        wandb.init(project=args.wandb_project,
                   name=f"{args.model}_{run_tag}",
                   config=vars(args))

    if args.split_dir:
        train_loader, val_loader, test_loader = get_dataloaders_from_split_dir(
            split_dir=args.split_dir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            balanced_train=args.balanced_train,
        )
        save_tag = Path(args.split_dir).name or "splitdir"
    elif args.split_csv:
        train_loader, val_loader, test_loader = get_dataloaders_from_split_csv(
            split_csv=args.split_csv,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            data_root=args.data_root,
        )
        save_tag = Path(args.split_csv).stem
    elif args.dataset == "ff":
        train_loader, val_loader, test_loader = get_ff_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            compression=args.compression,
            seed=args.seed,
        )
        save_tag = args.compression
    else:
        train_loader, val_loader, test_loader = get_celebdf_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            max_frames_per_video=args.max_frames_per_video,
            max_videos_per_class=args.max_videos_per_class,
            seed=args.seed,
        )
        save_tag = "celebdf"

    model = build_model(args.model, pretrained=not args.no_pretrained,
                        dropout=args.dropout, device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = -1.0
    best_path = save_dir / f"best_{args.model}_{save_tag}.pth"

    for epoch in range(1, args.epochs + 1):
        t = time.time()
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"\nEpoch {epoch}/{args.epochs} ({time.time()-t:.1f}s)")
        print(f"  Train -> Loss: {train_loss:.4f} | AUC: {train_auc:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}   | AUC: {val_auc:.4f}")

        if not args.no_wandb:
            wandb.log({"epoch": epoch, "train/loss": train_loss,
                       "train/auc": train_auc, "val/loss": val_loss,
                       "val/auc": val_auc, "lr": scheduler.get_last_lr()[0]})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"epoch": epoch, "model_name": args.model,
                        "model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "args": vars(args)}, best_path)
            print(f"  Best model saved - Val AUC: {val_auc:.4f}")

    print("\n" + "="*50)
    test_loss, test_auc = evaluate(model, test_loader, criterion, device)
    print(f"Test AUC: {test_auc:.4f} | Test Loss: {test_loss:.4f}")
    if not args.no_wandb:
        wandb.log({"test/loss": test_loss, "test/auc": test_auc})
        wandb.finish()
    if best_path.exists(): 
        print(f"Best Val AUC: {best_val_auc:.4f} | Saved: {best_path}")
    else:
        print(f"Best Val AUC: {best_val_auc:.4f} | No checkpoint saved")


if __name__ == "__main__":
    main()  
