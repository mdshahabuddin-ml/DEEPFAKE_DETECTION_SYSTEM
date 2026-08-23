"""
src/train.py

Training loop for deepfake detection with W&B logging.

Supported datasets:
    ff
    celebdf
    dfdc
    multidataset

Multi-dataset training:
    Indian Deepfake Face
    WildDeepfake
    DFDC
    FaceForensics++
    Celeb-DF v2

Example:

    python src/train.py ^
        --model xception ^
        --dataset multidataset ^
        --epochs 1 ^
        --batch_size 16 ^
        --num_workers 2 ^
        --no_wandb

PowerShell can also use the command on one line.
"""

import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import roc_auc_score


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Dataset imports
# ============================================================

from src.dataset import (
    DeepfakeDataset,
    get_train_transforms,
    get_val_transforms,

    get_ff_dataloaders,
    get_celebdf_dataloaders,

    get_dataloaders_from_split_csv,
    get_dataloaders_from_split_dir,

    get_multidataset_splits,
)


# ============================================================
# Model
# ============================================================

from src.model import build_model


# ============================================================
# Argument parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Deepfake detection training"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        default="xception",
        choices=["xception", "efficientnet", "vit"],
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--dataset",
        type=str,
        default="ff",
        choices=[
            "ff",
            "celebdf",
            "dfdc",
            "multidataset",
        ],
    )

    # --------------------------------------------------------
    # Existing dataset arguments
    # --------------------------------------------------------

    parser.add_argument(
        "--data_root",
        type=str,
        default="data/FaceForensics",
    )

    parser.add_argument(
        "--compression",
        type=str,
        default="c23",
    )

    # --------------------------------------------------------
    # Multi-dataset paths
    # --------------------------------------------------------

    parser.add_argument(
        "--indian_root",
        type=str,
        default=r"C:\Users\mdsha\Documents\Indian-deepfake-face",
        help="Path to Indian Deepfake Face dataset",
    )

    parser.add_argument(
        "--wild_root",
        type=str,
        default=r"C:\Users\mdsha\OneDrive\Desktop\Wild Deepfake Dataset (2)\thesis_test_frames",
        help="Path to WildDeepfake extracted frames",
    )

    parser.add_argument(
        "--dfdc_root",
        type=str,
        default=r"C:\Users\mdsha\OneDrive\Desktop\deepfake_detection_system\data\dfdc_faces\DFDC Dtata_set_kaggle",
        help="Path to DFDC extracted faces",
    )

    parser.add_argument(
        "--dfdc_metadata",
        type=str,
        default=r"C:\Users\mdsha\OneDrive\Desktop\DFDC_dataset\DFDC Dtata_set_kaggle\metadata.json",
        help="Path to DFDC metadata.json",
    )

    parser.add_argument(
        "--ff_root",
        type=str,
        default=r"C:\Users\mdsha\OneDrive\Desktop\deepfake_detection_system\data\ff_faces",
        help="Path to extracted FaceForensics++ faces",
    )

    parser.add_argument(
        "--celebdf_root",
        type=str,
        default=r"C:\Users\mdsha\OneDrive\Desktop\deepfake_detection_system\data\celeb-df-v2",
        help="Path to Celeb-DF v2",
    )

    # --------------------------------------------------------
    # Dataset processing
    # --------------------------------------------------------

    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max_videos_per_class",
        type=int,
        default=None,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=299,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Train from random weights instead of pretrained weights",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------

    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
    )

    # --------------------------------------------------------
    # Existing split files
    # --------------------------------------------------------

    parser.add_argument(
        "--split_csv",
        type=str,
        default=None,
        help="CSV containing path,label,split",
    )

    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Directory containing train.csv/val.csv/test.csv",
    )

    # --------------------------------------------------------
    # W&B
    # --------------------------------------------------------

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="deepfake-detection",
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )

    # --------------------------------------------------------
    # Balanced training
    # --------------------------------------------------------

    parser.add_argument(
        "--balanced_train",
        action="store_true",
        help="Use weighted sampling during training",
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Create DataLoaders for multi-dataset split
# ============================================================

def create_multidataset_dataloaders(
    train_samples,
    val_samples,
    test_samples,
    batch_size=32,
    img_size=299,
    num_workers=4,
    balanced_train=False,
):
    """
    Create PyTorch DataLoaders from the samples returned by
    get_multidataset_splits().
    """

    print("\n" + "=" * 60)
    print("Creating multi-dataset DataLoaders")
    print("=" * 60)

    # --------------------------------------------------------
    # Existing augmentation pipeline
    # --------------------------------------------------------

    train_transform = get_train_transforms(img_size)

    val_transform = get_val_transforms(img_size)

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = DeepfakeDataset(
        train_samples,
        transform=train_transform,
    )

    val_dataset = DeepfakeDataset(
        val_samples,
        transform=val_transform,
    )

    test_dataset = DeepfakeDataset(
        test_samples,
        transform=val_transform,
    )

    # --------------------------------------------------------
    # Optional weighted sampler
    # --------------------------------------------------------

    train_sampler = None

    if balanced_train:

        labels = [
            label
            for _, label in train_samples
        ]

        class_counts = {
            0: labels.count(0),
            1: labels.count(1),
        }

        weights = []

        for label in labels:

            if class_counts[label] > 0:
                weights.append(
                    1.0 / class_counts[label]
                )
            else:
                weights.append(0.0)

        weights = torch.DoubleTensor(weights)

        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

        print("WeightedRandomSampler: ENABLED")

    else:

        print("WeightedRandomSampler: DISABLED")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:

        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # --------------------------------------------------------
    # Print loader information
    # --------------------------------------------------------

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
    )

    print(
        f"Test batches: {len(test_loader)}"
    )

    print("=" * 60)

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
):

    model.train()

    total_loss = 0.0

    all_labels = []
    all_probs = []

    for i, (images, labels) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = (
            labels
            .float()
            .unsqueeze(1)
            .to(
                device,
                non_blocking=True,
            )
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(images)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels,
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        probabilities = torch.sigmoid(
            logits
        )

        all_probs.extend(
            probabilities
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if i % 50 == 0:

            print(
                f"  Epoch {epoch} "
                f"[{i}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    try:

        auc = roc_auc_score(
            all_labels,
            all_probs,
        )

    except Exception:

        auc = 0.0

    average_loss = (
        total_loss / len(loader)
        if len(loader) > 0
        else 0.0
    )

    return average_loss, auc


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_probs = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = (
            labels
            .float()
            .unsqueeze(1)
            .to(
                device,
                non_blocking=True,
            )
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        total_loss += loss.item()

        probabilities = torch.sigmoid(
            logits
        )

        all_probs.extend(
            probabilities
            .cpu()
            .numpy()
            .flatten()
        )

        all_labels.extend(
            labels
            .cpu()
            .numpy()
            .flatten()
        )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    try:

        auc = roc_auc_score(
            all_labels,
            all_probs,
        )

    except Exception:

        auc = 0.0

    average_loss = (
        total_loss / len(loader)
        if len(loader) > 0
        else 0.0
    )

    return average_loss, auc


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(args.seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device == "cpu":

        print(
            "\nWARNING: CUDA is not available."
        )

        print(
            "Xception training will run on CPU "
            "and may be slow.\n"
        )

    else:

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # W&B
    # --------------------------------------------------------

    wandb = None

    if not args.no_wandb:

        try:

            import wandb

            run_tag = (
                "multidataset"
                if args.dataset == "multidataset"
                else (
                    args.compression
                    if args.dataset == "ff"
                    else args.dataset
                )
            )

            wandb.init(
                project=args.wandb_project,
                name=f"{args.model}_{run_tag}",
                config=vars(args),
            )

        except Exception as e:

            print(
                f"[WARNING] W&B initialization failed: {e}"
            )

            print(
                "Continuing without W&B."
            )

            wandb = None

    # ========================================================
    # Multi-dataset
    # ========================================================

    if args.dataset == "multidataset":

        print("\n")
        print("=" * 60)
        print("MULTI-DATASET TRAINING MODE")
        print("=" * 60)

        # ----------------------------------------------------
        # Build balanced video-level splits
        # ----------------------------------------------------

        (
            train_samples,
            val_samples,
            test_samples,
        ) = get_multidataset_splits(

            indian_root=args.indian_root,

            wild_root=args.wild_root,

            dfdc_root=args.dfdc_root,

            dfdc_metadata=args.dfdc_metadata,

            ff_root=args.ff_root,

            celebdf_root=args.celebdf_root,

            max_frames_per_video=args.max_frames_per_video,

            seed=args.seed,

            train_ratio=0.70,

            val_ratio=0.15,

            test_ratio=0.15,
        )

        # ----------------------------------------------------
        # Create DataLoaders
        # ----------------------------------------------------

        (
            train_loader,
            val_loader,
            test_loader,
        ) = create_multidataset_dataloaders(

            train_samples=train_samples,

            val_samples=val_samples,

            test_samples=test_samples,

            batch_size=args.batch_size,

            img_size=args.img_size,

            num_workers=args.num_workers,

            balanced_train=args.balanced_train,
        )

        save_tag = "multidataset"

    # ========================================================
    # Existing split directory
    # ========================================================

    elif args.split_dir:

        (
            train_loader,
            val_loader,
            test_loader,
        ) = get_dataloaders_from_split_dir(

            split_dir=args.split_dir,

            batch_size=args.batch_size,

            img_size=args.img_size,

            num_workers=args.num_workers,

            balanced_train=args.balanced_train,
        )

        save_tag = (
            Path(args.split_dir).name
            or "splitdir"
        )

    # ========================================================
    # Existing split CSV
    # ========================================================

    elif args.split_csv:

        (
            train_loader,
            val_loader,
            test_loader,
        ) = get_dataloaders_from_split_csv(

            split_csv=args.split_csv,

            batch_size=args.batch_size,

            img_size=args.img_size,

            num_workers=args.num_workers,

            data_root=args.data_root,
        )

        save_tag = (
            Path(args.split_csv).stem
        )

    # ========================================================
    # FaceForensics++
    # ========================================================

    elif args.dataset == "ff":

        (
            train_loader,
            val_loader,
            test_loader,
        ) = get_ff_dataloaders(

            data_root=args.data_root,

            batch_size=args.batch_size,

            img_size=args.img_size,

            num_workers=args.num_workers,

            compression=args.compression,

            seed=args.seed,
        )

        save_tag = args.compression

    # ========================================================
    # Celeb-DF / DFDC existing mode
    # ========================================================

    else:

        (
            train_loader,
            val_loader,
            test_loader,
        ) = get_celebdf_dataloaders(

            data_root=args.data_root,

            batch_size=args.batch_size,

            img_size=args.img_size,

            num_workers=args.num_workers,

            max_frames_per_video=args.max_frames_per_video,

            max_videos_per_class=args.max_videos_per_class,

            seed=args.seed,
        )

        save_tag = "celebdf"

    # ========================================================
    # Build model
    # ========================================================

    print("\n")
    print("=" * 60)
    print("Building model")
    print("=" * 60)

    model = build_model(

        args.model,

        pretrained=not args.no_pretrained,

        dropout=args.dropout,

        device=device,
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.BCEWithLogitsLoss()

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = AdamW(

        model.parameters(),

        lr=args.lr,

        weight_decay=1e-4,
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = CosineAnnealingLR(

        optimizer,

        T_max=args.epochs,

        eta_min=1e-6,
    )

    # ========================================================
    # Save directory
    # ========================================================

    save_dir = Path(
        args.save_dir
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Best model
    # ========================================================

    best_val_auc = -1.0

    best_path = (
        save_dir
        / f"best_{args.model}_{save_tag}.pth"
    )

    print(
        f"\nCheckpoint path:"
    )

    print(
        best_path
    )

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        start_time = time.time()

        print("\n")
        print("=" * 60)
        print(
            f"Epoch {epoch}/{args.epochs}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss, train_auc = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            device,

            epoch,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_auc = evaluate(

            model,

            val_loader,

            criterion,

            device,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        elapsed = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            f"\nEpoch {epoch}/{args.epochs} "
            f"({elapsed:.1f}s)"
        )

        print(
            f"  Train -> "
            f"Loss: {train_loss:.4f} | "
            f"AUC: {train_auc:.4f}"
        )

        print(
            f"  Val   -> "
            f"Loss: {val_loss:.4f} | "
            f"AUC: {val_auc:.4f}"
        )

        print(
            f"  LR: "
            f"{scheduler.get_last_lr()[0]:.8f}"
        )

        # ----------------------------------------------------
        # W&B logging
        # ----------------------------------------------------

        if wandb is not None:

            wandb.log({

                "epoch": epoch,

                "train/loss": train_loss,

                "train/auc": train_auc,

                "val/loss": val_loss,

                "val/auc": val_auc,

                "lr": scheduler.get_last_lr()[0],
            })

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_auc > best_val_auc:

            best_val_auc = val_auc

            torch.save(

                {

                    "epoch": epoch,

                    "model_name": args.model,

                    "model_state_dict":
                        model.state_dict(),

                    "val_auc": val_auc,

                    "args": vars(args),
                },

                best_path,
            )

            print(
                f"\n  Best model saved!"
            )

            print(
                f"  Val AUC: {val_auc:.4f}"
            )

            print(
                f"  Path: {best_path}"
            )

    # ========================================================
    # Final test
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL TEST")
    print("=" * 60)

    test_loss, test_auc = evaluate(

        model,

        test_loader,

        criterion,

        device,
    )

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test AUC:  {test_auc:.4f}"
    )

    # ========================================================
    # W&B final logging
    # ========================================================

    if wandb is not None:

        wandb.log({

            "test/loss": test_loss,

            "test/auc": test_auc,
        })

        wandb.finish()

    # ========================================================
    # Final checkpoint information
    # ========================================================

    print("\n")
    print("=" * 60)

    if best_path.exists():

        print(
            f"Best Val AUC: "
            f"{best_val_auc:.4f}"
        )

        print(
            f"Saved model: "
            f"{best_path}"
        )

    else:
 
        print(
            f"Best Val AUC: "
            f"{best_val_auc:.4f}"
        )

        print(
            "No checkpoint was saved." 
        ) 

    print("=" * 60)

# ============================================================
# Entry point  
# ============================================================ 
 
if __name__ == "__main__": 
    main()     