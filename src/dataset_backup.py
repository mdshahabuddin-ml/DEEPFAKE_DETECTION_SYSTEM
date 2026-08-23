"""
src/dataset.py
Dataset classes for deepfake detection.
Supports: FaceForensics++, Celeb-DF v2, DFDC
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 299) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
           A.ImageCompression(quality_range=(60, 100), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 299) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class DeepfakeDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((299, 299, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


def load_ff_samples(data_root, compression="c23", methods=None, max_frames_per_video=30):
    if methods is None:
        methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    samples = []
    data_root = Path(data_root)
    real_dir = data_root / "original_sequences" / "youtube" / compression / "frames"
    if real_dir.exists():
        for video_dir in sorted(real_dir.iterdir()):
            for f in sorted(video_dir.glob("*.png"))[:max_frames_per_video]:
                samples.append((str(f), 0))
    else:
        print(f"[WARNING] Real frames not found: {real_dir}")
    for method in methods:
        fake_dir = data_root / "manipulated_sequences" / method / compression / "frames"
        if fake_dir.exists():
            for video_dir in sorted(fake_dir.iterdir()):
                for f in sorted(video_dir.glob("*.png"))[:max_frames_per_video]:
                    samples.append((str(f), 1))
        else:
            print(f"[WARNING] Fake frames not found: {fake_dir}")
    print(f"FF++ ({compression}): {sum(1 for _,l in samples if l==0)} real, {sum(1 for _,l in samples if l==1)} fake")
    return samples


def load_celebdf_samples(data_root, max_frames_per_video=30, max_videos_per_class=None):
    samples = []
    data_root = Path(data_root)
    videos_seen = {0: 0, 1: 0}

    label_dirs = [
        ("Celeb-real", 0),
        ("YouTube-real", 0),
        ("Celeb-synthesis", 1),
    ]
    for folder_name, label in label_dirs:
        directory = data_root / folder_name
        if not directory.exists():
            print(f"[WARNING] Not found: {directory}")
            continue

        direct_frames = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))
        if direct_frames:
            if max_videos_per_class is not None and videos_seen[label] >= max_videos_per_class:
                continue
            samples.extend((str(f), label) for f in direct_frames[:max_frames_per_video])
            videos_seen[label] += 1
            continue

        for video_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
            if max_videos_per_class is not None and videos_seen[label] >= max_videos_per_class:
                break
            frames = sorted(video_dir.glob("*.png")) + sorted(video_dir.glob("*.jpg"))
            for f in frames[:max_frames_per_video]:
                samples.append((str(f), label))
            if frames:
                videos_seen[label] += 1

    print(f"Celeb-DF v2: {sum(1 for _,l in samples if l==0)} real, {sum(1 for _,l in samples if l==1)} fake")
    return samples


def split_samples(samples, val_split=0.15, test_split=0.15, seed=42):
    import random
    random.seed(seed)
    samples = list(samples)
    random.shuffle(samples)
    n = len(samples)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    test_samples = samples[:n_test]
    val_samples = samples[n_test:n_test + n_val]
    train_samples = samples[n_test + n_val:]
    return train_samples, val_samples, test_samples


def stratified_split_samples(samples, val_split=0.15, test_split=0.15, seed=42):
    import random
    rng = random.Random(seed)
    by_label = {}
    for path, label in samples:
        by_label.setdefault(label, []).append((path, label))

    train_samples, val_samples, test_samples = [], [], []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        n = len(label_samples)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        test_samples.extend(label_samples[:n_test])
        val_samples.extend(label_samples[n_test:n_test + n_val])
        train_samples.extend(label_samples[n_test + n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_samples, val_samples, test_samples


def save_split_csv(samples, output_csv, val_split=0.15, test_split=0.15,
                   seed=42, stratified=True):
    if stratified:
        train_samples, val_samples, test_samples = stratified_split_samples(
            samples, val_split=val_split, test_split=test_split, seed=seed)
    else:
        train_samples, val_samples, test_samples = split_samples(
            samples, val_split=val_split, test_split=test_split, seed=seed)

    rows = []
    for split_name, split_samples_list in [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ]:
        for path, label in split_samples_list:
            rows.append({"path": str(path), "label": int(label), "split": split_name})

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved split CSV: {output_csv}")
    print(df["split"].value_counts().to_string())
    return output_csv


def load_split_csv(split_csv, data_root=None):
    split_csv = Path(split_csv)
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    df = pd.read_csv(split_csv)
    required_cols = {"path", "label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Split CSV missing columns: {sorted(missing)}")

    data_root = Path(data_root) if data_root else None
    split_to_samples = {"train": [], "val": [], "test": []}

    for _, row in df.iterrows():
        split_name = str(row["split"]).strip().lower()
        if split_name not in split_to_samples:
            continue

        p = Path(str(row["path"]))
        if not p.is_absolute() and data_root is not None:
            # If CSV already stores project-relative paths that exist, keep them.
            # Otherwise resolve relative to data_root.
            if not p.exists():
                p = data_root / p
        split_to_samples[split_name].append((str(p), int(row["label"])))

    return split_to_samples


def load_samples_from_csv(csv_path):
    """Load (path, label) pairs from a simple CSV with columns path,label."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {sorted(missing)}")
    return [(str(r["path"]), int(r["label"])) for _, r in df.iterrows()]


def get_dataloaders_from_split_dir(split_dir, batch_size=32, img_size=299,
                                   num_workers=4, balanced_train=False):
    """
    Build train/val/test dataloaders from a directory containing separate
    train.csv, val.csv, and test.csv files (each with columns path,label).
    """
    split_dir = Path(split_dir)
    train = load_samples_from_csv(split_dir / "train.csv")
    val = load_samples_from_csv(split_dir / "val.csv")
    test = load_samples_from_csv(split_dir / "test.csv")
    return make_dataloaders(train, val, test, batch_size=batch_size,
                            img_size=img_size, num_workers=num_workers,
                            balanced_train=balanced_train)


def get_dataloaders_from_split_csv(split_csv, batch_size=32, img_size=299,
                                   num_workers=4, data_root=None):
    split_to_samples = load_split_csv(split_csv, data_root=data_root)
    return make_dataloaders(
        split_to_samples["train"],
        split_to_samples["val"],
        split_to_samples["test"],
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
    )


def make_dataloaders(train_samples, val_samples, test_samples, batch_size=32,
                     img_size=299, num_workers=4, balanced_train=False):
    train_ds = DeepfakeDataset(train_samples, transform=get_train_transforms(img_size))
    val_ds = DeepfakeDataset(val_samples, transform=get_val_transforms(img_size))
    test_ds = DeepfakeDataset(test_samples, transform=get_val_transforms(img_size))
    sampler = None
    if balanced_train:
        labels = np.asarray([label for _, label in train_samples])
        class_counts = np.bincount(labels, minlength=2)
        sample_weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        print(f"Balanced training sampler: {class_counts[0]} real, {class_counts[1]} fake")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def load_dfdc_samples(data_root, metadata_csv=None, max_frames_per_video=20):
    samples = []
    data_root = Path(data_root)
    if metadata_csv and Path(metadata_csv).exists():
        df = pd.read_csv(metadata_csv)
        for _, row in df.iterrows():
            video_frames_dir = data_root / "frames" / Path(row["filename"]).stem
            if video_frames_dir.exists():
                label = 1 if row["label"] == "FAKE" else 0
                for f in sorted(video_frames_dir.glob("*.png"))[:max_frames_per_video]:
                    samples.append((str(f), label))
    else:
        for label_name, label in [("real", 0), ("fake", 1)]:
            label_dir = data_root / "frames" / label_name
            if label_dir.exists():
                for video_dir in sorted(label_dir.iterdir()):
                    for f in sorted(video_dir.glob("*.png"))[:max_frames_per_video]:
                        samples.append((str(f), label))
    print(f"DFDC: {sum(1 for _,l in samples if l==0)} real, {sum(1 for _,l in samples if l==1)} fake")
    return samples


def get_ff_dataloaders(data_root, batch_size=32, img_size=299, num_workers=4,
                       val_split=0.15, test_split=0.15, compression="c23", seed=42):
    all_samples = load_ff_samples(data_root, compression=compression)
    train_samples, val_samples, test_samples = split_samples(
        all_samples, val_split=val_split, test_split=test_split, seed=seed)
    return make_dataloaders(
        train_samples, val_samples, test_samples,
        batch_size=batch_size, img_size=img_size, num_workers=num_workers)


def get_celebdf_dataloaders(data_root, batch_size=32, img_size=299, num_workers=4,
                            val_split=0.15, test_split=0.15,
                            max_frames_per_video=30, max_videos_per_class=None,
                            seed=42):
    all_samples = load_celebdf_samples(
        data_root,
        max_frames_per_video=max_frames_per_video,
        max_videos_per_class=max_videos_per_class)
    train_samples, val_samples, test_samples = split_samples(
        all_samples, val_split=val_split, test_split=test_split, seed=seed)
    return make_dataloaders(
        train_samples, val_samples, test_samples,
        batch_size=batch_size, img_size=img_size, num_workers=num_workers)
