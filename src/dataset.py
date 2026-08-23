
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


def load_celebdf_samples(
    data_root,
    max_frames_per_video=20,
    max_videos_per_class=None,
):
    """
    Load Celeb-DF v2 directly from MP4 videos.

    Structure:
        data_root/
            Celeb-real/
                *.mp4
            YouTube-real/
                *.mp4
            Celeb-synthesis/
                *.mp4

    REAL = 0
    FAKE = 1

    Frames are extracted temporarily into memory and written to
    cache files so the original Celeb-DF videos are not modified.
    """

    import cv2
    import hashlib

    samples = []
    data_root = Path(data_root)

    label_dirs = [
        ("Celeb-real", 0),
        ("YouTube-real", 0),
        ("Celeb-synthesis", 1),
    ]

    video_count = {0: 0, 1: 0}

    # Cache directory inside project
    cache_root = data_root / "_frames_cache"

    for folder_name, label in label_dirs:

        directory = data_root / folder_name

        if not directory.exists():
            print(f"[WARNING] Not found: {directory}")
            continue

        videos = sorted(directory.glob("*.mp4"))

        print(
            f"Celeb-DF {folder_name}: "
            f"{len(videos)} videos found"
        )

        for video_path in videos:

            if (
                max_videos_per_class is not None
                and video_count[label] >= max_videos_per_class
            ):
                break

            # ------------------------------------------------
            # Create unique cache directory for this video
            # ------------------------------------------------

            video_hash = hashlib.md5(
                str(video_path).encode("utf-8")
            ).hexdigest()[:12]

            video_cache = (
                cache_root /
                folder_name /
                video_hash
            )

            video_cache.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Open video
            # ------------------------------------------------

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print(
                    f"[WARNING] Could not open: {video_path}"
                )
                cap.release()
                continue

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if total_frames <= 0:
                cap.release()
                continue

            # ------------------------------------------------
            # Select evenly spaced frames
            # ------------------------------------------------

            frame_indices = np.linspace(
                0,
                total_frames - 1,
                min(
                    max_frames_per_video,
                    total_frames
                ),
                dtype=int
            )

            extracted = 0

            for frame_idx in frame_indices:

                cache_file = (
                    video_cache /
                    f"frame_{frame_idx:06d}.jpg"
                )

                # Already extracted
                if cache_file.exists():
                    samples.append(
                        (str(cache_file), label)
                    )
                    extracted += 1
                    continue

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(frame_idx)
                )

                success, frame = cap.read()

                if not success:
                    continue

                # Save extracted frame
                cv2.imwrite(
                    str(cache_file),
                    frame
                )

                samples.append(
                    (str(cache_file), label)
                )

                extracted += 1

            cap.release()

            if extracted > 0:
                video_count[label] += 1

    real_count = sum(
        label == 0
        for _, label in samples
    )

    fake_count = sum(
        label == 1
        for _, label in samples
    )

    print(
        f"Celeb-DF v2: "
        f"{real_count} real, "
        f"{fake_count} fake"
    )

    print(
        f"Videos used: "
        f"REAL={video_count[0]}, "
        f"FAKE={video_count[1]}"
    )

    print(
        f"Total frames: {len(samples)}"
    )

    return samples
# ============================================================
# FaceForensics++ Extracted Faces Loader
# ============================================================

def load_ff_faces_samples(data_root, max_frames_per_video=20):
    """
    Load already-extracted FaceForensics++ frames.

    Expected structure:

        data_root/
            real/
                video_1/
                    frame_001.png
                    frame_002.png
                video_2/
                    ...
            fake/
                video_1/
                    frame_001.png
                    ...

    Returns:
        List of (image_path, label)

        REAL = 0
        FAKE = 1
    """

    root = Path(data_root)
    samples = []

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    for class_name, label in [("real", 0), ("fake", 1)]:

        class_dir = root / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"FF++ directory not found: {class_dir}"
            )

        for video_dir in sorted(class_dir.iterdir()):

            if not video_dir.is_dir():
                continue

            frames = [
                f for f in sorted(video_dir.iterdir())
                if f.is_file() and f.suffix.lower() in extensions
            ]

            for frame in frames[:max_frames_per_video]:
                samples.append((str(frame), label))

    real_count = sum(label == 0 for _, label in samples)
    fake_count = sum(label == 1 for _, label in samples)

    print("FaceForensics++ extracted dataset loaded:")
    print(f"  REAL: {real_count}")
    print(f"  FAKE: {fake_count}")
    print(f"  TOTAL: {len(samples)}")

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
def load_indian_samples(data_root):
    """
    Load Indian Deepfake Face dataset.

    Expected structure:
        data_root/
            real/
            fake/

    Returns:
        List of (image_path, label, video_id)
        REAL = 0
        FAKE = 1
    """
    from pathlib import Path

    root = Path(data_root)
    samples = []

    real_dir = root / "real"
    fake_dir = root / "fake"

    if not real_dir.exists():
        raise FileNotFoundError(f"REAL directory not found: {real_dir}")

    if not fake_dir.exists():
        raise FileNotFoundError(f"FAKE directory not found: {fake_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # REAL = 0
    for path in real_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            samples.append((str(path), 0))

    # FAKE = 1
    for path in fake_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            samples.append((str(path), 1))

    print("Indian dataset loaded:")
    print(f"  REAL: {sum(label == 0 for _, label in samples)}")
    print(f"  FAKE: {sum(label == 1 for _, label in samples)}")
    print(f"  TOTAL: {len(samples)}")

    return samples
# ============================================================
# WildDeepfake Dataset
# ============================================================

def load_wilddeepfake_samples(data_root, max_frames_per_video=20):
    """
    Load WildDeepfake frame dataset.

    Expected structure:

        data_root/
            fake/
                fake_vid_0/
                    frame_0.png
                    frame_1.png
                    ...
                fake_vid_1/
                    ...
            real/
                real_vid_0/
                    frame_0.png
                    frame_1.png
                    ...
                real_vid_1/
                    ...

    Labels:
        REAL = 0
        FAKE = 1

    Returns:
        List of (image_path, label)
    """

    root = Path(data_root)
    samples = []

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    # --------------------------------------------------------
    # REAL videos
    # --------------------------------------------------------

    real_dir = root / "real"

    if not real_dir.exists():
        raise FileNotFoundError(
            f"WildDeepfake REAL directory not found: {real_dir}"
        )

    for video_dir in sorted(real_dir.iterdir()):

        if not video_dir.is_dir():
            continue

        frames = [
            p for p in sorted(video_dir.iterdir())
            if p.is_file() and p.suffix.lower() in extensions
        ]

        # Limit number of frames per video
        frames = frames[:max_frames_per_video]

        for frame_path in frames:
            samples.append((str(frame_path), 0))

    # --------------------------------------------------------
    # FAKE videos
    # --------------------------------------------------------

    fake_dir = root / "fake"

    if not fake_dir.exists():
        raise FileNotFoundError(
            f"WildDeepfake FAKE directory not found: {fake_dir}"
        )

    for video_dir in sorted(fake_dir.iterdir()):

        if not video_dir.is_dir():
            continue

        frames = [
            p for p in sorted(video_dir.iterdir())
            if p.is_file() and p.suffix.lower() in extensions
        ]

        # Limit number of frames per video
        frames = frames[:max_frames_per_video]

        for frame_path in frames:
            samples.append((str(frame_path), 1))

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    real_count = sum(
        label == 0 for _, label in samples
    )

    fake_count = sum(
        label == 1 for _, label in samples
    )

    print("WildDeepfake dataset loaded:")
    print(f"  REAL frames: {real_count}")
    print(f"  FAKE frames: {fake_count}")
    print(f"  TOTAL frames: {len(samples)}")

    return samples
# ============================================================
# DFDC Dataset
# ============================================================

def load_dfdc_samples(
    data_root,
    metadata_path,
    max_frames_per_video=20
):
    """
    Load DFDC extracted frames using metadata.json.

    data_root:
        Directory containing one folder per video.

    metadata_path:
        Original DFDC metadata.json.

    Returns:
        List of (image_path, label)

        REAL = 0
        FAKE = 1
    """

    import json

    root = Path(data_root)
    metadata_file = Path(metadata_path)

    if not root.exists():
        raise FileNotFoundError(
            f"DFDC frame directory not found: {root}"
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"DFDC metadata not found: {metadata_file}"
        )

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    samples = []

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    for video_name, info in metadata.items():

        video_id = Path(video_name).stem
        label_name = str(info.get("label", "")).upper()

        if label_name == "REAL":
            label = 0
        elif label_name == "FAKE":
            label = 1
        else:
            continue

        video_dir = root / video_id

        if not video_dir.exists():
            continue

        frames = [
            p for p in sorted(video_dir.iterdir())
            if p.is_file()
            and p.suffix.lower() in extensions
        ]

        frames = frames[:max_frames_per_video]

        for frame_path in frames:
            samples.append(
                (str(frame_path), label)
            )

    real_count = sum(
        label == 0 for _, label in samples
    )

    fake_count = sum(
        label == 1 for _, label in samples
    )

    print("DFDC dataset loaded:")
    print(f"  REAL frames: {real_count}")
    print(f"  FAKE frames: {fake_count}")
    print(f"  TOTAL frames: {len(samples)}")

    return samples
# ============================================================
# Balanced Multi-Dataset Sample Loader
# ============================================================

def get_multidataset_samples(
    indian_root,
    wild_root,
    dfdc_root,
    dfdc_metadata,
    ff_root=None,
    indian_fake_per_real=None,
    dfdc_fake_per_real=None,
    max_frames_per_video=20,
    seed=42,
):
    """
    Load and balance multiple deepfake datasets.

    Returns:
        List of (image_path, label)

    REAL = 0
    FAKE = 1

    Indian dataset is read directly from its original location.
    No files are copied.
    """

    import random

    rng = random.Random(seed)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    indian = load_indian_samples(indian_root)

    wild = load_wilddeepfake_samples(
        wild_root,
        max_frames_per_video=max_frames_per_video
    )

    dfdc = load_dfdc_samples(
        dfdc_root,
        dfdc_metadata,
        max_frames_per_video=max_frames_per_video
    )

    # --------------------------------------------------------
    # Optional FaceForensics++ loader
    # --------------------------------------------------------

    ff = []

    if ff_root is not None:
        ff_root = Path(ff_root)

        real_dir = ff_root / "real"
        fake_dir = ff_root / "fake"

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }

        if real_dir.exists():
            for p in real_dir.rglob("*"):
                if (
                    p.is_file()
                    and p.suffix.lower() in extensions
                ):
                    ff.append((str(p), 0))

        if fake_dir.exists():
            for p in fake_dir.rglob("*"):
                if (
                    p.is_file()
                    and p.suffix.lower() in extensions
                ):
                    ff.append((str(p), 1))

    # --------------------------------------------------------
    # Balance each dataset
    # --------------------------------------------------------

    def balance_dataset(samples, name):
        real = [
            s for s in samples
            if s[1] == 0
        ]

        fake = [
            s for s in samples
            if s[1] == 1
        ]

        n = min(len(real), len(fake))

        if n == 0:
            print(f"{name}: no balanced samples available")
            return []

        rng.shuffle(real)
        rng.shuffle(fake)

        balanced = real[:n] + fake[:n]
        rng.shuffle(balanced)

        print(
            f"{name}: "
            f"{n} REAL + {n} FAKE = "
            f"{len(balanced)}"
        )

        return balanced

    indian = balance_dataset(
        indian,
        "Indian"
    )

    wild = balance_dataset(
        wild,
        "WildDeepfake"
    )

    dfdc = balance_dataset(
        dfdc,
        "DFDC"
    )

    if ff:
        ff = balance_dataset(
            ff,
            "FaceForensics++"
        )

    # --------------------------------------------------------
    # Combine datasets
    # --------------------------------------------------------

    combined = []

    combined.extend(indian)
    combined.extend(wild)
    combined.extend(dfdc)
    combined.extend(ff)

    rng.shuffle(combined)

    real_count = sum(
        label == 0
        for _, label in combined
    )

    fake_count = sum(
        label == 1
        for _, label in combined
    )

    print("\nMulti-dataset summary:")
    print(f"  REAL:  {real_count}")
    print(f"  FAKE:  {fake_count}")
    print(f"  TOTAL: {len(combined)}")

    return combined 
# ============================================================
# Video-Level Split for Multi-Dataset Samples
# ============================================================

def split_multidataset_samples(
    samples,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Split samples while keeping frames from the same video together.

    Expected sample format:
        (image_path, label)

    For paths containing video/frame directories, the video ID is
    inferred from the parent directory.

    Returns:
        train_samples, val_samples, test_samples
    """

    import random
    from collections import defaultdict

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        )

    rng = random.Random(seed)

    # --------------------------------------------------------
    # Group samples by video/folder
    # --------------------------------------------------------

    groups = defaultdict(list)

    for path, label in samples:

        p = Path(path)

        # For frame datasets:
        # parent directory normally represents the video.
        video_id = p.parent.name

        # Keep label separate so a folder cannot mix labels.
        key = (video_id, label)

        groups[key].append((path, label))

    # --------------------------------------------------------
    # Separate REAL and FAKE groups
    # --------------------------------------------------------

    real_groups = [
        group for (video_id, label), group in groups.items()
        if label == 0
    ]

    fake_groups = [
        group for (video_id, label), group in groups.items()
        if label == 1
    ]

    rng.shuffle(real_groups)
    rng.shuffle(fake_groups)

    def split_groups(group_list):

        n = len(group_list)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = group_list[:train_end]
        val = group_list[train_end:val_end]
        test = group_list[val_end:]

        return train, val, test

    real_train, real_val, real_test = split_groups(real_groups)
    fake_train, fake_val, fake_test = split_groups(fake_groups)

    # --------------------------------------------------------
    # Flatten groups
    # --------------------------------------------------------

    def flatten(groups_list):
        result = []

        for group in groups_list:
            result.extend(group)

        return result

    train_samples = (
        flatten(real_train) +
        flatten(fake_train)
    )

    val_samples = (
        flatten(real_val) +
        flatten(fake_val)
    )

    test_samples = (
        flatten(real_test) +
        flatten(fake_test)
    )

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    def stats(name, data):
        real = sum(label == 0 for _, label in data)
        fake = sum(label == 1 for _, label in data)

        print(
            f"{name}: "
            f"REAL={real}, "
            f"FAKE={fake}, "
            f"TOTAL={len(data)}"
        )

    print("\nVideo-level split:")
    stats("TRAIN", train_samples)
    stats("VAL", val_samples)
    stats("TEST", test_samples)

    return train_samples, val_samples, test_samples

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

# ============================================================
# Multi-Dataset Train / Validation / Test Split
# ============================================================

def get_multidataset_splits(
    indian_root,
    wild_root,
    dfdc_root,
    dfdc_metadata,
    ff_root,
    celebdf_root,
    max_frames_per_video=20,
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
):
    """
    Build balanced train/validation/test splits from:

        Indian Deepfake Face
        WildDeepfake
        DFDC
        FaceForensics++ extracted frames
        Celeb-DF v2

    Returns:
        train_samples, val_samples, test_samples

    Each sample:
        (image_path, label)

    REAL = 0
    FAKE = 1
    """

    import random
    from collections import defaultdict

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        )

    rng = random.Random(seed)

    print("\n" + "=" * 60)
    print("Loading multi-dataset training data")
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    indian = load_indian_samples(indian_root)

    wild = load_wilddeepfake_samples(
        wild_root,
        max_frames_per_video=max_frames_per_video
    )

    dfdc = load_dfdc_samples(
        dfdc_root,
        dfdc_metadata,
        max_frames_per_video=max_frames_per_video
    )

    # IMPORTANT:
    # Use the tested extracted FaceForensics++ loader.
    ff = load_ff_faces_samples(
        ff_root,
        max_frames_per_video=max_frames_per_video
    )

    celebdf = load_celebdf_samples(
        celebdf_root,
        max_frames_per_video=max_frames_per_video
    )

    # --------------------------------------------------------
    # Balance each dataset
    # --------------------------------------------------------

    def balance_samples(samples, dataset_name):
        real = [sample for sample in samples if sample[1] == 0]
        fake = [sample for sample in samples if sample[1] == 1]

        n = min(len(real), len(fake))

        if n == 0:
            raise ValueError(
                f"{dataset_name} does not contain both REAL and FAKE samples."
            )

        rng.shuffle(real)
        rng.shuffle(fake)

        balanced = real[:n] + fake[:n]
        rng.shuffle(balanced)

        print(
            f"{dataset_name}: "
            f"REAL={len(real[:n])}, "
            f"FAKE={len(fake[:n])}, "
            f"TOTAL={len(balanced)}"
        )

        return balanced

    indian = balance_samples(indian, "Indian")
    wild = balance_samples(wild, "WildDeepfake")
    dfdc = balance_samples(dfdc, "DFDC")
    ff = balance_samples(ff, "FaceForensics++")
    celebdf = balance_samples(celebdf, "Celeb-DF v2")

    # --------------------------------------------------------
    # Dataset-specific video-level grouping
    # --------------------------------------------------------

    def split_dataset_samples(samples, dataset_name):
        groups = defaultdict(list)

        for path, label in samples:
            p = Path(path)

            if dataset_name in {
                "DFDC",
                "WildDeepfake",
                "FaceForensics++",
            }:
                # Frames in the same video directory stay together.
                group_id = str(p.parent)
            else:
                # For image-based datasets, keep each image as a group.
                group_id = str(p)

            groups[(group_id, label)].append((path, label))

        real_groups = [
            group
            for (group_id, label), group in groups.items()
            if label == 0
        ]

        fake_groups = [
            group
            for (group_id, label), group in groups.items()
            if label == 1
        ]

        rng.shuffle(real_groups)
        rng.shuffle(fake_groups)

        def split_groups(group_list):
            n = len(group_list)

            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            return (
                group_list[:train_end],
                group_list[train_end:val_end],
                group_list[val_end:],
            )

        real_train, real_val, real_test = split_groups(real_groups)
        fake_train, fake_val, fake_test = split_groups(fake_groups)

        def flatten(groups):
            result = []
            for group in groups:
                result.extend(group)
            return result

        train = flatten(real_train) + flatten(fake_train)
        val = flatten(real_val) + flatten(fake_val)
        test = flatten(real_test) + flatten(fake_test)

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        print(
            f"{dataset_name} split: "
            f"TRAIN={len(train)}, "
            f"VAL={len(val)}, "
            f"TEST={len(test)}"
        )

        return train, val, test

    # --------------------------------------------------------
    # Split every dataset separately
    # --------------------------------------------------------

    datasets = [
        ("Indian", indian),
        ("WildDeepfake", wild),
        ("DFDC", dfdc),
        ("FaceForensics++", ff),
        ("Celeb-DF", celebdf),
    ]

    train_samples = []
    val_samples = []
    test_samples = []

    for dataset_name, samples in datasets:
        train, val, test = split_dataset_samples(
            samples,
            dataset_name
        )

        train_samples.extend(train)
        val_samples.extend(val)
        test_samples.extend(test)

    # --------------------------------------------------------
    # Final class balancing
    # --------------------------------------------------------

    def final_balance(samples):
        real = [sample for sample in samples if sample[1] == 0]
        fake = [sample for sample in samples if sample[1] == 1]

        n = min(len(real), len(fake))

        rng.shuffle(real)
        rng.shuffle(fake)

        result = real[:n] + fake[:n]
        rng.shuffle(result)

        return result

    train_samples = final_balance(train_samples)
    val_samples = final_balance(val_samples)
    test_samples = final_balance(test_samples)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def show_stats(name, samples):
        real = sum(label == 0 for _, label in samples)
        fake = sum(label == 1 for _, label in samples)

        print(
            f"{name}: "
            f"REAL={real}, "
            f"FAKE={fake}, "
            f"TOTAL={len(samples)}"
        )

    print("\n" + "=" * 60)
    print("FINAL MULTI-DATASET SPLIT")
    print("=" * 60)

    show_stats("TRAIN", train_samples)
    show_stats("VAL", val_samples)
    show_stats("TEST", test_samples)

    print("=" * 60)

    return train_samples, val_samples, test_samples