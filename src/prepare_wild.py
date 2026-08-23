"""
prepare_wild.py

Creates train.csv, val.csv and test.csv
for the Wild Deepfake Dataset.

Dataset Structure

thesis_test_frames/
│
├── fake/
│    ├── fake_vid_0/
│    ├── fake_vid_1/
│    └── ...
│
└── real/
     ├── real_vid_0/
     ├── real_vid_1/
     └── ...
"""

from pathlib import Path
import argparse
import random
import csv
import os


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}


def collect_images(folder, label):
    samples = []

    for img in folder.rglob("*"):

        if img.suffix.lower() in IMAGE_EXTENSIONS:
            samples.append((str(img.resolve()), label))

    return samples


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_csv",
        default="results/splits/wild_deepfake_split.csv"
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    random.seed(args.seed)

    root = Path(args.data_dir)

    fake_folder = root / "thesis_test_frames" / "fake"
    real_folder = root / "thesis_test_frames" / "real"

    if not fake_folder.exists():
        raise FileNotFoundError(fake_folder)

    if not real_folder.exists():
        raise FileNotFoundError(real_folder)

    fake = collect_images(fake_folder, 1)
    real = collect_images(real_folder, 0)

    print(f"Fake images : {len(fake)}")
    print(f"Real images : {len(real)}")

    samples = fake + real

    random.shuffle(samples)

    n = len(samples)

    train_end = int(n * args.train_ratio)

    val_end = train_end + int(n * args.val_ratio)

    train = samples[:train_end]

    val = samples[train_end:val_end]

    test = samples[val_end:]

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "path",
                "label",
                "split"
            ]
        )

        for p, l in train:
            writer.writerow([p, l, "train"])

        for p, l in val:
            writer.writerow([p, l, "val"])

        for p, l in test:
            writer.writerow([p, l, "test"])

    print()

    print("=" * 50)

    print("Dataset prepared successfully")

    print(f"Train : {len(train)}")

    print(f"Val   : {len(val)}")

    print(f"Test  : {len(test)}")

    print()

    print("CSV saved to")

    print(args.output_csv)


if __name__ == "__main__":
    main()