"""
src/make_dataset_splits.py
Organize a dataset into separate train.csv / val.csv / test.csv splits.

Scans a data root laid out as:
    <root>/real/**/*.jpg|png   (label 0)
    <root>/fake/**/*.jpg|png   (label 1)
(the folder names bonafide/genuine and spoof/synthetic are also accepted)

and writes three CSV files, each with columns `path,label`, using a
stratified split so class balance is preserved across train/val/test.

Usage:
    python -m src.make_dataset_splits --data_root data/processed/celeb-df-v2
    python -m src.make_dataset_splits --data_root data/faces --output_dir data/splits \
                                      --val_split 0.15 --test_split 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from src.dataset import stratified_split_samples
from src.logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_REAL_NAMES = {"real", "bonafide", "genuine", "original"}
_FAKE_NAMES = {"fake", "spoof", "synthetic", "manipulated"}


def scan_samples(data_root: Path) -> List[Tuple[str, int]]:
    """Collect (path, label) pairs from real/ and fake/ subfolders."""
    data_root = Path(data_root)
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    samples: List[Tuple[str, int]] = []
    for sub in sorted(p for p in data_root.iterdir() if p.is_dir()):
        name = sub.name.lower()
        if name in _REAL_NAMES:
            label = 0
        elif name in _FAKE_NAMES:
            label = 1
        else:
            logger.warning("Skipping unrecognized class folder: %s", sub.name)
            continue
        for f in sorted(sub.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                samples.append((str(f), label))

    n_real = sum(1 for _, l in samples if l == 0)
    n_fake = sum(1 for _, l in samples if l == 1)
    logger.info("Scanned %d samples (%d real, %d fake)", len(samples), n_real, n_fake)
    if not samples:
        raise SystemExit(
            f"No images found under {data_root}. Expected real/ and fake/ subfolders."
        )
    return samples


def _write_csv(path: Path, rows: List[Tuple[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)


def make_splits(data_root: str, output_dir: str = None, val_split: float = 0.15,
                test_split: float = 0.15, seed: int = 42) -> dict:
    """Create train.csv / val.csv / test.csv. Returns per-split counts."""
    data_root = Path(data_root)
    output_dir = Path(output_dir) if output_dir else data_root
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = scan_samples(data_root)
    train, val, test = stratified_split_samples(
        samples, val_split=val_split, test_split=test_split, seed=seed
    )

    counts = {}
    for name, rows in (("train", train), ("val", val), ("test", test)):
        out = output_dir / f"{name}.csv"
        _write_csv(out, [(p, l) for p, l in rows])
        counts[name] = len(rows)
        logger.info("Wrote %s (%d rows) -> %s", name, len(rows), out)

    return counts


def parse_args():
    p = argparse.ArgumentParser(description="Create train/val/test CSV splits")
    p.add_argument("--data_root", required=True, help="Root with real/ and fake/ subfolders")
    p.add_argument("--output_dir", default=None, help="Where to write the CSVs (default: data_root)")
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    counts = make_splits(args.data_root, args.output_dir,
                         args.val_split, args.test_split, args.seed)
    total = sum(counts.values())
    print(f"\nDone. train={counts['train']} val={counts['val']} test={counts['test']} "
          f"(total {total})")


if __name__ == "__main__":
    main()
