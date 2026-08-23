"""Tests for train/val/test CSV split generation."""

import cv2
import numpy as np

from src.dataset import load_samples_from_csv
from src.make_dataset_splits import make_splits, scan_samples


def _make_dataset(root, n_per_class=20):
    for cls in ("real", "fake"):
        (root / cls).mkdir(parents=True)
        for i in range(n_per_class):
            img = (np.random.rand(32, 32, 3) * 255).astype("uint8")
            cv2.imwrite(str(root / cls / f"{i}.jpg"), img)


def test_scan_finds_labeled_samples(tmp_path):
    _make_dataset(tmp_path, 10)
    samples = scan_samples(tmp_path)
    assert len(samples) == 20
    assert {l for _, l in samples} == {0, 1}


def test_splits_are_disjoint_and_complete(tmp_path):
    _make_dataset(tmp_path, 20)
    counts = make_splits(str(tmp_path), seed=42)
    assert sum(counts.values()) == 40

    train = load_samples_from_csv(tmp_path / "train.csv")
    val = load_samples_from_csv(tmp_path / "val.csv")
    test = load_samples_from_csv(tmp_path / "test.csv")

    paths = [p for p, _ in train + val + test]
    assert len(paths) == 40
    assert len(set(paths)) == 40           # no overlap across splits


def test_splits_preserve_class_balance(tmp_path):
    _make_dataset(tmp_path, 20)
    make_splits(str(tmp_path), seed=42)
    train = load_samples_from_csv(tmp_path / "train.csv")
    fakes = sum(l for _, l in train)
    # Stratified -> roughly balanced (exactly half here with equal classes).
    assert fakes == len(train) // 2
