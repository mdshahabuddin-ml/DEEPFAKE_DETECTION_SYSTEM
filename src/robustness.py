"""
src/robustness.py
Adversarial / degradation robustness benchmark.

Re-evaluates the detector on a set of face images after applying real-world
corruptions (JPEG compression, blur, noise, crop, resize) and reports how much
predictions shift — the kind of stress test a research lab runs before claiming
a model is deployable.

Data layout (label optional but enables accuracy metrics):
    <root>/real/*.jpg   (label 0)
    <root>/fake/*.jpg   (label 1)
    or just <root>/*.jpg (unlabeled -> reports label-flip rate only)

Usage:
    python -m src.robustness --data_root data/robustness_eval
    python -m src.robustness --data_root data/faces --max_images 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import DEFAULT_THRESHOLD
from src.inference import ModelNotFoundError, get_detector
from src.logger import get_logger
from src.results_writer import save_batch_csv

logger = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Corruptions (operate on BGR uint8 images)
# ---------------------------------------------------------------------------
def _jpeg_compress(img: np.ndarray, quality: int = 30) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def _gaussian_blur(img: np.ndarray, k: int = 7) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), 0)


def _gaussian_noise(img: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _center_crop(img: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def _downscale(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    return cv2.resize(small, (w, h))  # scale back up to lose detail


CORRUPTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "clean": lambda x: x,
    "jpeg_q30": _jpeg_compress,
    "blur_7x7": _gaussian_blur,
    "noise_15": _gaussian_noise,
    "crop_80": _center_crop,
    "downscale_0.5": _downscale,
}


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------
def _load_samples(root: Path, max_images: Optional[int]) -> List[Tuple[Path, Optional[int]]]:
    root = Path(root)
    samples: List[Tuple[Path, Optional[int]]] = []
    labeled = False
    for name, label in (("real", 0), ("fake", 1)):
        sub = root / name
        if sub.exists():
            labeled = True
            for f in sorted(sub.rglob("*")):
                if f.suffix.lower() in IMAGE_EXTS:
                    samples.append((f, label))
    if not labeled:
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                samples.append((f, None))
    if max_images:
        samples = samples[:max_images]
    return samples


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(data_root: str, max_images: Optional[int] = None,
                  threshold: float = DEFAULT_THRESHOLD) -> List[dict]:
    """
    Evaluate every corruption over the dataset and return one summary row per
    corruption (mean fake prob, accuracy if labeled, and label-flip rate vs clean).
    """
    detector = get_detector()
    detector.threshold = threshold
    samples = _load_samples(Path(data_root), max_images)
    if not samples:
        raise SystemExit(f"No images found under {data_root} (expected real/ fake/ or *.jpg).")

    logger.info("Robustness benchmark on %d images x %d corruptions",
                len(samples), len(CORRUPTIONS))

    # Cache the clean prediction per image to measure label flips.
    clean_labels: Dict[Path, Optional[int]] = {}
    rows: List[dict] = []

    for cname, fn in CORRUPTIONS.items():
        probs, correct, total, flips, valid = [], 0, 0, 0, 0
        for path, label in samples:
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            corrupted = fn(bgr)
            res = detector.predict_image_array(corrupted)
            prob = res.get("fake_probability")
            if prob is None:  # no face after corruption
                continue
            valid += 1
            probs.append(prob)
            pred = 1 if prob >= threshold else 0

            if cname == "clean":
                clean_labels[path] = pred
            else:
                base = clean_labels.get(path)
                if base is not None and base != pred:
                    flips += 1

            if label is not None:
                total += 1
                correct += int(pred == label)

        row = {
            "corruption": cname,
            "images_scored": valid,
            "mean_fake_prob": round(float(np.mean(probs)), 4) if probs else None,
            "accuracy": round(correct / total, 4) if total else None,
            "label_flip_rate_vs_clean": (None if cname == "clean"
                                         else round(flips / valid, 4) if valid else None),
        }
        rows.append(row)
        logger.info("  %-14s acc=%s mean_fake=%s flips=%s",
                    cname, row["accuracy"], row["mean_fake_prob"],
                    row["label_flip_rate_vs_clean"])

    return rows


def _print_table(rows: List[dict]) -> None:
    print(f"\n{'Corruption':<15}{'Scored':>8}{'MeanFake':>10}{'Accuracy':>10}{'FlipRate':>10}")
    print("-" * 53)
    for r in rows:
        acc = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "  n/a"
        mf = f"{r['mean_fake_prob']:.3f}" if r["mean_fake_prob"] is not None else "  n/a"
        fr = (f"{r['label_flip_rate_vs_clean']:.3f}"
              if r["label_flip_rate_vs_clean"] is not None else "  -")
        print(f"{r['corruption']:<15}{r['images_scored']:>8}{mf:>10}{acc:>10}{fr:>10}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Adversarial/degradation robustness benchmark")
    p.add_argument("--data_root", required=True, help="Folder with real/ fake/ or *.jpg")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        rows = run_benchmark(args.data_root, args.max_images, args.threshold)
    except ModelNotFoundError as exc:
        raise SystemExit(str(exc))
    _print_table(rows)
    csv_path = save_batch_csv(rows, prefix="robustness")
    print(f"Saved robustness report: {csv_path}")


if __name__ == "__main__":
    main()
