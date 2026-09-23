"""
scripts/cache_spectrograms.py
Pre-compute log-mel spectrograms for all audio files and save as .npy files.
This avoids repeated librosa.load() + mel computation during training.

Usage:
    python scripts/cache_spectrograms.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.audio_dataset import (
    load_waveform, waveform_to_input, SAMPLE_RATE
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_CSV = PROJECT_ROOT / "data" / "audio_split.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "audio_cache"


def main():
    print("=" * 60)
    print("Spectrogram Caching")
    print("=" * 60)

    df = pd.read_csv(SPLIT_CSV)
    print(f"Loaded {len(df)} samples from {SPLIT_CSV}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print(f"Labels: {df['label'].value_counts().to_dict()}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "train").mkdir(exist_ok=True)
    (CACHE_DIR / "val").mkdir(exist_ok=True)
    (CACHE_DIR / "test").mkdir(exist_ok=True)

    cache_rows = []
    total = len(df)
    errors = 0
    start = time.time()

    for idx, row in df.iterrows():
        src_path = row["path"]
        label = row["label"]
        split = row["split"]

        basename = Path(src_path).stem
        cache_path = CACHE_DIR / split / f"{basename}.npy"

        if cache_path.exists():
            cache_rows.append({
                "cache_path": str(cache_path),
                "label": label,
                "split": split,
            })
            done = len(cache_rows)
            if done % 5000 == 0:
                elapsed = time.time() - start
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed > 0 else 0
                print(f"  [{done}/{total}] {done/total*100:.1f}% | ETA: {eta:.0f}s (cached)")
            continue

        try:
            y = load_waveform(src_path, sr=SAMPLE_RATE)
            spec = waveform_to_input(y, sr=SAMPLE_RATE)
            np.save(cache_path, spec.astype(np.float16))
            cache_rows.append({
                "cache_path": str(cache_path),
                "label": label,
                "split": split,
            })
        except Exception as e:
            errors += 1
            print(f"  ERROR [{errors}] {src_path}: {e}")

        done = len(cache_rows)
        if done % 500 == 0:
            elapsed = time.time() - start
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / speed if speed > 0 else 0
            pct = done / total * 100
            print(f"  [{done}/{total}] {pct:.1f}% | {speed:.1f} files/s | ETA: {eta:.0f}s")

    index_df = pd.DataFrame(cache_rows)
    index_path = CACHE_DIR / "cache_index.csv"
    index_df.to_csv(index_path, index=False)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Cached: {len(cache_rows)} files | Errors: {errors}")
    print(f"Index saved: {index_path}")
    print(f"Cache dir: {CACHE_DIR}")

    cache_size = sum(f.stat().st_size for f in CACHE_DIR.rglob("*.npy"))
    print(f"Cache size: {cache_size / (1024**3):.2f} GB")

    print(f"\nSplit counts:")
    print(index_df["split"].value_counts().to_string())

    print(f"\nNext step:")
    print(f"  python -m src.audio_train --split_csv data/audio_cache/cache_index.csv --model cnn --epochs 10 --batch_size 16 --num_workers 0 --cache_dir data/audio_cache")


if __name__ == "__main__":
    main()
