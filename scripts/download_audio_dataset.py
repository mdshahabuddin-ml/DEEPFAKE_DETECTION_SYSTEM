"""
scripts/download_audio_dataset.py
Download ASVspoof 2019 LA eval set from HuggingFace, extract to FLAC.
Uses decode=False to avoid torchcodec dependency, then decodes with soundfile.
"""

import os
import struct
import io
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "audio"
SPLIT_CSV = PROJECT_ROOT / "data" / "audio_split.csv"


def main():
    os.chdir(PROJECT_ROOT)

    import soundfile as sf
    import numpy as np
    import pandas as pd
    from datasets import load_dataset

    print("Loading ASVspoof 2019 LA eval from HuggingFace (decode=False)...")
    ds = load_dataset("SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", split="test", decode=False)
    print(f"Total samples: {len(ds)}")
    print(f"Features: {ds.features}")

    # Create output dirs
    bonafide_dir = DATA_DIR / "bonafide"
    spoof_dir = DATA_DIR / "spoof"
    bonafide_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    records = []
    n_bonafide = 0
    n_spoof = 0

    for i, sample in enumerate(ds):
        label = sample["label"]  # 0 = bonafide, 1 = spoof
        path = sample["path"]

        if label == 0:
            out_path = bonafide_dir / path
            n_bonafide += 1
        else:
            out_path = spoof_dir / path
            n_spoof += 1

        # Get audio bytes - with decode=False, audio is raw bytes
        audio_data = sample["audio"]

        # Save as FLAC
        if isinstance(audio_data, dict) and "bytes" in audio_data:
            # Write raw FLAC bytes
            with open(str(out_path), "wb") as f:
                f.write(audio_data["bytes"])
        elif isinstance(audio_data, bytes):
            with open(str(out_path), "wb") as f:
                f.write(audio_data)
        else:
            # Try to decode from array
            try:
                audio_array = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data.get("sampling_rate", 16000)
                sf.write(str(out_path), audio_array, sr, format="FLAC")
            except Exception as e:
                if (i + 1) % 1000 == 0:
                    print(f"    Skipping {path}: {e}")
                continue

        records.append({
            "path": str(out_path),
            "label": label,
            "split": "eval",
        })

        if (i + 1) % 5000 == 0:
            print(f"  Extracted {i + 1}/{len(ds)} ({n_bonafide} bonafide, {n_spoof} spoof)")

    print(f"\nTotal: {n_bonafide} bonafide, {n_spoof} spoof")

    # Create train/val/test split from eval data
    print("\nCreating train/val/test splits...")
    rng = random.Random(42)
    rng.shuffle(records)

    n = len(records)
    n_test = int(n * 0.15)
    n_val = int(n * 0.15)

    test_records = records[:n_test]
    val_records = records[n_test:n_test + n_val]
    train_records = records[n_test + n_val:]

    for r in train_records:
        r["split"] = "train"
    for r in val_records:
        r["split"] = "val"
    for r in test_records:
        r["split"] = "test"

    import pandas as pd
    df = pd.DataFrame(train_records + val_records + test_records)
    df.to_csv(SPLIT_CSV, index=False)

    print(f"Split CSV saved: {SPLIT_CSV}")
    print(df["split"].value_counts().to_string())
    for split in ["train", "val", "test"]:
        s = df[df["split"] == split]
        print(f"  {split}: {(s['label']==0).sum()} bonafide, {(s['label']==1).sum()} spoof")

    print(f"\nDone! Next: python -m src.audio_train --split_csv data/audio_split.csv --model efficientnet --epochs 15")


if __name__ == "__main__":
    main()
