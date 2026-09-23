"""
scripts/prepare_asvspoof.py
Parse ASVspoof 2019 LA protocol files and generate data/audio_split.csv.

Usage:
    python scripts/prepare_asvspoof.py

Requires LA.zip to be downloaded and extracted first (run download_asvspoof.py).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASVSPOOF_DIR = PROJECT_ROOT / "data" / "asvspoof2019" / "LA"
SPLIT_CSV = PROJECT_ROOT / "data" / "audio_split.csv"


def parse_protocol_file(protocol_path):
    """Parse an ASVspoof protocol file.

    Format: SPEAKER_ID AUDIO_FILE_NAME - KEY
    KEY is 'bonafide' or 'spoof'
    """
    samples = []
    if not protocol_path.exists():
        print(f"WARNING: Protocol file not found: {protocol_path}")
        return samples

    with open(protocol_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            speaker_id = parts[0]
            audio_file = parts[1]
            # parts[2] is '-' (unused column)
            key = parts[4]

            label = 0 if key == "bonafide" else 1
            samples.append({
                "speaker_id": speaker_id,
                "audio_file": audio_file,
                "label": label,
            })

    return samples


def find_audio_files(samples, flac_dir):
    """Match protocol entries to actual FLAC files on disk."""
    matched = []
    unmatched = 0

    for s in samples:
        audio_file = s["audio_file"]
        # Protocol uses filename without extension
        flac_path = flac_dir / f"{audio_file}.flac"

        if flac_path.exists():
            matched.append({
                "path": str(flac_path),
                "label": s["label"],
            })
        else:
            # Try looking in subdirectories
            found = False
            for sub in flac_dir.rglob(f"{audio_file}.flac"):
                matched.append({
                    "path": str(sub),
                    "label": s["label"],
                })
                found = True
                break
            if not found:
                unmatched += 1

    if unmatched:
        print(f"  WARNING: {unmatched} files not found on disk")

    return matched


def main():
    os.chdir(PROJECT_ROOT)

    print("=" * 60)
    print("ASVspoof 2019 LA - Data Preparation")
    print("=" * 60)

    # Find protocol files
    proto_dirs = [
        ASVSPOOF_DIR / "ASVspoof2019_LA_cm_protocols",
        ASVSPOOF_DIR / "LA" / "ASVspoof2019_LA_cm_protocols",
    ]

    proto_dir = None
    for d in proto_dirs:
        if d.exists():
            proto_dir = d
            break

    if proto_dir is None:
        print(f"ERROR: Protocol directory not found!")
        print(f"Searched in:")
        for d in proto_dirs:
            print(f"  {d}")
        print(f"\nPlease run download_asvspoof.py first.")
        return

    print(f"Protocol directory: {proto_dir}")
    print(f"Contents: {sorted([f.name for f in proto_dir.iterdir()])}")

    # Find FLAC directories
    flac_dirs = [
        ASVSPOOF_DIR / "ASVspoof2019_LA_train" / "flac",
        ASVSPOOF_DIR / "ASVspoof2019_LA_dev" / "flac",
        ASVSPOOF_DIR / "LA" / "ASVspoof2019_LA_train" / "flac",
        ASVSPOOF_DIR / "LA" / "ASVspoof2019_LA_dev" / "flac",
    ]

    train_flac = None
    dev_flac = None

    for d in flac_dirs:
        if d.exists() and "train" in str(d):
            train_flac = d
        elif d.exists() and "dev" in str(d):
            dev_flac = d

    if train_flac is None:
        # Try direct FLAC directories
        for d in [ASVSPOOF_DIR / "flac", ASVSPOOF_DIR / "LA" / "flac"]:
            if d.exists():
                train_flac = d
                dev_flac = d
                break

    if train_flac is None:
        print("ERROR: FLAC directories not found!")
        print("Directory structure:")
        for root, dirs, files in os.walk(ASVSPOOF_DIR):
            level = root.replace(str(ASVSPOOF_DIR), "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            if level < 2:
                subindent = " " * 2 * (level + 1)
                for f in sorted(files)[:5]:
                    print(f"{subindent}{f}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files)-5} more")
        return

    print(f"Train FLAC dir: {train_flac}")
    print(f"Dev FLAC dir:   {dev_flac}")

    # Parse protocol files
    train_proto = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_proto = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"

    print(f"\nParsing train protocol: {train_proto}")
    train_samples = parse_protocol_file(train_proto)
    print(f"  Train entries: {len(train_samples)}")

    print(f"Parsing dev protocol: {dev_proto}")
    dev_samples = parse_protocol_file(dev_proto)
    print(f"  Dev entries: {len(dev_samples)}")

    # Match to actual files
    print(f"\nMatching train files to disk...")
    train_matched = find_audio_files(train_samples, train_flac)
    print(f"  Train matched: {len(train_matched)}")

    print(f"Matching dev files to disk...")
    dev_matched = find_audio_files(dev_samples, dev_flac)
    print(f"  Dev matched: {len(dev_matched)}")

    # Split dev into val/test
    import random
    rng = random.Random(42)
    rng.shuffle(dev_matched)
    n_val = int(len(dev_matched) * 0.15)
    val_matched = dev_matched[:n_val]
    test_matched = dev_matched[n_val:]

    # Label splits
    for s in train_matched:
        s["split"] = "train"
    for s in val_matched:
        s["split"] = "val"
    for s in test_matched:
        s["split"] = "test"

    # Combine
    all_data = train_matched + val_matched + test_matched

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(all_data)
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLIT_CSV, index=False)

    print(f"\n{'=' * 60}")
    print(f"Split CSV saved: {SPLIT_CSV}")
    print(f"{'=' * 60}")
    print(f"\nSplit counts:")
    print(df["split"].value_counts().to_string())
    print(f"\nLabel distribution:")
    for split in ["train", "val", "test"]:
        s = df[df["split"] == split]
        n_bonafide = (s["label"] == 0).sum()
        n_spoof = (s["label"] == 1).sum()
        print(f"  {split}: {n_bonafide} bonafide, {n_spoof} spoof")

    print(f"\nNext step:")
    print(f"  python -m src.audio_train --split_csv data/audio_split.csv --model cnn --epochs 10 --batch_size 16 --num_workers 0")
    print(f"  python -m src.audio_train --split_csv data/audio_split.csv --model efficientnet --epochs 15 --batch_size 8 --num_workers 0")


if __name__ == "__main__":
    main()
