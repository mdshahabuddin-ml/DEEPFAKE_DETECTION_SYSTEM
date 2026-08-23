"""Prepare labelled face crops and source-disjoint splits from a DFDC folder.

The DFDC metadata maps every fake video to its original video.  This script
keeps an original and all of its fakes in one split, avoiding identity leakage
between training, validation, and test sets.

Example:
    python -m src.prepare_dfdc --input_dir "E:\\DFDC_dataset\\DFDC Dtata_set_kaggle" \
        --output_dir data/processed/dfdc --frames_per_video 4
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DFDC frames and source-disjoint splits.")
    parser.add_argument("--input_dir", required=True, help="Folder containing DFDC MP4 files and metadata.json")
    parser.add_argument("--output_dir", default="data/processed/dfdc")
    parser.add_argument("--frames_per_video", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Optional limit for a quick smoke run; omit for the complete dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_id(filename: str, item: dict) -> str:
    """Return an identity group: a real source and every fake derived from it."""
    return item.get("original", filename)


def assign_splits(metadata: dict, seed: int) -> dict[str, str]:
    """Assign full source groups to 70/15/15 train/validation/test splits."""
    groups = sorted({source_id(name, item) for name, item in metadata.items()})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_test, n_val = max(1, round(n * 0.15)), max(1, round(n * 0.15))
    split_for_group = {}
    for index, group in enumerate(groups):
        split_for_group[group] = "test" if index < n_test else "val" if index < n_test + n_val else "train"
    return {name: split_for_group[source_id(name, item)] for name, item in metadata.items()}


def crop_face_or_frame(frame, detector, image_size: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces):
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        margin = int(max(w, h) * 0.20)
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
        frame = frame[y0:y1, x0:x1]
    return cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)


def extract_video(video_path: Path, output_dir: Path, detector, frames_per_video: int,
                  image_size: int, overwrite: bool) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARNING] Skipping unreadable video: {video_path.name}")
        return []
    frame_count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    targets = sorted({round(i * (frame_count - 1) / max(1, frames_per_video - 1)) for i in range(frames_per_video)})
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for number, target in enumerate(targets):
        out_file = output_dir / f"frame_{number:02d}.jpg"
        if out_file.exists() and not overwrite:
            saved.append(out_file)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = crop_face_or_frame(frame, detector, image_size)
        if cv2.imwrite(str(out_file), crop):
            saved.append(out_file)
    cap.release()
    return saved


def main():
    args = parse_args()
    if args.frames_per_video < 1:
        raise ValueError("--frames_per_video must be at least 1")
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing DFDC metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    splits = assign_splits(metadata, args.seed)
    entries = [(name, item) for name, item in sorted(metadata.items()) if (input_dir / name).exists()]
    if args.max_videos is not None:
        # Sample complete source groups, retaining one real and one fake from
        # each. This keeps every split binary even in a small smoke run.
        by_group = {}
        for name, item in entries:
            by_group.setdefault(source_id(name, item), []).append((name, item))
        rng = random.Random(args.seed)
        selected = []
        group_budget = max(3, args.max_videos // 2)
        for split_name, fraction in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
            candidates = [g for g in by_group if splits.get(g) == split_name]
            rng.shuffle(candidates)
            wanted = max(1, round(group_budget * fraction))
            for group in candidates[:wanted]:
                group_entries = by_group[group]
                real = next((entry for entry in group_entries if entry[1]["label"] == "REAL"), None)
                fake = next((entry for entry in group_entries if entry[1]["label"] == "FAKE"), None)
                if real:
                    selected.append(real)
                if fake:
                    selected.append(fake)
        entries = sorted(selected[:args.max_videos])
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV Haar cascade is unavailable")

    rows = []
    for index, (filename, item) in enumerate(entries, 1):
        video_path = input_dir / filename
        frames = extract_video(video_path, output_dir / "frames" / video_path.stem, detector,
                               args.frames_per_video, args.image_size, args.overwrite)
        label = 1 if item["label"] == "FAKE" else 0
        for frame in frames:
            rows.append({"path": str(frame.resolve()), "label": label, "split": splits[filename],
                         "video": filename, "source": source_id(filename, item)})
        if index % 50 == 0 or index == len(entries):
            print(f"Prepared {index}/{len(entries)} videos; {len(rows)} usable frames")

    if not rows:
        raise RuntimeError("No frames were extracted. Check the input videos and OpenCV installation.")
    csv_path = output_dir / "dfdc_split_seed42.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    df = pd.DataFrame(rows)
    print(f"Saved: {csv_path}")
    print("Frames by split and label:\n" + df.groupby(["split", "label"]).size().to_string())
    print("Source groups by split:\n" + df.groupby("split")["source"].nunique().to_string())


if __name__ == "__main__":
    main()
