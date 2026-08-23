"""
Create reproducible train/val/test split CSV.

Usage:
    python src/make_split_csv.py --dataset celebdf \
        --data_root data/processed/celeb-df-v2 \
        --output_csv results/splits/celebdf_split_seed42.csv
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.dataset import load_celebdf_samples, load_ff_samples, save_split_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="celebdf", choices=["celebdf", "ff"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--compression", type=str, default="c23")
    parser.add_argument("--max_frames_per_video", type=int, default=10)
    parser.add_argument("--max_videos_per_class", type=int, default=300)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non_stratified", action="store_true")
    parser.add_argument("--output_csv", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "celebdf":
        samples = load_celebdf_samples(
            data_root=args.data_root,
            max_frames_per_video=args.max_frames_per_video,
            max_videos_per_class=args.max_videos_per_class,
        )
    else:
        samples = load_ff_samples(
            data_root=args.data_root,
            compression=args.compression,
            max_frames_per_video=args.max_frames_per_video,
        )

    save_split_csv(
        samples=samples,
        output_csv=args.output_csv,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        stratified=not args.non_stratified,
    )


if __name__ == "__main__":
    main()
