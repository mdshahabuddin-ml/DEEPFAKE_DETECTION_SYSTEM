"""
FaceForensics++ Video-Level Error Analysis at Threshold 0.65

Input:
    results/ff_errors/video_predictions.csv

Output:
    results/ff_errors/video_errors_065.csv
    results/ff_errors/video_fp_065.csv
    results/ff_errors/video_fn_065.csv
"""

from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "ff_errors"
    / "video_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ff_errors"
)

THRESHOLD = 0.65


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("FF++ VIDEO ERROR ANALYSIS - THRESHOLD 0.65")
    print("=" * 70)

    print(f"\nInput:")
    print(INPUT_FILE)

    print(f"\nThreshold: {THRESHOLD}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nTotal videos: {len(df)}"
    )

    required_columns = {
        "video_group",
        "true_label",
        "mean_probability_fake",
        "fake_frame_ratio",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------

    df["prediction_065"] = (
        df["mean_probability_fake"] >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Determine result
    # --------------------------------------------------------

    def classify(row):

        true_label = int(row["true_label"])
        prediction = int(row["prediction_065"])

        if true_label == 0 and prediction == 0:
            return "TRUE_NEGATIVE"

        if true_label == 0 and prediction == 1:
            return "FALSE_POSITIVE"

        if true_label == 1 and prediction == 0:
            return "FALSE_NEGATIVE"

        if true_label == 1 and prediction == 1:
            return "TRUE_POSITIVE"

        return "UNKNOWN"

    df["result_065"] = df.apply(
        classify,
        axis=1
    )

    # --------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------

    all_output = (
        OUTPUT_DIR
        / "video_errors_065.csv"
    )

    df.to_csv(
        all_output,
        index=False
    )

    # --------------------------------------------------------
    # False positives
    # --------------------------------------------------------

    false_positives = df[
        df["result_065"] == "FALSE_POSITIVE"
    ].copy()

    false_positives = false_positives.sort_values(
        "mean_probability_fake",
        ascending=False
    )

    fp_output = (
        OUTPUT_DIR
        / "video_fp_065.csv"
    )

    false_positives.to_csv(
        fp_output,
        index=False
    )

    # --------------------------------------------------------
    # False negatives
    # --------------------------------------------------------

    false_negatives = df[
        df["result_065"] == "FALSE_NEGATIVE"
    ].copy()

    false_negatives = false_negatives.sort_values(
        "mean_probability_fake",
        ascending=True
    )

    fn_output = (
        OUTPUT_DIR
        / "video_fn_065.csv"
    )

    false_negatives.to_csv(
        fn_output,
        index=False
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    y_true = df["true_label"].to_numpy()
    y_pred = df["prediction_065"].to_numpy()

    tn = int(
        np.sum(
            (y_true == 0) &
            (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0) &
            (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1) &
            (y_pred == 0)
        )
    )

    tp = int(
        np.sum(
            (y_true == 1) &
            (y_pred == 1)
        )
    )

    total = len(df)

    accuracy = (
        (tp + tn) / total
        if total > 0 else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VIDEO-LEVEL RESULTS @ THRESHOLD 0.65")
    print("=" * 70)

    print(
        f"Videos       : {total}"
    )

    print(
        f"Accuracy     : {accuracy:.4f}"
    )

    print(
        f"Precision    : {precision:.4f}"
    )

    print(
        f"Recall       : {recall:.4f}"
    )

    print(
        f"F1 Score     : {f1:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        f"[[{tn:2d} {fp:2d}]"
    )
    print(
        f" [{fn:2d} {tp:2d}]]"
    )

    # --------------------------------------------------------
    # False positives
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FALSE POSITIVES @ 0.65")
    print("=" * 70)

    print(
        f"Count: {len(false_positives)}"
    )

    if len(false_positives) > 0:

        print(
            false_positives[
                [
                    "video_group",
                    "total_frames",
                    "mean_probability_fake",
                    "fake_frame_ratio",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # False negatives
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FALSE NEGATIVES @ 0.65")
    print("=" * 70)

    print(
        f"Count: {len(false_negatives)}"
    )

    if len(false_negatives) > 0:

        print(
            false_negatives[
                [
                    "video_group",
                    "total_frames",
                    "mean_probability_fake",
                    "fake_frame_ratio",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Saved files
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"All results:\n{all_output}"
    )

    print(
        f"\nFalse positives:\n{fp_output}"
    )

    print(
        f"\nFalse negatives:\n{fn_output}"
    )

    print("\n" + "=" * 70)
    print("THRESHOLD 0.65 ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()