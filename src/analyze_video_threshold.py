"""
Analyze different thresholds for FF++ video-level predictions.

Input:
    results/ff_errors/video_predictions.csv

Output:
    results/ff_errors/threshold_analysis.csv
"""

from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "ff_errors"
    / "video_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "ff_errors"
    / "threshold_analysis.csv"
)


def calculate_metrics(df, threshold):

    y_true = df["true_label"].to_numpy()

    probabilities = (
        df["mean_probability_fake"].to_numpy()
    )

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    tn = int(
        np.sum((y_true == 0) & (y_pred == 0))
    )

    fp = int(
        np.sum((y_true == 0) & (y_pred == 1))
    )

    fn = int(
        np.sum((y_true == 1) & (y_pred == 0))
    )

    tp = int(
        np.sum((y_true == 1) & (y_pred == 1))
    )

    total = len(y_true)

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

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def main():

    print("=" * 70)
    print("FF++ VIDEO-LEVEL THRESHOLD ANALYSIS")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"File not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nInput videos: {len(df)}"
    )

    print(
        f"Real videos: "
        f"{(df['true_label'] == 0).sum()}"
    )

    print(
        f"Fake videos: "
        f"{(df['true_label'] == 1).sum()}"
    )

    thresholds = np.arange(
        0.30,
        0.81,
        0.05
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            df,
            float(threshold)
        )

        results.append(metrics)

    results_df = pd.DataFrame(
        results
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("THRESHOLD RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
            }
        )
    )

    # Best threshold by F1
    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST F1 THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold : "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{best_f1['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_f1['f1']:.4f}"
    )

    print(
        f"FP        : "
        f"{int(best_f1['false_positive'])}"
    )

    print(
        f"FN        : "
        f"{int(best_f1['false_negative'])}"
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()