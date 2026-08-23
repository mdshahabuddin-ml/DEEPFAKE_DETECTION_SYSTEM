from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


CSV_PATH = Path("results/ff_full_video_evaluation/all_video_results.csv")


def evaluate_threshold(df, threshold):
    y_true = (df["true_label"].str.upper() == "FAKE").astype(int)
    y_pred = (df["mean_probability_fake"] >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def main():
    print("=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation file not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    print(f"\nEvaluation samples: {len(df)}")

    current = evaluate_threshold(df, 0.65)
    optimized = evaluate_threshold(df, 0.47)

    print("\n" + "-" * 70)
    print("CURRENT THRESHOLD: 0.65")
    print("-" * 70)

    print(f"Accuracy : {current['accuracy']:.4f}")
    print(f"Precision: {current['precision']:.4f}")
    print(f"Recall   : {current['recall']:.4f}")
    print(f"F1       : {current['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(
        f"TN={current['TN']}  "
        f"FP={current['FP']}  "
        f"FN={current['FN']}  "
        f"TP={current['TP']}"
    )

    print("\n" + "-" * 70)
    print("OPTIMIZED THRESHOLD: 0.47")
    print("-" * 70)

    print(f"Accuracy : {optimized['accuracy']:.4f}")
    print(f"Precision: {optimized['precision']:.4f}")
    print(f"Recall   : {optimized['recall']:.4f}")
    print(f"F1       : {optimized['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(
        f"TN={optimized['TN']}  "
        f"FP={optimized['FP']}  "
        f"FN={optimized['FN']}  "
        f"TP={optimized['TP']}"
    )

    print("\n" + "-" * 70)
    print("IMPROVEMENT")
    print("-" * 70)

    print(
        f"Accuracy change : "
        f"{(optimized['accuracy'] - current['accuracy']) * 100:+.2f}%"
    )

    print(
        f"Precision change: "
        f"{(optimized['precision'] - current['precision']) * 100:+.2f}%"
    )

    print(
        f"Recall change   : "
        f"{(optimized['recall'] - current['recall']) * 100:+.2f}%"
    )

    print(
        f"F1 change       : "
        f"{(optimized['f1'] - current['f1']) * 100:+.2f}%"
    )

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if optimized["f1"] > current["f1"]:
        print("Threshold 0.47 gives a better F1 score on this evaluation set.")
    else:
        print("Threshold 0.65 gives a better F1 score on this evaluation set.")

    print("=" * 70)


if __name__ == "__main__":
    main()