"""
src/plot_results.py

Visualization of dataset-wise deepfake detection results.

Results are based on the completed evaluation of:
    - Indian Deepfake Face
    - WildDeepfake
    - DFDC
    - FaceForensics++
    - Celeb-DF v2

Outputs:
    results/
        dataset_metrics.png
        accuracy_comparison.png
        f1_comparison.png
        auc_comparison.png
        confusion_matrices.png
        dataset_results.csv

Run:
    python src/plot_results.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Dataset-wise evaluation results
# ============================================================

RESULTS = {
    "Indian": {
        "accuracy": 0.9638,
        "precision": 0.9587,
        "recall": 0.9694,
        "f1": 0.9640,
        "auc": 0.9951,
        "tn": 344,
        "fp": 15,
        "fn": 11,
        "tp": 348,
        "samples": 718,
    },

    "WildDeepfake": {
        "accuracy": 0.9423,
        "precision": 0.9831,
        "recall": 0.9002,
        "f1": 0.9398,
        "auc": 0.9917,
        "tn": 572,
        "fp": 9,
        "fn": 58,
        "tp": 523,
        "samples": 1162,
    },

    "DFDC": {
        "accuracy": 0.9140,
        "precision": 0.9165,
        "recall": 0.9110,
        "f1": 0.9137,
        "auc": 0.9777,
        "tn": 917,
        "fp": 83,
        "fn": 89,
        "tp": 911,
        "samples": 2000,
    },

    "FaceForensics++": {
        "accuracy": 0.8440,
        "precision": 0.7905,
        "recall": 0.9360,
        "f1": 0.8571,
        "auc": 0.9463,
        "tn": 752,
        "fp": 248,
        "fn": 64,
        "tp": 936,
        "samples": 2000,
    },

    "Celeb-DF v2": {
        "accuracy": 0.9585,
        "precision": 0.9842,
        "recall": 0.9320,
        "f1": 0.9574,
        "auc": 0.9925,
        "tn": 985,
        "fp": 15,
        "fn": 68,
        "tp": 932,
        "samples": 2000,
    },
}


# ============================================================
# Convert results to DataFrame
# ============================================================

def create_dataframe():
    rows = []

    for dataset, metrics in RESULTS.items():
        rows.append({
            "Dataset": dataset,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC-AUC": metrics["auc"],
            "Samples": metrics["samples"],
        })

    return pd.DataFrame(rows)


# ============================================================
# Save CSV
# ============================================================

def save_csv(df):
    output_file = RESULTS_DIR / "dataset_results.csv"

    csv_df = df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]:
        csv_df[column] = (csv_df[column] * 100).round(2)

    csv_df.to_csv(output_file, index=False)

    print(f"Saved CSV: {output_file}")


# ============================================================
# Helper: add percentage labels
# ============================================================

def add_bar_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ============================================================
# Plot 1: All metrics
# ============================================================

def plot_all_metrics(df):

    datasets = df["Dataset"]

    x = np.arange(len(datasets))
    width = 0.16

    fig, ax = plt.subplots(figsize=(14, 7))

    metrics = [
        ("Accuracy", "Accuracy"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1", "F1 Score"),
        ("ROC-AUC", "ROC-AUC"),
    ]

    for i, (column, label) in enumerate(metrics):

        bars = ax.bar(
            x + (i - 2) * width,
            df[column],
            width,
            label=label,
        )

    ax.set_title(
        "Dataset-wise Deepfake Detection Performance",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(
        datasets,
        rotation=20,
        ha="right",
    )

    ax.set_ylim(0, 1.05)

    ax.set_yticks(
        np.arange(0, 1.1, 0.1)
    )

    ax.set_yticklabels(
        [f"{int(x * 100)}%" for x in np.arange(0, 1.1, 0.1)]
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(
        loc="lower right"
    )

    plt.tight_layout()

    output_file = RESULTS_DIR / "dataset_metrics.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot: {output_file}")


# ============================================================
# Plot 2: Accuracy
# ============================================================

def plot_accuracy(df):

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        df["Dataset"],
        df["Accuracy"],
    )

    ax.set_title(
        "Accuracy by Dataset",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")

    ax.set_ylim(0, 1.05)

    ax.set_yticks(
        np.arange(0, 1.1, 0.1)
    )

    ax.set_yticklabels(
        [f"{int(x * 100)}%" for x in np.arange(0, 1.1, 0.1)]
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    add_bar_labels(ax, bars)

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    output_file = RESULTS_DIR / "accuracy_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot: {output_file}")


# ============================================================
# Plot 3: F1 Score
# ============================================================

def plot_f1(df):

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        df["Dataset"],
        df["F1"],
    )

    ax.set_title(
        "F1 Score by Dataset",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("F1 Score")

    ax.set_ylim(0, 1.05)

    ax.set_yticks(
        np.arange(0, 1.1, 0.1)
    )

    ax.set_yticklabels(
        [f"{int(x * 100)}%" for x in np.arange(0, 1.1, 0.1)]
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    add_bar_labels(ax, bars)

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    output_file = RESULTS_DIR / "f1_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot: {output_file}")


# ============================================================
# Plot 4: ROC-AUC
# ============================================================

def plot_auc(df):

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        df["Dataset"],
        df["ROC-AUC"],
    )

    ax.set_title(
        "ROC-AUC by Dataset",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("ROC-AUC")

    ax.set_ylim(0, 1.05)

    ax.set_yticks(
        np.arange(0, 1.1, 0.1)
    )

    ax.set_yticklabels(
        [f"{int(x * 100)}%" for x in np.arange(0, 1.1, 0.1)]
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    add_bar_labels(ax, bars)

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    output_file = RESULTS_DIR / "auc_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot: {output_file}")


# ============================================================
# Plot 5: Confusion Matrices
# ============================================================

def plot_confusion_matrices():

    datasets = list(RESULTS.keys())

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 9),
    )

    axes = axes.flatten()

    for index, dataset in enumerate(datasets):

        metrics = RESULTS[dataset]

        matrix = np.array([
            [
                metrics["tn"],
                metrics["fp"],
            ],
            [
                metrics["fn"],
                metrics["tp"],
            ],
        ])

        ax = axes[index]

        image = ax.imshow(
            matrix,
            interpolation="nearest",
        )

        ax.set_title(
            dataset,
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Predicted Label"
        )

        ax.set_ylabel(
            "True Label"
        )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ["REAL", "FAKE"]
        )

        ax.set_yticklabels(
            ["REAL", "FAKE"]
        )

        threshold = matrix.max() / 2

        for row in range(2):
            for col in range(2):

                ax.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )

        ax.set_ylim(
            1.5,
            -0.5
        )

    # Hide unused sixth subplot
    axes[-1].axis("off")

    fig.suptitle(
        "Confusion Matrices by Dataset",
        fontsize=18,
        fontweight="bold",
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    output_file = (
        RESULTS_DIR /
        "confusion_matrices.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved plot: {output_file}"
    )


# ============================================================
# Print summary
# ============================================================

def print_summary(df):

    print("\n")
    print("=" * 80)
    print("DATASET-WISE PERFORMANCE SUMMARY")
    print("=" * 80)

    print(
        f"{'Dataset':<22}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'AUC':>12}"
    )

    print("-" * 80)

    for _, row in df.iterrows():

        print(
            f"{row['Dataset']:<22}"
            f"{row['Accuracy'] * 100:>11.2f}%"
            f"{row['Precision'] * 100:>11.2f}%"
            f"{row['Recall'] * 100:>11.2f}%"
            f"{row['F1'] * 100:>11.2f}%"
            f"{row['ROC-AUC'] * 100:>11.2f}%"
        )

    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("DEEPFAKE DETECTION RESULT VISUALIZATION")
    print("=" * 60)

    print(
        f"\nOutput directory: {RESULTS_DIR.resolve()}"
    )

    df = create_dataframe()

    print_summary(df)

    print("\nGenerating CSV...")
    save_csv(df)

    print("\nGenerating plots...")

    plot_all_metrics(df)

    plot_accuracy(df)

    plot_f1(df)

    plot_auc(df)

    plot_confusion_matrices()

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated files:")

    for file in sorted(RESULTS_DIR.iterdir()):

        if file.is_file():

            print(
                f"  ✓ {file}"
            )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()