"""
src/analyze_ff_video_errors.py

Video-level error analysis for FaceForensics++.

This script aggregates frame-level predictions into video/group-level
predictions.

Input:
    results/ff_errors/predictions.csv

Output:
    results/ff_errors/video_predictions.csv
    results/ff_errors/video_error_summary.csv
    results/ff_errors/video_error_summary.txt
"""

from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "ff_errors"
    / "predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ff_errors"
)

THRESHOLD = 0.5


# ============================================================
# Extract video/group ID
# ============================================================

def extract_video_group(path):
    """
    Extract the FF++ video/group directory from an image path.

    Example:

        ...\\real\\01__exit_phone_room\\frame_000000.png

    returns:

        01__exit_phone_room
    """

    p = Path(str(path))

    return p.parent.name


# ============================================================
# Load predictions
# ============================================================

def load_predictions():

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = {
        "path",
        "true_label",
        "prediction",
        "probability_fake",
        "result",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


# ============================================================
# Add video/group information
# ============================================================

def prepare_dataframe(df):

    df = df.copy()

    df["video_group"] = df["path"].apply(
        extract_video_group
    )

    # Convert labels/probabilities to numeric
    df["true_label"] = pd.to_numeric(
        df["true_label"],
        errors="coerce"
    )

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce"
    )

    df["probability_fake"] = pd.to_numeric(
        df["probability_fake"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "true_label",
            "prediction",
            "probability_fake",
            "video_group",
        ]
    )

    return df


# ============================================================
# Video-level aggregation
# ============================================================

def aggregate_video_predictions(df):

    records = []

    grouped = df.groupby(
        ["video_group", "true_label"],
        sort=False
    )

    for (video_group, true_label), group in grouped:

        probabilities = group[
            "probability_fake"
        ].to_numpy()

        frame_predictions = group[
            "prediction"
        ].to_numpy()

        # ----------------------------------------------------
        # Average probability
        # ----------------------------------------------------

        mean_probability = float(
            np.mean(probabilities)
        )

        median_probability = float(
            np.median(probabilities)
        )

        min_probability = float(
            np.min(probabilities)
        )

        max_probability = float(
            np.max(probabilities)
        )

        # ----------------------------------------------------
        # Frame-level majority vote
        # ----------------------------------------------------

        majority_prediction = int(
            np.mean(frame_predictions) >= 0.5
        )

        # ----------------------------------------------------
        # Video-level probability prediction
        # ----------------------------------------------------

        video_prediction = int(
            mean_probability >= THRESHOLD
        )

        # ----------------------------------------------------
        # Number of frames predicted fake
        # ----------------------------------------------------

        fake_frame_count = int(
            np.sum(frame_predictions == 1)
        )

        real_frame_count = int(
            np.sum(frame_predictions == 0)
        )

        total_frames = len(group)

        fake_frame_ratio = (
            fake_frame_count / total_frames
            if total_frames > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Video-level correctness
        # ----------------------------------------------------

        video_correct = int(
            video_prediction == int(true_label)
        )

        # ----------------------------------------------------
        # Error type
        # ----------------------------------------------------

        if video_prediction == 1 and true_label == 0:

            video_result = "FALSE_POSITIVE"

        elif video_prediction == 0 and true_label == 1:

            video_result = "FALSE_NEGATIVE"

        elif video_prediction == 0 and true_label == 0:

            video_result = "TRUE_NEGATIVE"

        else:

            video_result = "TRUE_POSITIVE"

        records.append(
            {
                "video_group": video_group,
                "true_label": int(true_label),
                "total_frames": total_frames,

                "mean_probability_fake":
                    mean_probability,

                "median_probability_fake":
                    median_probability,

                "min_probability_fake":
                    min_probability,

                "max_probability_fake":
                    max_probability,

                "fake_frame_count":
                    fake_frame_count,

                "real_frame_count":
                    real_frame_count,

                "fake_frame_ratio":
                    fake_frame_ratio,

                "majority_prediction":
                    majority_prediction,

                "video_prediction":
                    video_prediction,

                "video_correct":
                    video_correct,

                "video_result":
                    video_result,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Calculate video-level metrics
# ============================================================

def calculate_metrics(video_df):

    total = len(video_df)

    if total == 0:
        return {}

    accuracy = (
        video_df["video_correct"].mean()
    )

    true_real = video_df[
        video_df["true_label"] == 0
    ]

    true_fake = video_df[
        video_df["true_label"] == 1
    ]

    false_positive = len(
        video_df[
            (video_df["true_label"] == 0)
            & (video_df["video_prediction"] == 1)
        ]
    )

    false_negative = len(
        video_df[
            (video_df["true_label"] == 1)
            & (video_df["video_prediction"] == 0)
        ]
    )

    true_positive = len(
        video_df[
            (video_df["true_label"] == 1)
            & (video_df["video_prediction"] == 1)
        ]
    )

    true_negative = len(
        video_df[
            (video_df["true_label"] == 0)
            & (video_df["video_prediction"] == 0)
        ]
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "total_videos": total,
        "real_videos": len(true_real),
        "fake_videos": len(true_fake),

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


# ============================================================
# Analyze problematic videos
# ============================================================

def analyze_problematic_videos(video_df):

    false_positives = video_df[
        video_df["video_result"] == "FALSE_POSITIVE"
    ].copy()

    false_negatives = video_df[
        video_df["video_result"] == "FALSE_NEGATIVE"
    ].copy()

    false_positives = false_positives.sort_values(
        "mean_probability_fake",
        ascending=False
    )

    false_negatives = false_negatives.sort_values(
        "mean_probability_fake",
        ascending=True
    )

    return false_positives, false_negatives


# ============================================================
# Save summary
# ============================================================

def save_summary(
    video_df,
    false_positives,
    false_negatives,
    metrics,
):

    summary_file = (
        OUTPUT_DIR
        / "video_error_summary.txt"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "FACEFORENSICS++ VIDEO-LEVEL ERROR ANALYSIS\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Total videos: "
            f"{metrics['total_videos']}\n"
        )

        f.write(
            f"Real videos: "
            f"{metrics['real_videos']}\n"
        )

        f.write(
            f"Fake videos: "
            f"{metrics['fake_videos']}\n\n"
        )

        f.write(
            "VIDEO-LEVEL METRICS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Accuracy : "
            f"{metrics['accuracy']:.4f}\n"
        )

        f.write(
            f"Precision: "
            f"{metrics['precision']:.4f}\n"
        )

        f.write(
            f"Recall   : "
            f"{metrics['recall']:.4f}\n"
        )

        f.write(
            f"F1 Score : "
            f"{metrics['f1']:.4f}\n\n"
        )

        f.write(
            "CONFUSION MATRIX\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"True Negative : "
            f"{metrics['true_negative']}\n"
        )

        f.write(
            f"False Positive: "
            f"{metrics['false_positive']}\n"
        )

        f.write(
            f"False Negative: "
            f"{metrics['false_negative']}\n"
        )

        f.write(
            f"True Positive : "
            f"{metrics['true_positive']}\n\n"
        )

        # ----------------------------------------------------
        # False positives
        # ----------------------------------------------------

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "TOP VIDEO-LEVEL FALSE POSITIVES\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        for _, row in false_positives.head(20).iterrows():

            f.write(
                f"{row['video_group']} | "
                f"frames={int(row['total_frames'])} | "
                f"mean_p_fake="
                f"{row['mean_probability_fake']:.4f} | "
                f"fake_ratio="
                f"{row['fake_frame_ratio']:.2f}\n"
            )

        # ----------------------------------------------------
        # False negatives
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 70
            + "\n"
        )

        f.write(
            "TOP VIDEO-LEVEL FALSE NEGATIVES\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        for _, row in false_negatives.head(20).iterrows():

            f.write(
                f"{row['video_group']} | "
                f"frames={int(row['total_frames'])} | "
                f"mean_p_fake="
                f"{row['mean_probability_fake']:.4f} | "
                f"fake_ratio="
                f"{row['fake_frame_ratio']:.2f}\n"
            )

    return summary_file


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "FACEFORENSICS++ VIDEO-LEVEL ERROR ANALYSIS"
    )

    print(
        "=" * 60
    )

    print(
        f"Predictions:\n{PREDICTIONS_FILE}"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        "=" * 60
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\nLoading frame-level predictions..."
    )

    df = load_predictions()

    print(
        f"Total frame predictions: {len(df)}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df = prepare_dataframe(df)

    print(
        f"Valid predictions: {len(df)}"
    )

    # --------------------------------------------------------
    # Extract groups
    # --------------------------------------------------------

    print(
        "\nExtracting video/group IDs..."
    )

    print(
        f"Unique video groups: "
        f"{df['video_group'].nunique()}"
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    print(
        "\nAggregating frame predictions "
        "into video-level predictions..."
    )

    video_df = aggregate_video_predictions(
        df
    )

    print(
        f"Video-level samples: "
        f"{len(video_df)}"
    )

    # --------------------------------------------------------
    # Save video predictions
    # --------------------------------------------------------

    video_predictions_file = (
        OUTPUT_DIR
        / "video_predictions.csv"
    )

    video_df.to_csv(
        video_predictions_file,
        index=False
    )

    print(
        f"\nSaved:\n"
        f"{video_predictions_file}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        video_df
    )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    false_positives, false_negatives = (
        analyze_problematic_videos(
            video_df
        )
    )

    # --------------------------------------------------------
    # Save error summaries
    # --------------------------------------------------------

    error_summary_df = pd.concat(
        [
            false_positives.assign(
                error_priority="FALSE_POSITIVE"
            ),
            false_negatives.assign(
                error_priority="FALSE_NEGATIVE"
            ),
        ],
        ignore_index=True,
    )

    error_summary_file = (
        OUTPUT_DIR
        / "video_error_summary.csv"
    )

    error_summary_df.to_csv(
        error_summary_file,
        index=False
    )

    summary_file = save_summary(
        video_df,
        false_positives,
        false_negatives,
        metrics,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "VIDEO-LEVEL RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Videos       : "
        f"{metrics['total_videos']}"
    )

    print(
        f"Real         : "
        f"{metrics['real_videos']}"
    )

    print(
        f"Fake         : "
        f"{metrics['fake_videos']}"
    )

    print(
        f"Accuracy     : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision    : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall       : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score     : "
        f"{metrics['f1']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"[[{metrics['true_negative']} "
        f"{metrics['false_positive']}]"
    )

    print(
        f" [{metrics['false_negative']} "
        f"{metrics['true_positive']}]]"
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "VIDEO-LEVEL FALSE POSITIVES"
    )

    print(
        "=" * 60
    )

    print(
        f"Count: {len(false_positives)}"
    )

    print(
        false_positives[
            [
                "video_group",
                "total_frames",
                "mean_probability_fake",
                "fake_frame_ratio",
            ]
        ].head(20).to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "VIDEO-LEVEL FALSE NEGATIVES"
    )

    print(
        "=" * 60
    )

    print(
        f"Count: {len(false_negatives)}"
    )

    print(
        false_negatives[
            [
                "video_group",
                "total_frames",
                "mean_probability_fake",
                "fake_frame_ratio",
            ]
        ].head(20).to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "VIDEO-LEVEL ERROR ANALYSIS COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"  {video_predictions_file}"
    )

    print(
        f"  {error_summary_file}"
    )

    print(
        f"  {summary_file}"
    )


if __name__ == "__main__":
    main()