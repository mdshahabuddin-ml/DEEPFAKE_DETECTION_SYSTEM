"""
Visualize difficult FF++ videos at threshold 0.65.

Creates:
    results/ff_errors/video_error_visualization.png
"""

from pathlib import Path

import math
import pandas as pd
import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "ff_errors"
    / "predictions.csv"
)

VIDEO_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "ff_errors"
    / "video_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "ff_errors"
    / "video_error_visualization.png"
)

THRESHOLD = 0.65
FRAMES_PER_VIDEO = 4


def load_video_predictions():

    if not VIDEO_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{VIDEO_PREDICTIONS_FILE}"
        )

    return pd.read_csv(
        VIDEO_PREDICTIONS_FILE
    )


def load_frame_predictions():

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{PREDICTIONS_FILE}"
        )

    return pd.read_csv(
        PREDICTIONS_FILE
    )


def get_error_videos(video_df):

    video_df = video_df.copy()

    video_df["prediction_065"] = (
        video_df["mean_probability_fake"]
        >= THRESHOLD
    ).astype(int)

    fp = video_df[
        (video_df["true_label"] == 0)
        & (video_df["prediction_065"] == 1)
    ]

    fn = video_df[
        (video_df["true_label"] == 1)
        & (video_df["prediction_065"] == 0)
    ]

    return fp, fn


def get_video_frames(
    frame_df,
    video_group,
    max_frames=4
):

    rows = frame_df[
        frame_df["path"].str.contains(
            video_group,
            regex=False,
            na=False
        )
    ].copy()

    if rows.empty:
        return rows

    rows = rows.sort_values(
        "path"
    )

    if len(rows) > max_frames:

        indices = [
            int(i)
            for i in
            __import__("numpy").linspace(
                0,
                len(rows) - 1,
                max_frames
            )
        ]

        rows = rows.iloc[
            sorted(set(indices))
        ]

    return rows


def read_image(path):

    image = cv2.imread(
        str(path)
    )

    if image is None:
        return None

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


def main():

    print("=" * 70)
    print("FF++ DIFFICULT VIDEO VISUALIZATION")
    print("=" * 70)

    video_df = load_video_predictions()
    frame_df = load_frame_predictions()

    fp, fn = get_error_videos(
        video_df
    )

    errors = []

    for _, row in fp.iterrows():

        errors.append(
            (
                "FALSE POSITIVE",
                row
            )
        )

    for _, row in fn.iterrows():

        errors.append(
            (
                "FALSE NEGATIVE",
                row
            )
        )

    print(
        f"\nFalse positives: {len(fp)}"
    )

    print(
        f"False negatives: {len(fn)}"
    )

    if not errors:

        print(
            "No errors found."
        )

        return

    rows_per_video = FRAMES_PER_VIDEO
    total_plots = len(errors) * rows_per_video

    columns = 4
    rows = math.ceil(
        total_plots / columns
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, rows * 4)
    )

    axes = axes.flatten()

    plot_index = 0

    for error_type, video in errors:

        video_group = video[
            "video_group"
        ]

        mean_probability = video[
            "mean_probability_fake"
        ]

        frame_ratio = video[
            "fake_frame_ratio"
        ]

        frames = get_video_frames(
            frame_df,
            video_group,
            FRAMES_PER_VIDEO
        )

        print(
            f"\n{error_type}: "
            f"{video_group}"
        )

        print(
            f"Mean fake probability: "
            f"{mean_probability:.4f}"
        )

        print(
            f"Fake frame ratio: "
            f"{frame_ratio:.2f}"
        )

        if frames.empty:

            print(
                "  No frame records found."
            )

            continue

        for _, frame in frames.iterrows():

            if plot_index >= len(axes):
                break

            image = read_image(
                frame["path"]
            )

            ax = axes[plot_index]

            if image is not None:

                ax.imshow(image)

            else:

                ax.text(
                    0.5,
                    0.5,
                    "Image unavailable",
                    ha="center",
                    va="center"
                )

            probability = frame[
                "probability_fake"
            ]

            ax.set_title(
                f"{error_type}\n"
                f"{video_group}\n"
                f"p(fake)={probability:.3f}",
                fontsize=8
            )

            ax.axis("off")

            plot_index += 1

    for i in range(
        plot_index,
        len(axes)
    ):

        axes[i].axis(
            "off"
        )

    fig.suptitle(
        "FF++ Difficult Videos — Threshold 0.65",
        fontsize=16
    )

    plt.tight_layout()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETED")
    print("=" * 70)

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()