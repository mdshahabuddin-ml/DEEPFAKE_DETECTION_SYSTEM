"""
Analyze FaceForensics++ false positives and false negatives
by video/group.

Input:
    results/ff_errors/predictions.csv

Output:
    results/ff_errors/error_groups.csv
    results/ff_errors/error_group_summary.txt
"""

from pathlib import Path
import pandas as pd


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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_group(path):
    """
    Extract the video/group name from an FF++ frame path.

    Example:

    real/01__exit_phone_room/frame_000090.png

    -> 01__exit_phone_room

    fake/01_03__hugging_happy__ISF9SP4G/frame_000040.png

    -> 01_03__hugging_happy__ISF9SP4G
    """

    p = Path(path)

    return p.parent.name


def main():

    print("=" * 60)
    print("FACEFORENSICS++ ERROR GROUP ANALYSIS")
    print("=" * 60)

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Predictions file not found:\n{PREDICTIONS_FILE}"
        )

    print(f"Reading:\n{PREDICTIONS_FILE}")

    df = pd.read_csv(PREDICTIONS_FILE)

    print(f"\nTotal predictions: {len(df)}")

    # --------------------------------------------------------
    # Extract video/group
    # --------------------------------------------------------

    df["group"] = df["path"].apply(extract_group)

    # --------------------------------------------------------
    # Determine errors
    # --------------------------------------------------------

    df["error"] = df["true_label"] != df["prediction"]

    df["error_type"] = "CORRECT"

    df.loc[
        (df["true_label"] == 0)
        & (df["prediction"] == 1),
        "error_type"
    ] = "FALSE_POSITIVE"

    df.loc[
        (df["true_label"] == 1)
        & (df["prediction"] == 0),
        "error_type"
    ] = "FALSE_NEGATIVE"

    # --------------------------------------------------------
    # Group statistics
    # --------------------------------------------------------

    grouped = (
        df.groupby(
            ["true_label", "group"],
            dropna=False
        )
        .agg(
            total_samples=("path", "count"),
            errors=("error", "sum"),
            mean_probability_fake=(
                "probability_fake",
                "mean"
            ),
            min_probability_fake=(
                "probability_fake",
                "min"
            ),
            max_probability_fake=(
                "probability_fake",
                "max"
            ),
        )
        .reset_index()
    )

    grouped["error_rate"] = (
        grouped["errors"]
        / grouped["total_samples"]
    )

    # --------------------------------------------------------
    # Sort worst groups first
    # --------------------------------------------------------

    grouped = grouped.sort_values(
        ["errors", "error_rate"],
        ascending=False
    )

    output_csv = OUTPUT_DIR / "error_groups.csv"

    grouped.to_csv(
        output_csv,
        index=False
    )

    print(f"\nSaved: {output_csv}")

    # --------------------------------------------------------
    # False-positive groups
    # --------------------------------------------------------

    fp = grouped[
        grouped["true_label"] == 0
    ].copy()

    fp = fp.sort_values(
        ["errors", "error_rate"],
        ascending=False
    )

    # --------------------------------------------------------
    # False-negative groups
    # --------------------------------------------------------

    fn = grouped[
        grouped["true_label"] == 1
    ].copy()

    fn = fn.sort_values(
        ["errors", "error_rate"],
        ascending=False
    )

    # --------------------------------------------------------
    # Print top FP groups
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP FALSE-POSITIVE GROUPS")
    print("=" * 60)

    print(
        fp[
            [
                "group",
                "total_samples",
                "errors",
                "error_rate",
                "mean_probability_fake",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Print top FN groups
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP FALSE-NEGATIVE GROUPS")
    print("=" * 60)

    print(
        fn[
            [
                "group",
                "total_samples",
                "errors",
                "error_rate",
                "mean_probability_fake",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_file = (
        OUTPUT_DIR
        / "error_group_summary.txt"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FACEFORENSICS++ ERROR GROUP ANALYSIS\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            f"Total predictions: {len(df)}\n"
        )

        f.write(
            f"False positives: "
            f"{len(df[(df.true_label == 0) & (df.prediction == 1)])}\n"
        )

        f.write(
            f"False negatives: "
            f"{len(df[(df.true_label == 1) & (df.prediction == 0)])}\n\n"
        )

        f.write(
            "TOP FALSE-POSITIVE GROUPS\n"
        )

        f.write("-" * 60 + "\n")

        f.write(
            fp.head(20).to_string(index=False)
        )

        f.write("\n\n")

        f.write(
            "TOP FALSE-NEGATIVE GROUPS\n"
        )

        f.write("-" * 60 + "\n")

        f.write(
            fn.head(20).to_string(index=False)
        )

    print(f"\nSaved: {summary_file}")

    print("\n" + "=" * 60)
    print("ERROR GROUP ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()