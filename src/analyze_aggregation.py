import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CSV_PATH = r"results\ff_full_video_evaluation\all_video_results.csv"

THRESHOLD = 0.47

df = pd.read_csv(CSV_PATH)

y_true = (df["true_label"].str.upper() == "FAKE").astype(int)

mean_prob = df["mean_probability_fake"].astype(float)
fake_ratio = df["fake_frame_ratio"].astype(float)

print("=" * 70)
print("VIDEO-LEVEL AGGREGATION ANALYSIS")
print("=" * 70)

print(f"Samples: {len(df)}")
print(f"Threshold: {THRESHOLD}")

# ---------------------------------------------------------
# Method 1: Mean probability
# ---------------------------------------------------------

pred_mean = (mean_prob >= THRESHOLD).astype(int)

# ---------------------------------------------------------
# Method 2: Fake frame ratio
# ---------------------------------------------------------

pred_ratio = (fake_ratio >= THRESHOLD).astype(int)

# ---------------------------------------------------------
# Method 3: Combined score
# ---------------------------------------------------------

combined_score = (0.7 * mean_prob) + (0.3 * fake_ratio)

pred_combined = (combined_score >= THRESHOLD).astype(int)


def evaluate(name, y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(f"Accuracy : {accuracy_score(y_true, pred):.4f}")
    print(f"Precision: {precision_score(y_true, pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_true, pred, zero_division=0):.4f}")

    print("\nConfusion Matrix:")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")


evaluate(
    "METHOD 1: MEAN PROBABILITY",
    y_true,
    pred_mean
)

evaluate(
    "METHOD 2: FAKE FRAME RATIO",
    y_true,
    pred_ratio
)

evaluate(
    "METHOD 3: COMBINED SCORE (70% MEAN + 30% RATIO)",
    y_true,
    pred_combined
)

# ---------------------------------------------------------
# Show cases where methods disagree
# ---------------------------------------------------------

result = df[[
    "video",
    "true_label",
    "mean_probability_fake",
    "fake_frame_ratio"
]].copy()

result["mean_prediction"] = np.where(
    result["mean_probability_fake"] >= THRESHOLD,
    "FAKE",
    "REAL"
)

result["ratio_prediction"] = np.where(
    result["fake_frame_ratio"] >= THRESHOLD,
    "FAKE",
    "REAL"
)

result["combined_score"] = (
    0.7 * result["mean_probability_fake"]
    + 0.3 * result["fake_frame_ratio"]
)

result["combined_prediction"] = np.where(
    result["combined_score"] >= THRESHOLD,
    "FAKE",
    "REAL"
)

result["mean_correct"] = (
    result["mean_prediction"].str.upper()
    == result["true_label"].str.upper()
)

result["ratio_correct"] = (
    result["ratio_prediction"].str.upper()
    == result["true_label"].str.upper()
)

result["combined_correct"] = (
    result["combined_prediction"].str.upper()
    == result["true_label"].str.upper()
)

disagreements = result[
    (result["mean_prediction"] != result["ratio_prediction"])
    |
    (result["mean_prediction"] != result["combined_prediction"])
]

print("\n" + "=" * 70)
print("METHOD DISAGREEMENTS")
print("=" * 70)

print(
    disagreements.to_string(index=False)
)

output = r"results\ff_full_video_evaluation\aggregation_analysis.csv"

result.to_csv(output, index=False)

print("\nSaved:")
print(output)

print("=" * 70)