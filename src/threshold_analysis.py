from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CSV_PATH = Path("results/ff_full_video_evaluation/all_video_results.csv")

df = pd.read_csv(CSV_PATH)

# Convert labels to binary:
# REAL = 0
# FAKE = 1
y_true = (df["true_label"].str.upper() == "FAKE").astype(int)

# Probability of FAKE
y_prob = df["mean_probability_fake"].astype(float)

results = []

for threshold in np.arange(0.30, 0.86, 0.01):
    y_pred = (y_prob >= threshold).astype(int)

    results.append({
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    })

results_df = pd.DataFrame(results)

best = results_df.loc[results_df["f1"].idxmax()]

print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print("\nCurrent threshold: 0.65")
current = results_df[results_df["threshold"] == 0.65].iloc[0]
print(f"Accuracy : {current['accuracy']:.4f}")
print(f"Precision: {current['precision']:.4f}")
print(f"Recall   : {current['recall']:.4f}")
print(f"F1       : {current['f1']:.4f}")

print("\nBest threshold by F1:")
print(f"Threshold: {best['threshold']:.2f}")
print(f"Accuracy : {best['accuracy']:.4f}")
print(f"Precision: {best['precision']:.4f}")
print(f"Recall   : {best['recall']:.4f}")
print(f"F1       : {best['f1']:.4f}")

print("\nTop 10 thresholds by F1:")
print(
    results_df
    .sort_values("f1", ascending=False)
    .head(10)
    .to_string(index=False)
)

output = Path("results/ff_full_video_evaluation/threshold_analysis.csv")
results_df.to_csv(output, index=False)

print(f"\nResults saved to:")
print(output)
print("=" * 70)