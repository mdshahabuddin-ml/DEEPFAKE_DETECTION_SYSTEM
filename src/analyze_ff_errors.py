"""
src/analyze_ff_errors.py

FaceForensics++ error analysis for the trained Xception model.

The script:
1. Loads the trained Xception checkpoint.
2. Loads the extracted FF++ face dataset.
3. Creates a balanced FF++ evaluation set.
4. Runs inference.
5. Identifies:
       - False Positives
       - False Negatives
       - Correct predictions
6. Saves predictions and error samples.

Run:
    python src/analyze_ff_errors.py
"""

import sys
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.dataset import (
    load_ff_faces_samples,
    DeepfakeDataset,
    get_val_transforms,
)

from src.model import build_model


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT /
    "models" /
    "best_xception_multidataset.pth"
)

FF_ROOT = (
    PROJECT_ROOT /
    "data" /
    "ff_faces"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "ff_errors"
)

MAX_FRAMES_PER_VIDEO = 20

MAX_SAMPLES = 2000

BATCH_SIZE = 32

NUM_WORKERS = 0

IMG_SIZE = 299

THRESHOLD = 0.5


# ============================================================
# Create output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Device
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load model
# ============================================================

def load_trained_model():

    print("=" * 60)
    print("LOADING XCEPTION MODEL")
    print("=" * 60)

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    from src.checkpoint import safe_torch_load
    checkpoint = safe_torch_load(
        CHECKPOINT,
        map_location=DEVICE
    )

    print(
        f"Epoch: {checkpoint.get('epoch')}"
    )

    print(
        f"Validation AUC: "
        f"{checkpoint.get('val_auc')}"
    )

    print(
        f"Model: "
        f"{checkpoint.get('model_name')}"
    )

    model = build_model(
        "xception",
        pretrained=False,
        dropout=0.5,
        device=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "\nXception checkpoint loaded successfully."
    )

    return model


# ============================================================
# Load FF++ samples
# ============================================================

def load_ff_dataset():

    print("\n" + "=" * 60)
    print("LOADING FACEFORENSICS++")
    print("=" * 60)

    samples = load_ff_faces_samples(
        FF_ROOT,
        max_frames_per_video=MAX_FRAMES_PER_VIDEO,
    )

    print(
        f"\nTotal FF++ samples: {len(samples)}"
    )

    return samples


# ============================================================
# Balance FF++ samples
# ============================================================

def balance_samples(samples):

    real = [
        sample
        for sample in samples
        if sample[1] == 0
    ]

    fake = [
        sample
        for sample in samples
        if sample[1] == 1
    ]

    print("\nBefore balancing:")

    print(
        f"REAL: {len(real)}"
    )

    print(
        f"FAKE: {len(fake)}"
    )

    n = min(
        len(real),
        len(fake)
    )

    # Use deterministic selection
    real = real[:n]
    fake = fake[:n]

    samples = real + fake

    print("\nAfter balancing:")

    print(
        f"REAL: {len(real)}"
    )

    print(
        f"FAKE: {len(fake)}"
    )

    print(
        f"TOTAL: {len(samples)}"
    )

    return samples


# ============================================================
# Limit evaluation size
# ============================================================

def limit_samples(samples):

    if len(samples) <= MAX_SAMPLES:

        return samples

    real = [
        sample
        for sample in samples
        if sample[1] == 0
    ]

    fake = [
        sample
        for sample in samples
        if sample[1] == 1
    ]

    per_class = MAX_SAMPLES // 2

    samples = (
        real[:per_class]
        +
        fake[:per_class]
    )

    print(
        f"\nEvaluation limited to "
        f"{len(samples)} samples."
    )

    return samples


# ============================================================
# Run inference
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    samples,
):

    print("\n" + "=" * 60)
    print("RUNNING FF++ INFERENCE")
    print("=" * 60)

    dataset = DeepfakeDataset(
        samples,
        transform=get_val_transforms(
            IMG_SIZE
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    all_labels = []
    all_probs = []
    all_paths = []

    total_batches = len(loader)

    for batch_index, (images, labels) in enumerate(
        loader,
        start=1
    ):

        images = images.to(
            DEVICE
        )

        logits = model(
            images
        )

        probabilities = torch.sigmoid(
            logits
        ).detach().cpu().numpy().flatten()

        labels = labels.numpy().flatten()

        start_index = (
            (batch_index - 1)
            * BATCH_SIZE
        )

        end_index = (
            start_index
            + len(labels)
        )

        batch_samples = samples[
            start_index:end_index
        ]

        batch_paths = [
            path
            for path, _ in batch_samples
        ]

        all_labels.extend(
            labels.tolist()
        )

        all_probs.extend(
            probabilities.tolist()
        )

        all_paths.extend(
            batch_paths
        )

        if (
            batch_index == 1
            or batch_index % 20 == 0
            or batch_index == total_batches
        ):

            print(
                f"  Batch "
                f"{batch_index}/"
                f"{total_batches}"
            )

    return (
        all_paths,
        all_labels,
        all_probs,
    )


# ============================================================
# Analyze predictions
# ============================================================

def analyze_predictions(
    paths,
    labels,
    probabilities,
):

    predictions = [
        1
        if probability >= THRESHOLD
        else 0
        for probability in probabilities
    ]

    results = []

    for path, true_label, probability, prediction in zip(
        paths,
        labels,
        probabilities,
        predictions,
    ):

        if (
            true_label == 0
            and prediction == 1
        ):

            result_type = "FALSE_POSITIVE"

        elif (
            true_label == 1
            and prediction == 0
        ):

            result_type = "FALSE_NEGATIVE"

        elif (
            true_label == 0
            and prediction == 0
        ):

            result_type = "TRUE_NEGATIVE"

        else:

            result_type = "TRUE_POSITIVE"

        results.append({
            "path": path,
            "true_label": int(true_label),
            "prediction": int(prediction),
            "probability_fake": float(
                probability
            ),
            "result": result_type,
        })

    return results


# ============================================================
# Calculate metrics
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
):

    predictions = [
        1
        if probability >= THRESHOLD
        else 0
        for probability in probabilities
    ]

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        auc,
        cm,
    )


# ============================================================
# Save predictions CSV
# ============================================================

def save_predictions(results):

    output_file = (
        OUTPUT_DIR /
        "predictions.csv"
    )

    fieldnames = [
        "path",
        "true_label",
        "prediction",
        "probability_fake",
        "result",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Save false positives
# ============================================================

def save_false_positives(results):

    false_positives = [
        result
        for result in results
        if result["result"]
        == "FALSE_POSITIVE"
    ]

    output_file = (
        OUTPUT_DIR /
        "false_positives.csv"
    )

    fieldnames = [
        "path",
        "true_label",
        "prediction",
        "probability_fake",
        "result",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            false_positives
        )

    print(
        f"False positives: "
        f"{len(false_positives)}"
    )

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Save false negatives
# ============================================================

def save_false_negatives(results):

    false_negatives = [
        result
        for result in results
        if result["result"]
        == "FALSE_NEGATIVE"
    ]

    output_file = (
        OUTPUT_DIR /
        "false_negatives.csv"
    )

    fieldnames = [
        "path",
        "true_label",
        "prediction",
        "probability_fake",
        "result",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            false_negatives
        )

    print(
        f"False negatives: "
        f"{len(false_negatives)}"
    )

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Save summary
# ============================================================

def save_summary(
    accuracy,
    precision,
    recall,
    f1,
    auc,
    cm,
    results,
):

    output_file = (
        OUTPUT_DIR /
        "summary.txt"
    )

    false_positives = sum(
        result["result"]
        == "FALSE_POSITIVE"
        for result in results
    )

    false_negatives = sum(
        result["result"]
        == "FALSE_NEGATIVE"
        for result in results
    )

    true_negatives = sum(
        result["result"]
        == "TRUE_NEGATIVE"
        for result in results
    )

    true_positives = sum(
        result["result"]
        == "TRUE_POSITIVE"
        for result in results
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "FACEFORENSICS++ ERROR ANALYSIS\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Total samples: "
            f"{len(results)}\n"
        )

        file.write(
            f"Threshold: "
            f"{THRESHOLD}\n\n"
        )

        file.write(
            "METRICS\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            f"Accuracy : "
            f"{accuracy:.4f}\n"
        )

        file.write(
            f"Precision: "
            f"{precision:.4f}\n"
        )

        file.write(
            f"Recall   : "
            f"{recall:.4f}\n"
        )

        file.write(
            f"F1 Score : "
            f"{f1:.4f}\n"
        )

        file.write(
            f"ROC-AUC  : "
            f"{auc:.4f}\n\n"
        )

        file.write(
            "CONFUSION MATRIX\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            str(cm)
            + "\n\n"
        )

        file.write(
            "ERROR COUNTS\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            f"True Negatives : "
            f"{true_negatives}\n"
        )

        file.write(
            f"False Positives: "
            f"{false_positives}\n"
        )

        file.write(
            f"False Negatives: "
            f"{false_negatives}\n"
        )

        file.write(
            f"True Positives  : "
            f"{true_positives}\n"
        )

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("FACEFORENSICS++ ERROR ANALYSIS")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Maximum samples: "
        f"{MAX_SAMPLES}"
    )

    print(
        f"Threshold: "
        f"{THRESHOLD}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_trained_model()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    samples = load_ff_dataset()

    # --------------------------------------------------------
    # Balance
    # --------------------------------------------------------

    samples = balance_samples(
        samples
    )

    # --------------------------------------------------------
    # Limit
    # --------------------------------------------------------

    samples = limit_samples(
        samples
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    (
        paths,
        labels,
        probabilities,
    ) = run_inference(
        model,
        samples,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    (
        accuracy,
        precision,
        recall,
        f1,
        auc,
        cm,
    ) = calculate_metrics(
        labels,
        probabilities,
    )

    # --------------------------------------------------------
    # Analyze errors
    # --------------------------------------------------------

    results = analyze_predictions(
        paths,
        labels,
        probabilities,
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    save_predictions(
        results
    )

    save_false_positives(
        results
    )

    save_false_negatives(
        results
    )

    save_summary(
        accuracy,
        precision,
        recall,
        f1,
        auc,
        cm,
        results,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FF++ ERROR ANALYSIS RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score : "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    false_positives = sum(
        result["result"]
        == "FALSE_POSITIVE"
        for result in results
    )

    false_negatives = sum(
        result["result"]
        == "FALSE_NEGATIVE"
        for result in results
    )

    print(
        f"\nFalse Positives: "
        f"{false_positives}"
    )

    print(
        f"False Negatives: "
        f"{false_negatives}"
    )

    print("\n")
    print("=" * 60)
    print("ERROR ANALYSIS COMPLETED")
    print("=" * 60)

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()