"""
src/evaluate_per_dataset.py

Evaluate the trained Xception model separately on:
    1. Indian Deepfake Face
    2. WildDeepfake
    3. DFDC
    4. FaceForensics++
    5. Celeb-DF v2

Uses:
    models/best_xception_multidataset.pth

REAL = 0
FAKE = 1
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------------
# Project path
# ------------------------------------------------------------

sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import (
    DeepfakeDataset,
    get_val_transforms,
    load_indian_samples,
    load_wilddeepfake_samples,
    load_dfdc_samples,
    load_ff_faces_samples,
    load_celebdf_samples,
)

from src.model import build_model


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = Path("models/best_xception_multidataset.pth")

INDIAN_ROOT = Path(
    r"C:\Users\mdsha\Documents\Indian-deepfake-face"
)

WILD_ROOT = Path(
    r"C:\Users\mdsha\OneDrive\Desktop\Wild Deepfake Dataset (2)\thesis_test_frames"
)

DFDC_ROOT = Path(
    r"C:\Users\mdsha\OneDrive\Desktop\deepfake_detection_system\data\dfdc_faces\DFDC Dtata_set_kaggle"
)

DFDC_METADATA = Path(
    r"C:\Users\mdsha\OneDrive\Desktop\DFDC_dataset\DFDC Dtata_set_kaggle\metadata.json"
)

FF_ROOT = Path(
    r"C:\Users\mdsha\OneDrive\Desktop\deepfake_detection_system\data\ff_faces"
)

CELEBDF_ROOT = Path(
    r"C:\Users\mdsha\OneDrive\Desktop\deepfake_detection_system\data\celeb-df-v2"
)

BATCH_SIZE = 32
IMG_SIZE = 299
NUM_WORKERS = 0

# To keep this evaluation manageable on your CPU,
# limit the number of samples per dataset.
# Set to None later if you want the complete datasets.
MAX_SAMPLES_PER_DATASET = 2000

MAX_FRAMES_PER_VIDEO = 20


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 60)
    print("LOADING XCEPTION MODEL")
    print("=" * 60)

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    from src.checkpoint import safe_torch_load
    checkpoint = safe_torch_load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model_name = checkpoint.get(
        "model_name",
        "xception"
    )

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Epoch: {checkpoint.get('epoch')}")
    print(f"Validation AUC: {checkpoint.get('val_auc')}")
    print(f"Model: {model_name}")

    model = build_model(
        model_name,
        pretrained=False,
        dropout=0.5,
        device=str(DEVICE),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("\nXception checkpoint loaded successfully.")

    return model


# ============================================================
# BALANCE DATASET
# ============================================================

def balance_samples(samples, dataset_name):

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

    n = min(
        len(real),
        len(fake)
    )

    if n == 0:

        raise ValueError(
            f"{dataset_name} does not contain "
            "both REAL and FAKE samples."
        )

    # Deterministic selection
    real = real[:n]
    fake = fake[:n]

    balanced = real + fake

    return balanced


# ============================================================
# LIMIT DATASET SIZE
# ============================================================

def limit_samples(samples, max_samples):

    if max_samples is None:
        return samples

    if len(samples) <= max_samples:
        return samples

    # Keep equal REAL / FAKE representation
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

    per_class = max_samples // 2

    real = real[:per_class]
    fake = fake[:per_class]

    return real + fake


# ============================================================
# EVALUATE DATASET
# ============================================================

@torch.no_grad()
def evaluate_dataset(
    model,
    samples,
    dataset_name,
):

    print("\n" + "=" * 60)
    print(f"EVALUATING: {dataset_name}")
    print("=" * 60)

    print(f"Samples: {len(samples)}")

    real_count = sum(
        label == 0
        for _, label in samples
    )

    fake_count = sum(
        label == 1
        for _, label in samples
    )

    print(f"REAL: {real_count}")
    print(f"FAKE: {fake_count}")

    dataset = DeepfakeDataset(
        samples,
        transform=get_val_transforms(
            IMG_SIZE
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_batches = len(loader)

    for batch_index, (images, labels) in enumerate(loader, start=1):

        images = images.to(
            DEVICE
        )

        logits = model(images)

        probabilities = torch.sigmoid(
            logits
        ).view(-1)

        predictions = (
            probabilities >= 0.5
        ).long()

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        if (
            batch_index == 1
            or batch_index % 20 == 0
            or batch_index == total_batches
        ):

            print(
                f"  Batch "
                f"{batch_index}/{total_batches}"
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

    except ValueError:

        auc = 0.0

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print(f"{dataset_name} RESULTS")
    print("-" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "REAL",
                "FAKE"
            ],
            zero_division=0,
        )
    )

    return {
        "dataset": dataset_name,
        "samples": len(samples),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("DATASET-WISE XCEPTION EVALUATION")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(
        f"Maximum samples per dataset: "
        f"{MAX_SAMPLES_PER_DATASET}"
    )

    model = load_model()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    print("\nLoading Indian dataset...")

    indian = load_indian_samples(
        str(INDIAN_ROOT)
    )

    print("\nLoading WildDeepfake...")

    wild = load_wilddeepfake_samples(
        str(WILD_ROOT),
        max_frames_per_video=MAX_FRAMES_PER_VIDEO
    )

    print("\nLoading DFDC...")

    dfdc = load_dfdc_samples(
        str(DFDC_ROOT),
        str(DFDC_METADATA),
        max_frames_per_video=MAX_FRAMES_PER_VIDEO
    )

    print("\nLoading FaceForensics++...")

    ff = load_ff_faces_samples(
        str(FF_ROOT),
        max_frames_per_video=MAX_FRAMES_PER_VIDEO
    )

    print("\nLoading Celeb-DF v2...")

    celebdf = load_celebdf_samples(
        str(CELEBDF_ROOT),
        max_frames_per_video=MAX_FRAMES_PER_VIDEO
    )

    # --------------------------------------------------------
    # Balance datasets
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BALANCING DATASETS")
    print("=" * 60)

    datasets = {}

    datasets["Indian"] = balance_samples(
        indian,
        "Indian"
    )

    datasets["WildDeepfake"] = balance_samples(
        wild,
        "WildDeepfake"
    )

    datasets["DFDC"] = balance_samples(
        dfdc,
        "DFDC"
    )

    datasets["FaceForensics++"] = balance_samples(
        ff,
        "FaceForensics++"
    )

    datasets["Celeb-DF v2"] = balance_samples(
        celebdf,
        "Celeb-DF v2"
    )

    # --------------------------------------------------------
    # Limit samples
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LIMITING EVALUATION SIZE")
    print("=" * 60)

    for name in list(datasets.keys()):

        datasets[name] = limit_samples(
            datasets[name],
            MAX_SAMPLES_PER_DATASET
        )

        real = sum(
            label == 0
            for _, label in datasets[name]
        )

        fake = sum(
            label == 1
            for _, label in datasets[name]
        )

        print(
            f"{name}: "
            f"REAL={real}, "
            f"FAKE={fake}, "
            f"TOTAL={len(datasets[name])}"
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = []

    for name, samples in datasets.items():

        result = evaluate_dataset(
            model,
            samples,
            name
        )

        results.append(result)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("FINAL DATASET-WISE RESULTS")
    print("=" * 80)

    print(
        f"{'Dataset':<20}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'AUC':>12}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['dataset']:<20}"
            f"{result['accuracy']:>12.4f}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>12.4f}"
            f"{result['auc']:>12.4f}"
        )

    print("=" * 80)

    print("\nDataset-wise evaluation completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()