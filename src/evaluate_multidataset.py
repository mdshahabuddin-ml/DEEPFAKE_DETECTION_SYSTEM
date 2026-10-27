"""
src/evaluate_multidataset.py

Evaluate the saved Xception model on the multi-dataset test split.
"""

import sys
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import (
    get_multidataset_splits,
    DeepfakeDataset,
    get_val_transforms,
)
from src.model import build_model


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = "models/best_xception_multidataset.pth"

INDIAN_ROOT = r"C:\Users\mdsha\Documents\Indian-deepfake-face"

WILD_ROOT = (
    r"C:\Users\mdsha\OneDrive\Desktop"
    r"\Wild Deepfake Dataset (2)\thesis_test_frames"
)

DFDC_ROOT = (
    r"C:\Users\mdsha\OneDrive\Desktop"
    r"\deepfake_detection_system\data\dfdc_faces"
    r"\DFDC Dtata_set_kaggle"
)

DFDC_METADATA = (
    r"C:\Users\mdsha\OneDrive\Desktop"
    r"\DFDC_dataset\DFDC Dtata_set_kaggle"
    r"\metadata.json"
)

FF_ROOT = (
    r"C:\Users\mdsha\OneDrive\Desktop"
    r"\deepfake_detection_system\data\ff_faces"
)

CELEBDF_ROOT = (
    r"C:\Users\mdsha\OneDrive\Desktop"
    r"\deepfake_detection_system\data\celeb-df-v2"
)

IMG_SIZE = 299
BATCH_SIZE = 8

# IMPORTANT:
# Start with a small subset because your machine is CPU-only.
MAX_TEST_SAMPLES = 9282

SEED = 42


# ============================================================
# Main
# ============================================================

def main():

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")

    print("=" * 60)
    print("MULTI-DATASET XCEPTION EVALUATION")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Maximum test samples: {MAX_TEST_SAMPLES}")

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint_path = Path(CHECKPOINT)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    from src.checkpoint import safe_torch_load
    checkpoint = safe_torch_load(
        checkpoint_path,
        map_location=device
    )

    print("\nCheckpoint:")
    print(f"  Epoch: {checkpoint.get('epoch')}")
    print(f"  Validation AUC: {checkpoint.get('val_auc')}")
    print(f"  Model: {checkpoint.get('model_name')}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        "xception",
        pretrained=False,
        dropout=0.5,
        device=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("\nXception checkpoint loaded successfully.")

    # --------------------------------------------------------
    # Build test split
    # --------------------------------------------------------

    print("\nCreating multi-dataset test split...")

    _, _, test_samples = get_multidataset_splits(
        indian_root=INDIAN_ROOT,
        wild_root=WILD_ROOT,
        dfdc_root=DFDC_ROOT,
        dfdc_metadata=DFDC_METADATA,
        ff_root=FF_ROOT,
        celebdf_root=CELEBDF_ROOT,
        max_frames_per_video=20,
        seed=SEED,
    )

    print(f"\nFull test set: {len(test_samples)} samples")

    # --------------------------------------------------------
    # Limit test samples
    # --------------------------------------------------------

    rng = random.Random(SEED)

    if len(test_samples) > MAX_TEST_SAMPLES:
        test_samples = rng.sample(
            test_samples,
            MAX_TEST_SAMPLES
        )

    print(
        f"Evaluation subset: "
        f"{len(test_samples)} samples"
    )

    real_count = sum(
        label == 0
        for _, label in test_samples
    )

    fake_count = sum(
        label == 1
        for _, label in test_samples
    )

    print(f"  REAL: {real_count}")
    print(f"  FAKE: {fake_count}")

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------

    dataset = DeepfakeDataset(
        test_samples,
        transform=get_val_transforms(IMG_SIZE),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    print("\nRunning evaluation...")

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(loader):

            images = images.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy().flatten()

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            all_probabilities.extend(
                probabilities.tolist()
            )

            all_predictions.extend(
                predictions.tolist()
            )

            all_labels.extend(
                labels.numpy().tolist()
            )

            if batch_idx % 10 == 0:
                print(
                    f"  Batch "
                    f"{batch_idx + 1}/"
                    f"{len(loader)}"
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

    print("\n")
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["REAL", "FAKE"],
            zero_division=0
        )
    )

    print("=" * 60)


if __name__ == "__main__":
    main()