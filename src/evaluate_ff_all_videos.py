import sys
import csv
from pathlib import Path
from collections import Counter

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference import get_detector


# ============================================================
# CONFIGURATION
# ============================================================

REAL_DIR = PROJECT_ROOT / "data" / "ff_faces" / "real"
FAKE_DIR = PROJECT_ROOT / "data" / "ff_faces" / "fake"

OUTPUT_DIR = PROJECT_ROOT / "results" / "ff_full_video_evaluation"

RESULTS_CSV = OUTPUT_DIR / "all_video_results.csv"
SUMMARY_TXT = OUTPUT_DIR / "summary.txt"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# FIND VIDEO FOLDERS
# ============================================================

def get_video_folders(directory):
    if not directory.exists():
        return []

    folders = [
        path
        for path in directory.iterdir()
        if path.is_dir()
    ]

    return sorted(folders)


# ============================================================
# FIND FRAMES
# ============================================================

def get_frame_files(folder):
    frames = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(frames)


# ============================================================
# EVALUATE ONE VIDEO FOLDER
# ============================================================

def evaluate_video(detector, folder, true_label):

    frames = get_frame_files(folder)

    if not frames:
        print(f"WARNING: No frames found: {folder}")
        return None

    fake_probabilities = []

    failed_frames = 0

    for frame_number, frame_path in enumerate(
        frames,
        start=1
    ):

        try:

            result = detector.predict_image(
                str(frame_path)
            )

            probability = float(
                result["probability_fake"]
            )

            fake_probabilities.append(
                probability
            )

        except Exception as exc:

            failed_frames += 1

            print(
                f"    Frame error: "
                f"{frame_path.name} -> {exc}"
            )

    if not fake_probabilities:
        return None

    # --------------------------------------------------------
    # VIDEO-LEVEL STATISTICS
    # --------------------------------------------------------

    mean_probability = (
        sum(fake_probabilities)
        / len(fake_probabilities)
    )

    fake_frame_count = sum(
        1
        for probability in fake_probabilities
        if probability >= detector.threshold
    )

    fake_frame_ratio = (
        fake_frame_count
        / len(fake_probabilities)
    )

    # --------------------------------------------------------
    # VIDEO-LEVEL DECISION
    # --------------------------------------------------------

    if mean_probability >= detector.threshold:
        prediction = "FAKE"
        confidence = mean_probability
    else:
        prediction = "REAL"
        confidence = 1.0 - mean_probability

    correct = prediction == true_label

    return {
        "video": folder.name,
        "true_label": true_label,
        "prediction": prediction,
        "total_frames": len(frames),
        "processed_frames": len(fake_probabilities),
        "failed_frames": failed_frames,
        "mean_probability_fake": mean_probability,
        "fake_frame_ratio": fake_frame_ratio,
        "confidence": confidence,
        "correct": correct,
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "video",
        "true_label",
        "prediction",
        "total_frames",
        "processed_frames",
        "failed_frames",
        "mean_probability_fake",
        "fake_frame_ratio",
        "confidence",
        "correct",
    ]

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(results):

    true_labels = [
        result["true_label"]
        for result in results
    ]

    predictions = [
        result["prediction"]
        for result in results
    ]

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0

    for true_label, prediction in zip(
        true_labels,
        predictions,
    ):

        if true_label == "REAL" and prediction == "REAL":
            true_negative += 1

        elif true_label == "REAL" and prediction == "FAKE":
            false_positive += 1

        elif true_label == "FAKE" and prediction == "REAL":
            false_negative += 1

        elif true_label == "FAKE" and prediction == "FAKE":
            true_positive += 1

    total = len(results)

    correct = (
        true_negative
        + true_positive
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
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
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
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
# SAVE SUMMARY
# ============================================================

def save_summary(metrics, detector):

    with open(
        SUMMARY_TXT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "FACEFORENSICS++ FULL VIDEO EVALUATION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Model: {detector.model_name}\n"
        )

        file.write(
            f"Checkpoint: {detector.checkpoint_path}\n"
        )

        file.write(
            f"Threshold: {detector.threshold}\n"
        )

        file.write(
            f"Device: {detector.device}\n\n"
        )

        file.write(
            f"Total videos: {metrics['total']}\n"
        )

        file.write(
            f"Correct: {metrics['correct']}\n"
        )

        file.write(
            f"Incorrect: {metrics['incorrect']}\n\n"
        )

        file.write(
            f"Accuracy: {metrics['accuracy']:.4f}\n"
        )

        file.write(
            f"Precision: {metrics['precision']:.4f}\n"
        )

        file.write(
            f"Recall: {metrics['recall']:.4f}\n"
        )

        file.write(
            f"F1 Score: {metrics['f1']:.4f}\n\n"
        )

        file.write(
            "CONFUSION MATRIX\n"
        )

        file.write(
            "-" * 40 + "\n"
        )

        file.write(
            f"True Negative : {metrics['true_negative']}\n"
        )

        file.write(
            f"False Positive: {metrics['false_positive']}\n"
        )

        file.write(
            f"False Negative: {metrics['false_negative']}\n"
        )

        file.write(
            f"True Positive : {metrics['true_positive']}\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FACEFORENSICS++ FULL VIDEO EVALUATION")
    print("=" * 70)

    print()
    print(f"Real directory:")
    print(REAL_DIR)

    print()
    print(f"Fake directory:")
    print(FAKE_DIR)

    print()

    if not REAL_DIR.exists():
        print("ERROR: REAL directory does not exist.")
        return

    if not FAKE_DIR.exists():
        print("ERROR: FAKE directory does not exist.")
        return

    # --------------------------------------------------------
    # FIND ALL VIDEOS
    # --------------------------------------------------------

    real_videos = get_video_folders(
        REAL_DIR
    )

    fake_videos = get_video_folders(
        FAKE_DIR
    )

    print(
        f"Real videos found: {len(real_videos)}"
    )

    print(
        f"Fake videos found: {len(fake_videos)}"
    )

    total_videos = (
        len(real_videos)
        + len(fake_videos)
    )

    print(
        f"Total videos: {total_videos}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    detector = get_detector()

    print()
    print(
        f"Using threshold: {detector.threshold}"
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("EVALUATING REAL VIDEOS")
    print("=" * 70)

    for index, folder in enumerate(
        real_videos,
        start=1
    ):

        print()
        print(
            f"[REAL {index}/{len(real_videos)}] "
            f"{folder.name}"
        )

        result = evaluate_video(
            detector,
            folder,
            "REAL",
        )

        if result is not None:

            results.append(result)

            print(
                f"    Mean fake probability: "
                f"{result['mean_probability_fake']:.4f}"
            )

            print(
                f"    Fake frame ratio: "
                f"{result['fake_frame_ratio']:.4f}"
            )

            print(
                f"    Prediction: "
                f"{result['prediction']}"
            )

            print(
                f"    Correct: "
                f"{result['correct']}"
            )

    print()
    print("=" * 70)
    print("EVALUATING FAKE VIDEOS")
    print("=" * 70)

    for index, folder in enumerate(
        fake_videos,
        start=1
    ):

        print()
        print(
            f"[FAKE {index}/{len(fake_videos)}] "
            f"{folder.name}"
        )

        result = evaluate_video(
            detector,
            folder,
            "FAKE",
        )

        if result is not None:

            results.append(result)

            print(
                f"    Mean fake probability: "
                f"{result['mean_probability_fake']:.4f}"
            )

            print(
                f"    Fake frame ratio: "
                f"{result['fake_frame_ratio']:.4f}"
            )

            print(
                f"    Prediction: "
                f"{result['prediction']}"
            )

            print(
                f"    Correct: "
                f"{result['correct']}"
            )

    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    if not results:

        print()
        print("ERROR: No videos were successfully evaluated.")
        return

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        results
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    save_csv(results)

    save_summary(
        metrics,
        detector,
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL FF++ VIDEO-LEVEL RESULTS")
    print("=" * 70)

    print(
        f"Total videos : {metrics['total']}"
    )

    print(
        f"Correct      : {metrics['correct']}"
    )

    print(
        f"Incorrect    : {metrics['incorrect']}"
    )

    print()

    print(
        f"Accuracy     : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Accuracy     : "
        f"{metrics['accuracy'] * 100:.2f}%"
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

    print()
    print("Confusion Matrix:")
    print(
        f"[[{metrics['true_negative']:4d} "
        f"{metrics['false_positive']:4d}]"
    )

    print(
        f" [{metrics['false_negative']:4d} "
        f"{metrics['true_positive']:4d}]]"
    )

    print()
    print(
        f"CSV saved to:"
    )

    print(
        RESULTS_CSV
    )

    print()
    print(
        f"Summary saved to:"
    )

    print(
        SUMMARY_TXT
    )

    print()
    print("=" * 70)
    print("FULL EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()