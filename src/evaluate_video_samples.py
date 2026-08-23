import sys
from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import get_detector


REAL_DIR = PROJECT_ROOT / "data" / "ff_faces" / "real"
FAKE_DIR = PROJECT_ROOT / "data" / "ff_faces" / "fake"

OUTPUT_DIR = PROJECT_ROOT / "results" / "video_evaluation"
OUTPUT_FILE = OUTPUT_DIR / "sample_results.csv"

NUM_REAL = 5
NUM_FAKE = 5

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_video_folders(directory, limit):
    folders = sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_dir()
        ]
    )

    return folders[:limit]


def get_frame_files(folder):
    frames = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(frames)


def predict_frame_folder(detector, folder):
    frames = get_frame_files(folder)

    if not frames:
        raise RuntimeError(
            f"No image frames found in:\n{folder}"
        )

    fake_probabilities = []
    predictions = []
    failed_frames = 0

    print(f"Frames found: {len(frames)}")

    for index, frame_path in enumerate(frames, start=1):

        try:
            result = detector.predict_image(str(frame_path))

            fake_probability = float(
                result["probability_fake"]
            )

            prediction = result["prediction"]

            fake_probabilities.append(fake_probability)
            predictions.append(prediction)

            if index <= 5 or index == len(frames):
                print(
                    f"Frame {index:4d}/{len(frames)} | "
                    f"Fake probability: {fake_probability:.4f} | "
                    f"Prediction: {prediction}"
                )

        except Exception as exc:
            failed_frames += 1

            print(
                f"Frame {index:4d}/{len(frames)} | "
                f"ERROR: {exc}"
            )

    if not fake_probabilities:
        raise RuntimeError(
            f"No frames could be processed:\n{folder}"
        )

    mean_fake_probability = (
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

    if mean_fake_probability >= detector.threshold:
        video_prediction = "FAKE"
        confidence = mean_fake_probability
    else:
        video_prediction = "REAL"
        confidence = 1.0 - mean_fake_probability

    return {
        "total_frames": len(frames),
        "processed_frames": len(fake_probabilities),
        "failed_frames": failed_frames,
        "mean_fake_probability": mean_fake_probability,
        "fake_frame_ratio": fake_frame_ratio,
        "prediction": video_prediction,
        "confidence": confidence,
    }


def evaluate_folder(
    detector,
    folder,
    true_label,
):
    print()
    print("-" * 70)
    print(f"TRUE LABEL : {true_label}")
    print(f"VIDEO      : {folder.name}")
    print("-" * 70)

    result = predict_frame_folder(
        detector,
        folder,
    )

    prediction = result["prediction"]

    correct = prediction == true_label

    print()
    print(
        f"Video prediction : {prediction}"
    )

    print(
        f"Mean fake prob.  : "
        f"{result['mean_fake_probability']:.4f}"
    )

    print(
        f"Fake frame ratio : "
        f"{result['fake_frame_ratio']:.4f}"
    )

    print(
        f"Confidence       : "
        f"{result['confidence']:.4f}"
    )

    print(
        f"Correct          : {correct}"
    )

    return {
        "video": folder.name,
        "true_label": true_label,
        "prediction": prediction,
        "total_frames": result["total_frames"],
        "processed_frames": result["processed_frames"],
        "failed_frames": result["failed_frames"],
        "mean_fake_probability": result[
            "mean_fake_probability"
        ],
        "fake_frame_ratio": result[
            "fake_frame_ratio"
        ],
        "confidence": result["confidence"],
        "correct": correct,
    }


def main():

    print("=" * 70)
    print("VIDEO-LEVEL SAMPLE EVALUATION")
    print("=" * 70)

    print(f"Real directory: {REAL_DIR}")
    print(f"Fake directory: {FAKE_DIR}")
    print()

    if not REAL_DIR.exists():
        print(
            f"ERROR: Real directory does not exist:\n"
            f"{REAL_DIR}"
        )
        return

    if not FAKE_DIR.exists():
        print(
            f"ERROR: Fake directory does not exist:\n"
            f"{FAKE_DIR}"
        )
        return

    real_videos = get_video_folders(
        REAL_DIR,
        NUM_REAL,
    )

    fake_videos = get_video_folders(
        FAKE_DIR,
        NUM_FAKE,
    )

    print(
        f"Real videos selected: "
        f"{len(real_videos)}"
    )

    print(
        f"Fake videos selected: "
        f"{len(fake_videos)}"
    )

    detector = get_detector()

    results = []

    print()
    print("=" * 70)
    print("EVALUATING REAL VIDEOS")
    print("=" * 70)

    for folder in real_videos:

        try:
            result = evaluate_folder(
                detector,
                folder,
                "REAL",
            )

            results.append(result)

        except Exception as exc:
            print()
            print(
                f"ERROR processing {folder.name}:"
            )
            print(exc)

    print()
    print("=" * 70)
    print("EVALUATING FAKE VIDEOS")
    print("=" * 70)

    for folder in fake_videos:

        try:
            result = evaluate_folder(
                detector,
                folder,
                "FAKE",
            )

            results.append(result)

        except Exception as exc:
            print()
            print(
                f"ERROR processing {folder.name}:"
            )
            print(exc)

    if not results:
        print()
        print("No videos were successfully evaluated.")
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "video",
            "true_label",
            "prediction",
            "total_frames",
            "processed_frames",
            "failed_frames",
            "mean_fake_probability",
            "fake_frame_ratio",
            "confidence",
            "correct",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    total = len(results)

    correct = sum(
        1
        for result in results
        if result["correct"]
    )

    incorrect = total - correct

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("FINAL SAMPLE EVALUATION")
    print("=" * 70)

    print(f"Total videos : {total}")
    print(f"Correct      : {correct}")
    print(f"Incorrect    : {incorrect}")

    print(
        f"Accuracy     : "
        f"{accuracy:.4f}"
    )

    print(
        f"Accuracy     : "
        f"{accuracy * 100:.2f}%"
    )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()