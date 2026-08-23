from pathlib import Path
import sys


# ---------------------------------------------------------
# Make project root available when this file is executed as:
# python src\predict_video.py ...
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference import get_detector


def predict_video_frames(video_folder, threshold=0.65):
    """
    Predict a FaceForensics++ extracted-frame folder as one video.

    The folder should contain frame_*.png files.
    """

    video_folder = Path(video_folder)

    if not video_folder.exists():
        raise FileNotFoundError(
            f"Video/frame folder not found: {video_folder}"
        )

    frame_paths = sorted(video_folder.glob("frame_*.png"))

    if not frame_paths:
        raise FileNotFoundError(
            f"No frame_*.png files found in: {video_folder}"
        )

    print("=" * 70)
    print("VIDEO-LEVEL DEEPFAKE DETECTION")
    print("=" * 70)
    print(f"Folder    : {video_folder}")
    print(f"Frames    : {len(frame_paths)}")
    print(f"Threshold : {threshold}")
    print("=" * 70)

    detector = get_detector()

    probabilities = []
    predictions = []

    processed = 0
    failed = 0

    for index, frame_path in enumerate(frame_paths, start=1):

        try:
            result = detector.predict_image(str(frame_path))

            probability = float(result["probability_fake"])
            prediction = result["prediction"]

            probabilities.append(probability)
            predictions.append(prediction)

            processed += 1

            if (
                index <= 5
                or index % 10 == 0
                or index == len(frame_paths)
            ):
                print(
                    f"Frame {index:4d}/{len(frame_paths)} | "
                    f"Fake probability: {probability:.4f} | "
                    f"Prediction: {prediction}"
                )

        except Exception as exc:

            failed += 1

            print(
                f"WARNING: Failed frame "
                f"{frame_path.name}: {exc}"
            )

    if not probabilities:
        raise RuntimeError(
            "No frames were successfully processed."
        )

    # -----------------------------------------------------
    # Calculate video-level statistics
    # -----------------------------------------------------

    mean_probability = (
        sum(probabilities) / len(probabilities)
    )

    fake_frame_count = sum(
        1
        for probability in probabilities
        if probability >= threshold
    )

    fake_frame_ratio = (
        fake_frame_count / len(probabilities)
    )

    if mean_probability >= threshold:
        final_prediction = "FAKE"
    else:
        final_prediction = "REAL"

    if final_prediction == "FAKE":
        confidence = mean_probability
    else:
        confidence = 1.0 - mean_probability

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("VIDEO-LEVEL RESULT")
    print("=" * 70)

    print(f"Total frames     : {len(frame_paths)}")
    print(f"Processed frames : {processed}")
    print(f"Failed frames    : {failed}")
    print(f"Mean fake prob.  : {mean_probability:.4f}")
    print(f"Fake frame ratio : {fake_frame_ratio:.4f}")
    print(f"Threshold        : {threshold:.2f}")
    print(f"Prediction       : {final_prediction}")
    print(f"Confidence       : {confidence:.4f}")

    print("=" * 70)

    return {
        "video_folder": str(video_folder),
        "total_frames": len(frame_paths),
        "processed_frames": processed,
        "failed_frames": failed,
        "mean_probability_fake": mean_probability,
        "fake_frame_ratio": fake_frame_ratio,
        "threshold": threshold,
        "prediction": final_prediction,
        "confidence": confidence,
    }


def main():

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            'python src\\predict_video.py '
            '"path\\to\\video_frames"'
        )

        print()
        print("Example:")
        print(
            'python src\\predict_video.py '
            '"data\\ff_faces\\fake\\01_02__outside_talking_still_laughing__YVGY8LOK"'
        )

        return

    video_folder = sys.argv[1]

    predict_video_frames(
        video_folder,
        threshold=0.65,
    )


if __name__ == "__main__":
    main()