from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEOS = [
    "01__walking_and_outside_surprised",
    "02__walking_and_outside_surprised",
    "02__meeting_serious",
    "03__meeting_serious",
    "03__walking_down_indoor_hall_disgust",
]

INPUT_ROOT = PROJECT_ROOT / "data" / "ff_faces" / "real"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "false_positive_faces"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

for video_name in VIDEOS:
    video_dir = INPUT_ROOT / video_name
    output_dir = OUTPUT_ROOT / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(video_dir.glob("*.png"))

    print("=" * 70)
    print(f"VIDEO: {video_name}")
    print(f"Frames: {len(frames)}")

    for frame_path in frames[:5]:
        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Could not read: {frame_path}")
            continue

        # Save original frame for inspection
        output_path = output_dir / frame_path.name
        cv2.imwrite(str(output_path), image)

        print(f"Saved: {output_path}")

print("=" * 70)
print("Inspection frames created.")
print(f"Output: {OUTPUT_ROOT}")