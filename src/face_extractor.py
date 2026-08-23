
import os
import argparse
from pathlib import Path
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
import torch

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}


def _save_face(mtcnn, image, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mtcnn(image, save_path=str(output_file))
    return output_file.exists()


def _extract_from_image(mtcnn, img_path, output_path, input_path):
    relative_path = img_path.relative_to(input_path).parent
    output_file = output_path / relative_path / img_path.name
    if output_file.exists():
        return 0

    try:
        img = Image.open(img_path).convert('RGB')
        return 1 if _save_face(mtcnn, img, output_file) else 0
    except Exception as e:
        print(f"Skipping {img_path} due to error: {e}")
        return 0


def _extract_from_video(mtcnn, video_path, output_path, input_path, every_n_frames, max_frames):
    relative_dir = video_path.relative_to(input_path).with_suffix('')
    target_dir = output_path / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping unreadable video: {video_path}")
        return 0

    saved = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n_frames == 0:
            output_file = target_dir / f"frame_{frame_idx:06d}.png"
            if not output_file.exists():
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                try:
                    if _save_face(mtcnn, image, output_file):
                        saved += 1
                except Exception as e:
                    print(f"Skipping frame {frame_idx} in {video_path.name}: {e}")

            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


def extract_faces(input_dir, output_dir, every_n_frames=10, max_frames=30,
                  image_size=299, max_videos=None):
    # Determine device (using GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for face detection: {device}")
    
    # Initialize MTCNN for face extraction
    mtcnn = MTCNN(image_size=image_size, margin=20, keep_all=False, device=device)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        print("Please create this folder and put your raw images/videos inside it.")
        return

    every_n_frames = max(1, int(every_n_frames))
    if max_frames is not None:
        max_frames = int(max_frames)

    print(f"Scanning for images/videos in: {input_path.resolve()}")

    total_saved = 0
    videos_processed = 0
    for media_path in input_path.rglob('*'):
        suffix = media_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            saved = _extract_from_image(mtcnn, media_path, output_path, input_path)
            total_saved += saved
            if saved:
                print(f"Processed image: {media_path}")
        elif suffix in VIDEO_EXTENSIONS:
            if max_videos is not None and videos_processed >= max_videos:
                continue
            saved = _extract_from_video(
                mtcnn=mtcnn,
                video_path=media_path,
                output_path=output_path,
                input_path=input_path,
                every_n_frames=every_n_frames,
                max_frames=max_frames,
            )
            total_saved += saved
            videos_processed += 1
            print(f"Processed video: {media_path.name} -> {saved} faces")

    print(f"Saved {total_saved} face crops to: {output_path.resolve()}")
    if videos_processed:
        print(f"Processed {videos_processed} videos")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces using MTCNN")
    
    # Made these optional by adding default values pointing to your data folders
    parser.add_argument('--input', type=str, default='data/celeb-df-v2',
                        help='Path to input directory containing images/videos')
    parser.add_argument('--output', type=str, default='data/processed/celeb-df-v2',
                        help='Path to output directory to save cropped faces')
    parser.add_argument('--every_n_frames', type=int, default=10,
                        help='Extract one frame every N video frames')
    parser.add_argument('--max_frames', type=int, default=30,
                        help='Maximum saved face crops per video')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Optional limit for quick test runs')
    parser.add_argument('--image_size', type=int, default=299)
    
    args = parser.parse_args()
    
    extract_faces(
        input_dir=args.input, 
        output_dir=args.output, 
        every_n_frames=args.every_n_frames,
        max_frames=args.max_frames,
        max_videos=args.max_videos,
        image_size=args.image_size
    )
    print("\nExtraction job finished!")
   