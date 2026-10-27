"""
src/realtime_webcam.py
Real-time deepfake detection from a live webcam feed.

Opens your webcam, detects the face in every frame, runs the trained
Xception (or EfficientNet / ViT) detector on the face crop, and draws a
live REAL / FAKE overlay with the fake-probability and FPS.

Run from the project root:
    python -m src.realtime_webcam
    python -m src.realtime_webcam --checkpoint models/best_xception_celebdf.pth --threshold 0.5

Controls (while the window is focused):
    q or ESC  -> quit
    m         -> toggle mirror (selfie view)
    h         -> toggle the info/help banner
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

# Support both "python -m src.realtime_webcam" and "python src/realtime_webcam.py"
try:
    from src.model import build_model, load_checkpoint
    from src.dataset import get_val_transforms
    from src.inference import FaceDetector
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.model import build_model, load_checkpoint
    from src.dataset import get_val_transforms
    from src.inference import FaceDetector


# ---------------------------------------------------------------------------
# Defaults -- keep these in sync with app.py
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "xception"
DEFAULT_CHECKPOINT = "models/best_xception_celebdf.pth"
DEFAULT_IMG_SIZE = 299
DEFAULT_THRESHOLD = 0.5


def load_detector(model_name, checkpoint_path, device):
    """Build the model and load trained weights, ready for inference."""
    model = build_model(model_name, pretrained=False, device=device)
    model = load_checkpoint(model, checkpoint_path, device=device)
    model.to(device)
    model.eval()
    return model


def load_face_cascade():
    """
    Return a face detector. Prefers OpenCV's fast Haar cascade and falls back
    to MTCNN on builds without Haar support (e.g. OpenCV 5). The returned
    object exposes largest_face_box(bgr) and is accepted by detect_largest_face.
    """
    return FaceDetector(backend="auto")


def detect_largest_face(bgr_frame, face_detector, pad_ratio=0.2):
    """
    Return (x0, y0, x1, y1) of the largest face, padded, or None.

    `face_detector` is a src.inference.FaceDetector (padding is applied inside
    it via FACE_MARGIN); `pad_ratio` is kept for backward compatibility.
    """
    return face_detector.largest_face_box(bgr_frame)


@torch.no_grad()
def predict_fake_prob(face_bgr, model, device, transform):
    """Return the fake probability in [0, 1] for a single face crop (BGR)."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(image=face_rgb)["image"].unsqueeze(0).to(device)
    prob = torch.sigmoid(model(tensor)).item()
    return prob


def draw_overlay(frame, box, prob, threshold, smoothed_prob, fps, show_help,
                 uncertain_margin=0.10):
    """Draw the face box, label, probability bar, and status text in-place.

    Borderline predictions within ±uncertain_margin of the threshold render
    as UNCERTAIN (amber) instead of forcing REAL/FAKE.
    """
    h_img, w_img = frame.shape[:2]

    if box is not None and prob is not None:
        x0, y0, x1, y1 = box
        distance = abs(float(smoothed_prob) - threshold)
        if distance <= uncertain_margin:
            color = (0, 165, 255)  # BGR: amber
            label = "UNCERTAIN"
        elif smoothed_prob >= threshold:
            color = (0, 0, 255)    # BGR: red
            label = "FAKE"
        else:
            color = (0, 200, 0)    # BGR: green
            label = "REAL"

        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

        # Label chip above the box
        text = f"{label}  {smoothed_prob * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        chip_y0 = max(0, y0 - th - 12)
        cv2.rectangle(frame, (x0, chip_y0), (x0 + tw + 12, y0), color, -1)
        cv2.putText(
            frame, text, (x0 + 6, y0 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Confidence bar at the bottom of the box
        bar_w = x1 - x0
        fill = int(bar_w * min(max(smoothed_prob, 0.0), 1.0))
        cv2.rectangle(frame, (x0, y1 + 6), (x1, y1 + 16), (60, 60, 60), -1)
        cv2.rectangle(frame, (x0, y1 + 6), (x0 + fill, y1 + 16), color, -1)
    else:
        cv2.putText(
            frame, "No face detected", (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA
        )

    # FPS (top-right)
    fps_text = f"{fps:4.1f} FPS"
    (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(
        frame, fps_text, (w_img - fw - 16, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )

    if show_help:
        banner = "q/ESC quit   m mirror   h hide help"
        cv2.putText(
            frame, banner, (16, h_img - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA
        )


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint}\n"
            f"Train a model first, or pass --checkpoint <path-to-.pth>."
        )

    print(f"Loading {args.model} on {device} ...")
    model = load_detector(args.model, str(checkpoint), device)
    transform = get_val_transforms(args.img_size)
    face_cascade = load_face_cascade()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # CAP_DSHOW = faster start on Windows
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera index {args.camera}. "
            f"Try a different --camera value (0, 1, 2, ...)."
        )

    # Rolling window smooths per-frame jitter into a stable prediction.
    prob_history = deque(maxlen=args.smooth)
    last_box = None
    last_prob = None
    mirror = args.mirror
    show_help = True

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    fps_frames = 0

    window = "Deepfake Detector - Live"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Running. Focus the window and press 'q' or ESC to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera; stopping.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            box = detect_largest_face(frame, face_cascade)

            # Run the (heavy) model only every N frames to keep the feed smooth.
            run_inference = box is not None and (frame_count % args.every_n == 0)
            if run_inference:
                x0, y0, x1, y1 = box
                face = frame[y0:y1, x0:x1]
                if face.size > 0:
                    last_prob = predict_fake_prob(face, model, device, transform)
                    prob_history.append(last_prob)
            last_box = box

            smoothed = float(np.mean(prob_history)) if prob_history else None

            # FPS update roughly once per second
            fps_frames += 1
            now = time.time()
            if now - fps_timer >= 0.5:
                fps = fps_frames / (now - fps_timer)
                fps_timer = now
                fps_frames = 0

            draw_overlay(
                frame, last_box,
                last_prob if last_box is not None else None,
                args.threshold,
                smoothed if smoothed is not None else 0.0,
                fps, show_help,
            )

            cv2.imshow(window, frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):        # q or ESC
                break
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("h"):
                show_help = not show_help

            # Stop if the user closed the window with the X button
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


def parse_args():
    p = argparse.ArgumentParser(description="Real-time webcam deepfake detection")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   choices=["xception", "efficientnet", "vit"])
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Path to trained .pth weights")
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Fake decision threshold in [0, 1]")
    p.add_argument("--every-n", type=int, default=2,
                   help="Run model inference every N frames (higher = smoother/faster)")
    p.add_argument("--smooth", type=int, default=8,
                   help="Rolling-average window (frames) for a stable prediction")
    p.add_argument("--mirror", action="store_true",
                   help="Start with a mirrored (selfie) view")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
