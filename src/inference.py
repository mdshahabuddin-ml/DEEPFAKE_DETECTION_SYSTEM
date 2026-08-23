"""
src/inference.py

Inference utilities for the Deepfake Detection System.

Provides:
    - DeepfakeDetector
    - FaceDetector
    - ModelNotFoundError
    - get_detector()

Supports:
    - Xception model
    - Image inference
    - Video inference
    - Webcam face inference
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch

from src.config import (
    MODEL_NAME,
    CHECKPOINT_PATH,
    DEVICE,
    IMG_SIZE,
    DEFAULT_THRESHOLD,
    FACE_DETECTOR,
    FACE_MARGIN,
    VIDEO_MAX_FRAMES,
    label_from_prob,
    confidence_from_prob,
)

from src.model import build_model

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:
    raise ImportError(
        "Albumentations is required for inference. "
        "Install it with: pip install albumentations"
    ) from exc


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model checkpoint cannot be found."""


# ============================================================================
# FACE DETECTOR
# ============================================================================

class FaceDetector:
    """
    Face detection helper.

    Supported backends:
        auto
        haar
        mtcnn
    """

    def __init__(
        self,
        backend: str = "auto",
        device: str = DEVICE,
    ):
        self.backend = backend.lower()
        self.device = device

        self.haar = None
        self.mtcnn = None

        if self.backend not in {"auto", "haar", "mtcnn"}:
            raise ValueError(
                "backend must be one of: auto, haar, mtcnn"
            )

        self._initialize()

    def _initialize(self) -> None:
        """Initialize the selected face detector."""

        # ------------------------------------------------------------
        # Haar detector
        # ------------------------------------------------------------

        if self.backend in {"auto", "haar"}:

            haar_path = cv2.data.haarcascades + (
                "haarcascade_frontalface_default.xml"
            )

            if Path(haar_path).exists():

                self.haar = cv2.CascadeClassifier(haar_path)

                if self.haar.empty():
                    self.haar = None

            elif self.backend == "haar":

                raise RuntimeError(
                    "OpenCV Haar cascade was not found."
                )

        # ------------------------------------------------------------
        # MTCNN detector
        # ------------------------------------------------------------

        if self.backend == "mtcnn" or (
            self.backend == "auto" and self.haar is None
        ):

            try:
                from facenet_pytorch import MTCNN

                self.mtcnn = MTCNN(
                    keep_all=True,
                    device=torch.device(self.device),
                )

            except ImportError:

                if self.backend == "mtcnn":
                    raise ImportError(
                        "facenet-pytorch is required for MTCNN. "
                        "Install with: pip install facenet-pytorch"
                    )

                self.mtcnn = None

    # ----------------------------------------------------------------
    # Haar detection
    # ----------------------------------------------------------------

    def _haar_boxes(
        self,
        frame_bgr: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:

        if self.haar is None:
            return []

        gray = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        boxes = []

        for x, y, w, h in faces:

            x0 = int(x)
            y0 = int(y)
            x1 = int(x + w)
            y1 = int(y + h)

            boxes.append(
                (
                    x0,
                    y0,
                    x1,
                    y1,
                )
            )

        return boxes

    # ----------------------------------------------------------------
    # MTCNN detection
    # ----------------------------------------------------------------

    def _mtcnn_boxes(
        self,
        frame_bgr: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:

        if self.mtcnn is None:
            return []

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        boxes, probabilities = self.mtcnn.detect(
            frame_rgb
        )

        if boxes is None:
            return []

        h, w = frame_bgr.shape[:2]

        result = []

        for i, box in enumerate(boxes):

            x0, y0, x1, y1 = box

            # Ignore invalid coordinates.
            if not np.isfinite(
                [x0, y0, x1, y1]
            ).all():
                continue

            # Convert to integers.
            x0 = int(round(x0))
            y0 = int(round(y0))
            x1 = int(round(x1))
            y1 = int(round(y1))

            # Clip the MTCNN box to the image.
            x0 = max(0, min(w - 1, x0))
            y0 = max(0, min(h - 1, y0))
            x1 = max(1, min(w, x1))
            y1 = max(1, min(h, y1))

            # Check that the clipped box is valid.
            if x1 <= x0 or y1 <= y0:
                continue

            box_w = x1 - x0
            box_h = y1 - y0

            # Ignore extremely tiny detections.
            if box_w < 20 or box_h < 20:
                continue

            result.append(
                (
                    x0,
                    y0,
                    x1,
                    y1,
                )
            )

        return result

    # ----------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------

    def detect_faces(
        self,
        frame_bgr: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:

        if frame_bgr is None or frame_bgr.size == 0:
            return []

        if self.backend == "haar":
            return self._haar_boxes(
                frame_bgr
            )

        if self.backend == "mtcnn":
            return self._mtcnn_boxes(
                frame_bgr
            )

        # ------------------------------------------------------------
        # AUTO
        # ------------------------------------------------------------

        boxes = self._haar_boxes(
            frame_bgr
        )

        if boxes:
            return boxes

        return self._mtcnn_boxes(
            frame_bgr
        )

    def largest_face_box(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:

        boxes = self.detect_faces(
            frame_bgr
        )

        if not boxes:
            return None

        return max(
            boxes,
            key=lambda box: (
                max(
                    0,
                    box[2] - box[0],
                )
                *
                max(
                    0,
                    box[3] - box[1],
                )
            ),
        )


# ============================================================================
# DEEPFAKE DETECTOR
# ============================================================================

class DeepfakeDetector:
    """
    Main deepfake inference class.

    Uses the trained Xception checkpoint.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        checkpoint_path: Path = CHECKPOINT_PATH,
        device: str = DEVICE,
        threshold: float = DEFAULT_THRESHOLD,
        img_size: int = IMG_SIZE,
    ):

        self.model_name = model_name
        self.checkpoint_path = Path(
            checkpoint_path
        )
        self.device = device
        self.threshold = float(
            threshold
        )
        self.img_size = int(
            img_size
        )

        if not self.checkpoint_path.exists():

            raise ModelNotFoundError(
                f"Model checkpoint not found:\n"
                f"{self.checkpoint_path}"
            )

        print("=" * 60)
        print("LOADING DEEPFAKE DETECTOR")
        print("=" * 60)

        print(
            f"Model      : {self.model_name}"
        )

        print(
            f"Checkpoint : {self.checkpoint_path}"
        )

        print(
            f"Device     : {self.device}"
        )

        print(
            f"Threshold  : {self.threshold}"
        )

        # ------------------------------------------------------------
        # Build model
        # ------------------------------------------------------------

        self.model = build_model(
            self.model_name,
            pretrained=False,
            dropout=0.5,
            device=self.device,
        )

        # ------------------------------------------------------------
        # Load checkpoint
        # ------------------------------------------------------------

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            state_dict = (
                checkpoint["model_state_dict"]
            )

        else:

            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.checkpoint_info = checkpoint

        print(
            "Checkpoint loaded successfully."
        )

        print("=" * 60)

        # ------------------------------------------------------------
        # Preprocessing
        # ------------------------------------------------------------

        self.transform = A.Compose(
            [
                A.Resize(
                    self.img_size,
                    self.img_size,
                ),

                A.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),

                ToTensorV2(),
            ]
        )

    # ========================================================================
    # IMAGE PREPROCESSING
    # ========================================================================

    def _preprocess_face(
        self,
        image_bgr: np.ndarray,
    ) -> torch.Tensor:

        if image_bgr is None:
            raise ValueError(
                "Input image is None."
            )

        if image_bgr.size == 0:
            raise ValueError(
                "Input image is empty."
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        transformed = self.transform(
            image=image_rgb
        )

        tensor = transformed[
            "image"
        ]

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        return tensor.to(
            self.device,
            non_blocking=True,
        )

    # ========================================================================
    # RAW MODEL PREDICTION
    # ========================================================================

    @torch.no_grad()
    def _predict_face(
        self,
        face_bgr: np.ndarray,
    ) -> float:

        tensor = self._preprocess_face(
            face_bgr
        )

        logits = self.model(
            tensor
        )

        probability = torch.sigmoid(
            logits
        )

        return float(
            probability.detach()
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

    # ========================================================================
    # FACE CROP
    # ========================================================================

    def _crop_with_margin(
        self,
        image_bgr: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> np.ndarray:

        h, w = image_bgr.shape[:2]

        x0, y0, x1, y1 = box

        face_w = max(
            1,
            x1 - x0,
        )

        face_h = max(
            1,
            y1 - y0,
        )

        margin_x = int(
            face_w * FACE_MARGIN
        )

        margin_y = int(
            face_h * FACE_MARGIN
        )

        x0 = max(
            0,
            x0 - margin_x,
        )

        y0 = max(
            0,
            y0 - margin_y,
        )

        x1 = min(
            w,
            x1 + margin_x,
        )

        y1 = min(
            h,
            y1 + margin_y,
        )

        crop = image_bgr[
            y0:y1,
            x0:x1,
        ]

        if crop.size == 0:
            return image_bgr

        return crop

    # ========================================================================
    # IMAGE PREDICTION
    # ========================================================================

    def predict_image(
        self,
        image_path: str,
        face_detector: Optional[
            FaceDetector
        ] = None,
    ) -> dict:

        image_path = str(
            image_path
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise FileNotFoundError(
                f"Could not read image:\n"
                f"{image_path}"
            )

        detector = (
            face_detector
            or FaceDetector(
                backend=FACE_DETECTOR,
                device=self.device,
            )
        )

        box = detector.largest_face_box(
            image
        )

        if box is not None:

            face = (
                self._crop_with_margin(
                    image,
                    box,
                )
            )

            probability = (
                self._predict_face(
                    face
                )
            )

            face_detected = True

        else:

            # No valid face was detected.
            #
            # Keep the existing fallback behavior.
            probability = (
                self._predict_face(
                    image
                )
            )

            face_detected = False

        label = label_from_prob(
            probability,
            self.threshold,
        )

        confidence = (
            confidence_from_prob(
                probability,
                self.threshold,
            )
        )

        return {
            "path": image_path,
            "label": label,
            "prediction": label,
            "probability_fake": probability,
            "confidence": confidence,
            "face_detected": face_detected,
            "face_box": box,
        }

    # ========================================================================
    # VIDEO PREDICTION
    # ========================================================================

    def predict_video(
        self,
        video_path: str,
        max_frames: int = VIDEO_MAX_FRAMES,
        face_detector: Optional[
            FaceDetector
        ] = None,
    ) -> dict:

        video_path = str(
            video_path
        )

        if not Path(
            video_path
        ).exists():

            raise FileNotFoundError(
                f"Video not found:\n"
                f"{video_path}"
            )

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video:\n"
                f"{video_path}"
            )

        detector = (
            face_detector
            or FaceDetector(
                backend=FACE_DETECTOR,
                device=self.device,
            )
        )

        probabilities = []

        frame_count = 0
        sampled_frames = 0
        face_frames = 0

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:
            total_frames = max_frames

        sample_interval = max(
            1,
            total_frames // max_frames,
        )

        while (
            sampled_frames < max_frames
        ):

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if (
                frame_count % sample_interval
                != 0
            ):
                continue

            box = detector.largest_face_box(
                frame
            )

            if box is not None:

                face = (
                    self._crop_with_margin(
                        frame,
                        box,
                    )
                )

                face_frames += 1

            else:

                face = frame

            try:

                probability = (
                    self._predict_face(
                        face
                    )
                )

                probabilities.append(
                    probability
                )

            except Exception:
                continue

            sampled_frames += 1

        cap.release()

        if not probabilities:

            raise RuntimeError(
                "No valid frames could be "
                "processed from video:\n"
                f"{video_path}"
            )

        # ------------------------------------------------------------
        # Aggregation
        # ------------------------------------------------------------

        probabilities_array = (
            np.asarray(
                probabilities,
                dtype=np.float32,
            )
        )

        mean_probability = float(
            np.mean(
                probabilities_array
            )
        )

        median_probability = float(
            np.median(
                probabilities_array
            )
        )

        max_probability = float(
            np.max(
                probabilities_array
            )
        )

        fake_frame_ratio = float(
            np.mean(
                probabilities_array
                >= self.threshold
            )
        )

        label = label_from_prob(
            mean_probability,
            self.threshold,
        )

        confidence = (
            confidence_from_prob(
                mean_probability,
                self.threshold,
            )
        )

        return {
            "path": video_path,
            "label": label,
            "prediction": label,
            "probability_fake": mean_probability,
            "mean_probability_fake": mean_probability,
            "median_probability_fake": median_probability,
            "max_probability_fake": max_probability,
            "fake_frame_ratio": fake_frame_ratio,
            "confidence": confidence,
            "frames_read": frame_count,
            "frames_sampled": sampled_frames,
            "frames_with_face": face_frames,
            "frame_probabilities": probabilities,
        }


# ============================================================================
# GLOBAL DETECTOR
# ============================================================================

_DETECTOR: Optional[
    DeepfakeDetector
] = None


def get_detector(
    force_reload: bool = False,
) -> DeepfakeDetector:

    global _DETECTOR

    if (
        _DETECTOR is None
        or force_reload
    ):

        _DETECTOR = (
            DeepfakeDetector()
        )

    return _DETECTOR


# ============================================================================
# SIMPLE CONVENIENCE FUNCTIONS
# ============================================================================

def predict_image(
    image_path: str,
) -> dict:

    detector = get_detector()

    return detector.predict_image(
        image_path
    )


def predict_video(
    video_path: str,
    max_frames: int = VIDEO_MAX_FRAMES,
) -> dict:

    detector = get_detector()

    return detector.predict_video(
        video_path,
        max_frames=max_frames,
    )


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DEEPFAKE INFERENCE MODULE")
    print("=" * 60)

    print(
        f"Device     : {DEVICE}"
    )

    print(
        f"Model      : {MODEL_NAME}"
    )

    print(
        f"Checkpoint : {CHECKPOINT_PATH}"
    )

    print(
        f"Threshold  : {DEFAULT_THRESHOLD}"
    )

    detector = get_detector()

    print()
    print(
        "DeepfakeDetector initialized successfully."
    )