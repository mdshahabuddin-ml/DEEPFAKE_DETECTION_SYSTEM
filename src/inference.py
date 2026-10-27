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
from src.exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    InferenceError,
    FaceDetectionError,
    VideoError,
    ImageError,
)
from src.logger import get_logger
from src.calibration import PredictionCalibrator, CalibrationResult, UncertaintyResult

logger = get_logger(__name__)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:
    raise ImportError(
        "Albumentations is required for inference. "
        "Install it with: pip install albumentations"
    ) from exc


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
                    import logging
                    logging.getLogger(__name__).warning(
                        "facenet-pytorch is not installed. "
                        "Falling back to Haar cascade for face detection. "
                        "Install with: pip install facenet-pytorch"
                    )
                    # Fall back to Haar instead of crashing
                    if self.haar is None:
                        haar_path = cv2.data.haarcascades + (
                            "haarcascade_frontalface_default.xml"
                        )
                        if Path(haar_path).exists():
                            self.haar = cv2.CascadeClassifier(haar_path)
                            if self.haar.empty():
                                self.haar = None

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

    def detect(
        self,
        frame_bgr: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """Alias for detect_faces (backward compat)."""
        return self.detect_faces(frame_bgr)

    def crop_largest(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return the largest face crop (BGR) or None."""
        from src.config import FACE_MARGIN
        box = self.largest_face_box(frame_bgr)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        h, w = frame_bgr.shape[:2]
        mx = int((x1 - x0) * FACE_MARGIN)
        my = int((y1 - y0) * FACE_MARGIN)
        x0 = max(0, x0 - mx)
        y0 = max(0, y0 - my)
        x1 = min(w, x1 + mx)
        y1 = min(h, y1 + my)
        crop = frame_bgr[y0:y1, x0:x1]
        return crop if crop.size > 0 else None

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
            # Missing weights: try the configured auto-download source first
            # (DEEPFAKE_CHECKPOINT_URL / DEEPFAKE_HF_REPO, incl. Streamlit
            # secrets) before giving up.
            from src.checkpoint import CheckpointDownloadError, ensure_checkpoint
            try:
                ensure_checkpoint(self.checkpoint_path)
            except CheckpointDownloadError as exc:
                raise ModelNotFoundError(
                    f"Model checkpoint download failed:\n{exc}\n\n"
                    f"Expected file: {self.checkpoint_path}\n"
                    "Fix DEEPFAKE_CHECKPOINT_URL / DEEPFAKE_HF_REPO "
                    "(env or Streamlit secrets), or place the .pth manually."
                ) from exc
            if not self.checkpoint_path.exists():
                raise ModelNotFoundError(
                    f"Model checkpoint not found:\n"
                    f"{self.checkpoint_path}\n\n"
                    "Place a trained .pth at that path, or configure an "
                    "auto-download source (DEEPFAKE_CHECKPOINT_URL or "
                    "DEEPFAKE_HF_REPO). On Streamlit Community Cloud, add it "
                    "under Settings > Secrets."
                )

        logger.info("=" * 60)
        logger.info("LOADING DEEPFAKE DETECTOR")
        logger.info("=" * 60)
        logger.info("Model      : %s", self.model_name)
        logger.info("Checkpoint : %s", self.checkpoint_path)
        logger.info("Device     : %s", self.device)
        logger.info("Threshold  : %s", self.threshold)

        # ------------------------------------------------------------
        # Build model
        # ------------------------------------------------------------

        try:
            self.model = build_model(
                self.model_name,
                pretrained=False,
                dropout=0.5,
                device=self.device,
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to build model: {e}") from e

        # ------------------------------------------------------------
        # Load checkpoint
        # ------------------------------------------------------------

        try:
            from src.checkpoint import safe_torch_load
            checkpoint = safe_torch_load(
                self.checkpoint_path,
                map_location=self.device,
            )

            if (
                isinstance(checkpoint, dict)
                and "model_state_dict" in checkpoint
            ):
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(
                state_dict,
                strict=True,
            )

            self.model.to(self.device)
            self.model.eval()

            self.checkpoint_info = checkpoint
            logger.info("Checkpoint loaded successfully.")
            logger.info("=" * 60)

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load checkpoint: {e}"
            ) from e

        # ------------------------------------------------------------
        # Reusable face detector (avoids re-init per call)
        # ------------------------------------------------------------

        try:
            self.face_detector = FaceDetector(
                backend=FACE_DETECTOR,
                device=self.device,
            )
        except Exception as e:
            logger.warning("Face detector init failed, using fallback: %s", e)
            self.face_detector = FaceDetector(
                backend="auto",
                device=self.device,
            )

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

        # ------------------------------------------------------------
        # Prediction calibrator
        # ------------------------------------------------------------

        self.calibrator = PredictionCalibrator(
            method="temperature",
            n_mc_passes=30,
        )
        logger.info("Prediction calibrator initialized")

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
        face_detector: Optional[FaceDetector] = None,
    ) -> dict:
        """
        Run inference on an image file.

        Args:
            image_path: Path to the image file
            face_detector: Optional face detector instance

        Returns:
            Dictionary with prediction results
        """
        import time as _time
        _t0 = _time.perf_counter()
        image_path = str(image_path)

        try:
            image = cv2.imread(image_path)
        except Exception as e:
            raise ImageError(f"Could not read image: {image_path}") from e

        if image is None:
            raise ImageError(f"Could not read image:\n{image_path}")

        detector = face_detector or self.face_detector

        box = detector.largest_face_box(image)

        if box is not None:
            face = self._crop_with_margin(image, box)
            probability = self._predict_face(face)
            face_detected = True
            logger.debug("Face detected at box: %s", box)
        else:
            # No valid face was detected.
            # Keep the existing fallback behavior.
            probability = self._predict_face(image)
            face_detected = False
            logger.warning("No face detected in image: %s", image_path)

        label = label_from_prob(probability, self.threshold)
        confidence = confidence_from_prob(probability, self.threshold)
        _elapsed = round(_time.perf_counter() - _t0, 4)

        return {
            "path": image_path,
            "file_name": Path(image_path).name,
            "label": label,
            "prediction": label,
            "probability_fake": probability,
            "fake_probability": probability,
            "confidence": confidence,
            "face_detected": face_detected,
            "face_box": box,
            "inference_time": _elapsed,
        }

    # ========================================================================
    # PREDICT FROM ARRAY (no disk read)
    # ========================================================================

    def predict_image_array(
        self,
        bgr_image: np.ndarray,
        face_detector: Optional[FaceDetector] = None,
    ) -> dict:
        """
        Run inference on a BGR numpy array directly (no file I/O).

        Args:
            bgr_image: Input image as BGR numpy array
            face_detector: Optional face detector instance

        Returns:
            Dictionary with prediction results
        """
        import time

        if bgr_image is None or bgr_image.size == 0:
            logger.warning("Empty or None image provided for prediction")
            return {
                "path": "",
                "label": "NO_FACE",
                "prediction": "NO_FACE",
                "probability_fake": 0.0,
                "confidence": 0.0,
                "face_detected": False,
                "face_box": None,
                "inference_time": 0.0,
            }

        t0 = time.perf_counter()

        detector = face_detector or self.face_detector
        box = detector.largest_face_box(bgr_image)

        if box is not None:
            face = self._crop_with_margin(bgr_image, box)
            probability = self._predict_face(face)
            face_detected = True
        else:
            probability = self._predict_face(bgr_image)
            face_detected = False
            logger.debug("No face detected in array prediction")

        label = label_from_prob(probability, self.threshold)
        confidence = confidence_from_prob(probability, self.threshold)

        inference_time = round(time.perf_counter() - t0, 4)
        logger.debug("Array prediction complete: %s (%.4f seconds)", label, inference_time)

        return {
            "path": "",
            "label": label,
            "prediction": label,
            "probability_fake": probability,
            "fake_probability": probability,
            "confidence": confidence,
            "face_detected": face_detected,
            "face_box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])] if box else None,
            "inference_time": inference_time,
        }

    # ========================================================================
    # BATCH IMAGE PREDICTION
    # ========================================================================

    def predict_images(
        self,
        image_paths: List[str],
        max_images: int = 100,
    ) -> List[dict]:
        """Run prediction on a list of image paths."""
        results = []
        for path in image_paths[:max_images]:
            try:
                result = self.predict_image(path)
                result["file_name"] = Path(path).name
                results.append(result)
            except Exception as e:
                results.append({
                    "path": str(path),
                    "file_name": Path(path).name,
                    "label": "ERROR",
                    "prediction": "ERROR",
                    "probability_fake": 0.0,
                    "confidence": 0.0,
                    "face_detected": False,
                    "error": str(e),
                })
        return results

    # ========================================================================
    # BATCH VIDEO PREDICTION
    # ========================================================================

    def predict_videos(
        self,
        video_paths: List[str],
        max_frames: int = VIDEO_MAX_FRAMES,
        max_videos: int = 50,
    ) -> List[dict]:
        """Run prediction on a list of video paths."""
        results = []
        for path in video_paths[:max_videos]:
            try:
                result = self.predict_video(path, max_frames=max_frames)
                result["file_name"] = Path(path).name
                results.append(result)
            except Exception as e:
                results.append({
                    "path": str(path),
                    "file_name": Path(path).name,
                    "label": "ERROR",
                    "prediction": "ERROR",
                    "probability_fake": 0.0,
                    "confidence": 0.0,
                    "error": str(e),
                })
        return results

    # ========================================================================
    # VIDEO ANNOTATION
    # ========================================================================

    def annotate_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: int = VIDEO_MAX_FRAMES,
    ) -> Optional[str]:
        """Annotate video frames with face boxes and predictions. Returns output path."""
        import tempfile

        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        if output_path is None:
            suffix = Path(video_path).stem
            output_path = str(Path(tempfile.gettempdir()) / f"annotated_{suffix}.mp4")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            logger.error("Could not open VideoWriter for %s", output_path)
            return None

        try:
            detector = self.face_detector
            frame_count = 0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, total // max_frames) if total > 0 else 1

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % interval != 0:
                    writer.write(frame)
                    continue

                boxes = detector.detect_faces(frame)
                annotated = frame.copy()
                for box in boxes:
                    face = self._crop_with_margin(frame, box)
                    prob = self._predict_face(face)
                    label = label_from_prob(prob, self.threshold)
                    color = (0, 0, 255) if label == "FAKE" else (0, 200, 0)
                    x0, y0, x1, y1 = box
                    cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
                    text = f"{label} {prob*100:.0f}%"
                    cv2.putText(annotated, text, (x0, y0 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                writer.write(annotated)
        finally:
            cap.release()
            writer.release()
        return output_path

    # ========================================================================
    # VIDEO PREDICTION
    # ========================================================================

    def predict_video(
        self,
        video_path: str,
        max_frames: int = VIDEO_MAX_FRAMES,
        face_detector: Optional[FaceDetector] = None,
    ) -> dict:
        """
        Run inference on a video file.

        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to sample
            face_detector: Optional face detector instance

        Returns:
            Dictionary with prediction results including frame-level stats
        """
        import time as _time
        _t0 = _time.perf_counter()
        video_path = str(video_path)

        if not Path(video_path).exists():
            raise VideoError(f"Video not found:\n{video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise VideoError(f"Could not open video:\n{video_path}")

        detector = face_detector or self.face_detector

        probabilities = []
        frame_count = 0
        sampled_frames = 0
        face_frames = 0

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                total_frames = max_frames

            sample_interval = max(1, total_frames // max_frames)

            logger.info("Processing video: %s (total frames: %d, sampling every %d frames)",
                       video_path, total_frames, sample_interval)

            while sampled_frames < max_frames:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                if frame_count % sample_interval != 0:
                    continue

                box = detector.largest_face_box(frame)

                if box is not None:
                    face = self._crop_with_margin(frame, box)
                    face_frames += 1
                else:
                    face = frame

                try:
                    probability = self._predict_face(face)
                    probabilities.append(probability)
                except Exception as e:
                    logger.warning("Failed to predict frame %d: %s", frame_count, e)
                    continue

                sampled_frames += 1

        finally:
            cap.release()

        if not probabilities:
            raise InferenceError(
                "No valid frames could be processed from video:\n"
                f"{video_path}"
            )

        # ------------------------------------------------------------
        # Aggregation
        # ------------------------------------------------------------

        probabilities_array = np.asarray(probabilities, dtype=np.float32)

        mean_probability = float(np.mean(probabilities_array))
        median_probability = float(np.median(probabilities_array))
        max_probability = float(np.max(probabilities_array))
        fake_frame_ratio = float(np.mean(probabilities_array >= self.threshold))

        label = label_from_prob(mean_probability, self.threshold)
        confidence = confidence_from_prob(mean_probability, self.threshold)

        logger.info("Video prediction complete: %s (confidence: %.2f%%, fake frames: %d/%d)",
                    label, confidence * 100, int(np.sum(probabilities_array >= self.threshold)),
                    len(probabilities))

        _elapsed = round(_time.perf_counter() - _t0, 4)
        return {
            "path": video_path,
            "file_name": Path(video_path).name,
            "label": label,
            "prediction": label,
            "probability_fake": mean_probability,
            "fake_probability": mean_probability,
            "mean_probability_fake": mean_probability,
            "median_probability_fake": median_probability,
            "max_probability_fake": max_probability,
            "fake_frame_ratio": fake_frame_ratio,
            "fake_frames": int(np.sum(probabilities_array >= self.threshold)),
            "real_frames": int(np.sum(probabilities_array < self.threshold)),
            "no_face_frames": sampled_frames - face_frames,
            "confidence": confidence,
            "frames_read": frame_count,
            "frames_sampled": sampled_frames,
            "frames_with_face": face_frames,
            "frame_probabilities": probabilities,
            "inference_time": _elapsed,
        }


    # ========================================================================
    # UNCERTAINTY-AWARE PREDICTION
    # ========================================================================

    def predict_with_uncertainty(
        self,
        image_bgr: np.ndarray,
        n_forward_passes: int = 30,
    ) -> dict:
        """
        Run prediction with MC Dropout uncertainty estimation.
        
        Args:
            image_bgr: Input image as BGR numpy array
            n_forward_passes: Number of MC Dropout forward passes
            
        Returns:
            Dictionary with prediction results and uncertainty metrics
        """
        import time
        
        if image_bgr is None or image_bgr.size == 0:
            return {
                "prediction": "NO_FACE",
                "fake_probability": 0.0,
                "confidence": 0.0,
                "uncertainty": 0.0,
                "confidence_interval": (0.0, 0.0),
                "face_detected": False,
                "inference_time": 0.0,
            }
        
        t0 = time.perf_counter()
        
        # Get face crop
        box = self.face_detector.largest_face_box(image_bgr)
        if box is not None:
            face = self._crop_with_margin(image_bgr, box)
            face_detected = True
        else:
            face = image_bgr
            face_detected = False
        
        # Preprocess
        tensor = self._preprocess_face(face)
        
        # Get uncertainty estimate
        uncertainty_result = self.calibrator.mc_dropout.estimate_uncertainty(
            self.model, tensor, self.device
        )
        
        # Get calibrated prediction
        calibration_result = self.calibrator.calibrate_single(
            self.model, tensor, self.device
        )
        
        label = label_from_prob(
            calibration_result.calibrated_prob,
            self.threshold,
        )
        confidence = confidence_from_prob(
            calibration_result.calibrated_prob,
            self.threshold,
        )
        
        inference_time = round(time.perf_counter() - t0, 4)
        
        logger.info(
            "Uncertainty prediction: %s (prob=%.4f, unc=%.4f, time=%.4f)",
            label, calibration_result.calibrated_prob,
            uncertainty_result.uncertainty, inference_time
        )
        
        return {
            "prediction": label,
            "label": label,
            "fake_probability": calibration_result.calibrated_prob,
            "probability_fake": calibration_result.calibrated_prob,
            "original_probability": calibration_result.original_prob,
            "confidence": confidence,
            "uncertainty": uncertainty_result.uncertainty,
            "std_prediction": uncertainty_result.std_prediction,
            "confidence_interval": uncertainty_result.confidence_interval,
            "predictive_entropy": uncertainty_result.predictive_entropy,
            "mutual_information": uncertainty_result.mutual_information,
            "face_detected": face_detected,
            "face_box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])] if box else None,
            "n_forward_passes": n_forward_passes,
            "inference_time": inference_time,
        }

    # ========================================================================
    # FACE LOCALIZATION (MULTI-FACE)
    # ========================================================================

    def localize_faces(
        self,
        image_bgr: np.ndarray,
        face_detector: Optional[FaceDetector] = None,
    ) -> List[dict]:
        """Detect and score every face in an image."""
        detector = face_detector or FaceDetector(
            backend=FACE_DETECTOR, device=self.device,
        )
        boxes = detector.detect_faces(image_bgr)
        results = []
        for box in boxes:
            face = self._crop_with_margin(image_bgr, box)
            prob = self._predict_face(face)
            label = label_from_prob(prob, self.threshold)
            confidence = confidence_from_prob(prob, self.threshold)
            results.append({
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "prediction": label,
                "confidence": confidence,
                "fake_probability": prob,
            })
        return results


# ============================================================================
# ANNOTATION HELPER
# ============================================================================

def annotate_faces(
    image_bgr: np.ndarray,
    results: List[dict],
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Draw per-face bounding boxes and labels on a copy of the image."""
    annotated = image_bgr.copy()
    for r in results:
        x0, y0, x1, y1 = r["box"]
        color = (0, 0, 255) if r["prediction"] == "FAKE" else (0, 200, 0)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
        label = f"{r['prediction']} {r['confidence'] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x0, y0 - th - 8), (x0 + tw + 4, y0), color, -1)
        cv2.putText(annotated, label, (x0 + 2, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


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