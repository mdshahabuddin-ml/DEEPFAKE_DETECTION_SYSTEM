"""
src/validation.py
Input validation utilities for the Deepfake Detection System.

Provides functions for validating uploaded files, media content, and
configuration parameters before processing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Set, Tuple

import cv2
import numpy as np

from src.config import (
    ALLOWED_IMAGE_EXT,
    ALLOWED_VIDEO_EXT,
    MAX_UPLOAD_MB,
    IMG_SIZE,
)
from src.exceptions import (
    FileValidationError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    ImageError,
    VideoError,
)
from src.logger import get_logger

logger = get_logger(__name__)


def validate_file_extension(
    filename: str,
    allowed_extensions: Set[str]
) -> str:
    """
    Validate that a file has an allowed extension.

    Args:
        filename: Name of the file to validate
        allowed_extensions: Set of allowed file extensions (e.g., {'.jpg', '.png'})

    Returns:
        The validated file extension (lowercase)

    Raises:
        UnsupportedFileTypeError: If file type is not allowed
    """
    if not filename:
        raise UnsupportedFileTypeError("No filename provided")

    ext = Path(filename).suffix.lower()

    if ext not in allowed_extensions:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Allowed: {sorted(allowed_extensions)}"
        )

    return ext


def validate_image_extension(filename: str) -> str:
    """
    Validate that an image file has an allowed extension.

    Args:
        filename: Name of the image file

    Returns:
        The validated file extension

    Raises:
        UnsupportedFileTypeError: If image type is not allowed
    """
    return validate_file_extension(filename, ALLOWED_IMAGE_EXT)


def validate_video_extension(filename: str) -> str:
    """
    Validate that a video file has an allowed extension.

    Args:
        filename: Name of the video file

    Returns:
        The validated file extension

    Raises:
        UnsupportedFileTypeError: If video type is not allowed
    """
    return validate_file_extension(filename, ALLOWED_VIDEO_EXT)


def validate_file_size(
    file_path: str,
    max_size_mb: int = MAX_UPLOAD_MB
) -> bool:
    """
    Validate that a file does not exceed the maximum size.

    Args:
        file_path: Path to the file to validate
        max_size_mb: Maximum allowed file size in megabytes

    Returns:
        True if file size is valid

    Raises:
        FileTooLargeError: If file exceeds maximum size
    """
    if not os.path.exists(file_path):
        raise FileValidationError(f"File not found: {file_path}")

    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_mb > max_size_mb:
        raise FileTooLargeError(
            f"File size ({file_size_mb:.2f} MB) exceeds limit ({max_size_mb} MB)"
        )

    logger.debug("File size validation passed: %.2f MB", file_size_mb)
    return True


def validate_image_readable(image_path: str) -> np.ndarray:
    """
    Validate that an image file can be read by OpenCV.

    Args:
        image_path: Path to the image file

    Returns:
        The loaded image as BGR numpy array

    Raises:
        ImageError: If image cannot be read
    """
    if not os.path.exists(image_path):
        raise ImageError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ImageError(f"Could not read image file: {image_path}")

    if image.size == 0:
        raise ImageError(f"Image file is empty: {image_path}")

    logger.debug("Image loaded successfully: %s (%dx%d)", 
                image_path, image.shape[1], image.shape[0])
    return image


def validate_video_readable(video_path: str) -> Tuple[cv2.VideoCapture, int, float]:
    """
    Validate that a video file can be read by OpenCV.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (VideoCapture object, total_frames, fps)

    Raises:
        VideoError: If video cannot be read
    """
    if not os.path.exists(video_path):
        raise VideoError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise VideoError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        raise VideoError(f"Video has no frames: {video_path}")

    if fps <= 0:
        logger.warning("Video FPS not detected, defaulting to 30.0")
        fps = 30.0

    logger.debug("Video loaded successfully: %s (%d frames, %.2f FPS)", 
                video_path, total_frames, fps)
    return cap, total_frames, fps


def validate_face_image(
    face_image: np.ndarray,
    min_size: int = 20
) -> bool:
    """
    Validate that a face crop is suitable for inference.

    Args:
        face_image: Face crop as numpy array
        min_size: Minimum dimension size for a valid face

    Returns:
        True if face image is valid

    Raises:
        ImageError: If face image is invalid
    """
    if face_image is None:
        raise ImageError("Face image is None")

    if face_image.size == 0:
        raise ImageError("Face image is empty")

    h, w = face_image.shape[:2]

    if h < min_size or w < min_size:
        raise ImageError(
            f"Face image too small: {w}x{h} (minimum: {min_size}x{min_size})"
        )

    logger.debug("Face image validation passed: %dx%d", w, h)
    return True


def validate_model_config(
    model_name: str,
    img_size: int = IMG_SIZE
) -> bool:
    """
    Validate model configuration parameters.

    Args:
        model_name: Name of the model architecture
        img_size: Expected input image size

    Returns:
        True if configuration is valid

    Raises:
        ValidationError: If configuration is invalid
    """
    valid_models = {"xception", "efficientnet", "vit"}

    if model_name not in valid_models:
        raise FileValidationError(
            f"Invalid model name: {model_name}. Must be one of: {valid_models}"
        )

    if img_size <= 0:
        raise FileValidationError(f"Invalid image size: {img_size}")

    logger.debug("Model config validation passed: %s (img_size=%d)", model_name, img_size)
    return True


def validate_tensor_input(
    tensor: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None
) -> bool:
    """
    Validate a tensor/array for model input.

    Args:
        tensor: Input tensor as numpy array
        expected_shape: Optional expected shape tuple

    Returns:
        True if tensor is valid

    Raises:
        ValidationError: If tensor is invalid
    """
    if tensor is None:
        raise FileValidationError("Input tensor is None")

    if not isinstance(tensor, np.ndarray):
        raise FileValidationError(f"Expected numpy array, got {type(tensor)}")

    if tensor.size == 0:
        raise FileValidationError("Input tensor is empty")

    if np.any(np.isnan(tensor)):
        raise FileValidationError("Input tensor contains NaN values")

    if np.any(np.isinf(tensor)):
        raise FileValidationError("Input tensor contains Inf values")

    if expected_shape is not None and tensor.shape != expected_shape:
        logger.warning(
            "Tensor shape mismatch: got %s, expected %s",
            tensor.shape, expected_shape
        )

    logger.debug("Tensor validation passed: shape=%s, dtype=%s", tensor.shape, tensor.dtype)
    return True
