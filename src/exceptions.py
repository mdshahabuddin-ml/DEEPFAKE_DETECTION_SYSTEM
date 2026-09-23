"""
src/exceptions.py
Custom exceptions for the Deepfake Detection System.

Provides a hierarchy of exceptions for better error handling and debugging.
"""

from __future__ import annotations


class DeepfakeDetectionError(Exception):
    """Base exception for all deepfake detection errors."""
    pass


class ModelError(DeepfakeDetectionError):
    """Errors related to model loading, inference, or checkpoint issues."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when the trained model checkpoint cannot be found."""
    pass


class ModelLoadError(ModelError):
    """Raised when the model fails to load from checkpoint."""
    pass


class InferenceError(ModelError):
    """Raised when model inference fails."""
    pass


class FaceDetectionError(DeepfakeDetectionError):
    """Errors related to face detection."""
    pass


class NoFaceDetectedError(FaceDetectionError):
    """Raised when no face is detected in the input."""
    pass


class FaceDetectionFailedError(FaceDetectionError):
    """Raised when face detection process fails."""
    pass


class MediaError(DeepfakeDetectionError):
    """Errors related to media file handling."""
    pass


class VideoError(MediaError):
    """Errors specific to video processing."""
    pass


class VideoNotFoundError(VideoError):
    """Raised when the video file cannot be found."""
    pass


class VideoReadError(VideoError):
    """Raised when the video cannot be read or decoded."""
    pass


class ImageError(MediaError):
    """Errors specific to image processing."""
    pass


class ImageReadError(ImageError):
    """Raised when the image cannot be read or decoded."""
    pass


class AudioError(DeepfakeDetectionError):
    """Errors related to audio processing."""
    pass


class AudioExtractionError(AudioError):
    """Raised when audio extraction from video fails."""
    pass


class AudioInferenceError(AudioError):
    """Raised when audio inference fails."""
    pass


class APIError(DeepfakeDetectionError):
    """Errors related to the REST API."""
    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


class ValidationError(DeepfakeDetectionError):
    """Raised when input validation fails."""
    pass


class FileValidationError(ValidationError):
    """Raised when uploaded file validation fails."""
    pass


class FileTooLargeError(FileValidationError):
    """Raised when uploaded file exceeds size limit."""
    pass


class UnsupportedFileTypeError(FileValidationError):
    """Raised when uploaded file type is not supported."""
    pass


class DatabaseError(DeepfakeDetectionError):
    """Errors related to database operations."""
    pass


class ExportError(DeepfakeDetectionError):
    """Errors related to ONNX export or model conversion."""
    pass


class ReportGenerationError(DeepfakeDetectionError):
    """Errors related to PDF report generation."""
    pass
