"""Tests for the inference engine (face detection + detector)."""

import numpy as np

from src.inference import (
    FaceDetector, ModelNotFoundError, DeepfakeDetector, annotate_faces,
)
from tests.conftest import requires_model


def test_annotate_faces_returns_rgb():
    img = (np.random.rand(200, 200, 3) * 255).astype("uint8")
    results = [
        {"box": [10, 10, 80, 80], "prediction": "FAKE", "confidence": 0.9},
        {"box": [100, 100, 180, 180], "prediction": "REAL", "confidence": 0.8},
    ]
    out = annotate_faces(img, results, threshold=0.5)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Drawing must not mutate the input image.
    assert not np.array_equal(out, img)


def test_annotate_faces_empty_list():
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    out = annotate_faces(img, [], threshold=0.5)
    assert out.shape == img.shape


def test_face_detector_no_face(dummy_bgr_image):
    """Random noise should yield no detected faces (not crash)."""
    fd = FaceDetector(backend="auto")
    boxes = fd.detect(dummy_bgr_image)
    assert isinstance(boxes, list)
    assert fd.largest_face_box(dummy_bgr_image) is None


def test_face_detector_handles_empty():
    fd = FaceDetector(backend="auto")
    assert fd.detect(np.empty((0, 0, 3), dtype="uint8")) == []


def test_missing_model_raises(tmp_path):
    """A bogus checkpoint path must raise ModelNotFoundError."""
    try:
        DeepfakeDetector(checkpoint_path=tmp_path / "does_not_exist.pth")
    except ModelNotFoundError:
        return
    raise AssertionError("Expected ModelNotFoundError for a missing checkpoint")


@requires_model
def test_predict_image_array_shape(dummy_bgr_image):
    """With a real checkpoint, prediction returns the expected keys."""
    from src.inference import get_detector
    result = get_detector().predict_image_array(dummy_bgr_image)
    assert "prediction" in result
    assert "inference_time" in result
    assert result["prediction"] in {"REAL", "FAKE", "NO_FACE"}
