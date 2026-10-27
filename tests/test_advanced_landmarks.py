"""
tests/test_advanced_landmarks.py
Standalone tests for advanced_forensics.landmarks (local data only).
"""
import numpy as np

from advanced_forensics import landmarks as LM


def _typical_5pt():
    # Synthetic plausible 5-point face (eyes, nose, mouth corners).
    return np.array([[80., 90.], [120., 90.], [100., 110.],
                     [85., 135.], [115., 135.]], dtype=np.float32)


def test_availability_reports_local_backend():
    info = LM.availability()
    assert info["available"] is True
    assert info["points_per_face"] == 5


def test_geometry_metrics_scale_invariant():
    g1 = LM.geometry_metrics(_typical_5pt())
    g2 = LM.geometry_metrics(_typical_5pt() * 2.0)
    assert abs(g1["mouth_width"] - g2["mouth_width"]) < 1e-6
    assert g1["symmetry"] < 0.05


def test_geometry_rejects_bad_shapes():
    for bad in [np.zeros((4, 2)), np.zeros((5, 3)), None]:
        try:
            LM.geometry_metrics(bad)
        except (ValueError, TypeError, AttributeError):
            continue
        else:
            raise AssertionError(f"accepted bad shape {bad}")


def test_ear_needs_six_points():
    assert LM.ear_supported(np.zeros((6, 2))) is True
    assert LM.ear_supported(_typical_5pt()) is False
    eye = np.array([[0., 0.], [1., 1.], [2., 1.],
                    [4., 0.], [2., -1.], [1., -1.]])
    assert LM.eye_aspect_ratio(eye) > 0
    try:
        LM.eye_aspect_ratio(_typical_5pt())
    except ValueError:
        pass
    else:
        raise AssertionError("EAR accepted 5-point input")


def test_landmark_signal_range_and_invalid():
    s = LM.landmark_signal(_typical_5pt())
    assert s["available"] is True
    assert 0.0 <= s["score"] <= 1.0
    assert s["uncalibrated_heuristic"] is True
    bad = LM.landmark_signal(np.zeros((3, 2)))
    assert bad["available"] is False and bad["score"] is None


def test_detect_landmarks_handles_no_face_and_corrupt():
    ana = LM.LandmarkAnalyzer()
    blank = np.zeros((160, 160, 3), dtype=np.uint8)
    assert ana.detect_landmarks(blank) == []
    assert ana.detect_landmarks(None) == []
    assert ana.detect_landmarks(np.zeros((0, 0, 3), dtype=np.uint8)) == []
