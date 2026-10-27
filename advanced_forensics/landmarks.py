"""
advanced_forensics.landmarks
============================

Facial landmark analysis using only locally installed libraries.

Source: ``facenet-pytorch`` MTCNN (already a project dependency, verified
to initialise and run offline) with ``landmarks=True`` yields 5-point
landmarks per face.  dlib / MediaPipe are NOT installed, so 68-point
landmarks are unavailable -- nothing is downloaded to obtain them.

The public API accepts generic Nx2 point arrays, so a denser landmark
source can be plugged in later without changing callers.

MTCNN 5-point order (facenet-pytorch convention): the two eye points,
the nose point, then the two mouth-corner points.  Indices are referred
to neutrally (p0..p4) with that mapping documented, because the exact
left/right anatomy cannot be verified without a reference face.

Landmark anomalies alone are NEVER proof of a deepfake; see
:func:`landmark_signal` notes and the ensemble module.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

# Neutral names for the 5 MTCNN points (eyes, nose, mouth corners).
POINT_NAMES_5 = ("p0_eye", "p1_eye", "p2_nose", "p3_mouth", "p4_mouth")

# Typical human face proportions (inter-eye-distance-normalised), used only
# to express *deviations* as interpretable features -- not as a verdict.
TYPICAL = {
    "mouth_width": 1.05,     # mouth-corner distance / inter-eye distance
    "nose_to_mouth": 0.55,   # nose->mouth-centre / inter-eye distance
    "eye_to_nose": 0.75,     # eye-centre->nose / inter-eye distance
}


def availability() -> dict:
    """Check the landmark backend without initialising any model."""
    try:
        import facenet_pytorch  # noqa: F401
    except ImportError:
        return {"available": False,
                "reason": "facenet-pytorch is not installed."}
    return {"available": True,
            "reason": "MTCNN 5-point landmarks via installed facenet-pytorch "
                      "(dlib/MediaPipe not installed; 68-point unavailable).",
            "points_per_face": 5}


def _as_points(raw) -> Optional[np.ndarray]:
    """Normalise MTCNN landmark output to (5, 2) float32, else None."""
    try:
        arr = np.asarray(raw, dtype=np.float32)
    except Exception:
        return None
    if arr.shape == (5, 2):
        return arr
    if arr.size == 10:  # flattened (x1..x5, y1..y5) or interleaved
        flat = arr.reshape(-1)
        # facenet-pytorch emits [x1..x5, y1..y5].
        return np.stack([flat[:5], flat[5:10]], axis=1).astype(np.float32)
    return None


class LandmarkAnalyzer:
    """MTCNN 5-point landmark detector (lazy, offline, read-only)."""

    def __init__(self, device: Optional[str] = None, min_face_size: int = 40):
        self.device = device
        self.min_face_size = int(min_face_size)
        self._mtcnn = None
        self._init_error: Optional[str] = None

    def _ensure(self) -> bool:
        if self._mtcnn is not None:
            return True
        if self._init_error is not None:
            return False
        try:
            from facenet_pytorch import MTCNN
            import torch

            device = self.device or ("cuda" if torch.cuda.is_available()
                                     else "cpu")
            self._mtcnn = MTCNN(keep_all=True, device=device,
                                min_face_size=self.min_face_size)
            self.device = device
            return True
        except Exception as exc:  # never crash the caller
            self._init_error = str(exc)
            logger.warning("Landmark backend unavailable: %s", exc)
            return False

    def detect_landmarks(self, bgr: np.ndarray) -> List[dict]:
        """Return [{box, points(5,2), prob}] for each face; [] if none.

        Never raises on bad input -- corrupt/empty frames yield [].
        """
        if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
            return []
        if not self._ensure():
            return []
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            boxes, probs, points = self._mtcnn.detect(rgb, landmarks=True)
        except Exception as exc:
            logger.warning("MTCNN landmark detection failed: %s", exc)
            return []
        if boxes is None:
            return []
        out = []
        for box, prob, pts in zip(boxes, probs or [], points or []):
            norm = _as_points(pts)
            if norm is None:
                continue
            try:
                p = float(prob[0]) if hasattr(prob, "__len__") else float(prob)
            except Exception:
                p = 0.0
            out.append({"box": [float(v) for v in np.asarray(box).ravel()[:4]],
                        "points": norm, "prob": p})
        return out


def geometry_metrics(points: np.ndarray) -> dict:
    """Interpretable face-geometry features from 5-point landmarks.

    All distances are normalised by the inter-eye distance so the features
    are scale invariant.  Returns plain floats (no verdict, no score).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (5, 2):
        raise ValueError(f"expected (5, 2) points, got {pts.shape}")
    eye_dist = float(np.linalg.norm(pts[0] - pts[1])) + 1e-8
    eye_mid = (pts[0] + pts[1]) / 2.0
    mouth_mid = (pts[3] + pts[4]) / 2.0
    mouth_width = float(np.linalg.norm(pts[3] - pts[4])) / eye_dist
    eye_to_nose = float(np.linalg.norm(eye_mid - pts[2])) / eye_dist
    nose_to_mouth = float(np.linalg.norm(pts[2] - mouth_mid)) / eye_dist
    # Mirror symmetry: horizontal distance of each eye from the nose x.
    symmetry = float(abs(abs(pts[0][0] - pts[2][0])
                         - abs(pts[1][0] - pts[2][0]))) / eye_dist
    return {"eye_dist_px": float(eye_dist),
            "mouth_width": mouth_width,
            "eye_to_nose": eye_to_nose,
            "nose_to_mouth": nose_to_mouth,
            "symmetry": symmetry}


def ear_supported(points: np.ndarray) -> bool:
    """True only for 6-point eye arrays (68-point style); 5-pt is not."""
    arr = np.asarray(points)
    return arr.shape == (6, 2)


def eye_aspect_ratio(eye6: np.ndarray) -> float:
    """Classic eye-aspect-ratio from 6 ordered eye landmarks.

    Raises ValueError for anything else (e.g. MTCNN 5-point output, where
    per-eye openness is technically unsupported).
    """
    eye = np.asarray(eye6, dtype=np.float64)
    if eye.shape != (6, 2):
        raise ValueError(
            f"EAR needs (6, 2) eye landmarks, got {eye.shape}. "
            "MTCNN 5-point output has one point per eye, so blink-rate/EAR "
            "analysis is unsupported with the current local backend.")
    vert = (np.linalg.norm(eye[1] - eye[5])
            + np.linalg.norm(eye[2] - eye[4]))
    horiz = np.linalg.norm(eye[0] - eye[3]) + 1e-8
    return float(vert / (2.0 * horiz))


def _squash(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def landmark_signal(points: np.ndarray) -> dict:
    """Heuristic geometry-deviation signal in [0, 1] (higher = odder).

    Compares normalised proportions against typical ranges.  This is an
    UNCALIBRATED heuristic for ensemble input only -- landmark quirks
    alone are never proof of manipulation.
    """
    try:
        g = geometry_metrics(points)
    except (ValueError, TypeError) as exc:
        return {"available": False, "score": None,
                "reason": f"invalid points: {exc}"}
    dev = (abs(g["mouth_width"] - TYPICAL["mouth_width"]) / 0.35
           + abs(g["nose_to_mouth"] - TYPICAL["nose_to_mouth"]) / 0.25
           + abs(g["eye_to_nose"] - TYPICAL["eye_to_nose"]) / 0.30
           + g["symmetry"] / 0.20)
    return {"available": True, "score": _squash(dev - 2.0),
            "features": g, "uncalibrated_heuristic": True,
            "notes": ["Heuristic geometry deviation; not proof of a "
                      "deepfake."]}
