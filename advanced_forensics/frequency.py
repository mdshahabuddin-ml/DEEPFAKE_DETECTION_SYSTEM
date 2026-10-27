"""
advanced_forensics.frequency
============================

Frequency-domain artifact analysis with NumPy / OpenCV / SciPy only
(all installed locally; nothing downloaded).

Features per face crop (or full frame):
- radially-averaged FFT log-magnitude profile: high-frequency energy
  ratio + log-log spectral slope (synthetic upsampling often flattens
  or spikes the high end);
- periodic-peak indicator: max/median ratio in the high-frequency band
  (GAN checkerboard / resampling lattices produce sharp peaks);
- 8x8-block boundary discontinuity (DCT-block / compression grid);
- high-pass noise-residual variance (over-smoothing vs noisy fakes).

Returns numerical features plus one UNCALIBRATED heuristic score in
[0, 1] for ensemble input.  It is not a calibrated probability and
must never be presented as one.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

HF_CUTOFF = 0.40  # fraction of Nyquist where "high frequency" starts


def availability() -> dict:
    try:
        import numpy  # noqa: F401
        import cv2  # noqa: F401
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}
    try:
        import scipy  # noqa: F401
        extra = "numpy+cv2+scipy"
    except ImportError:
        extra = "numpy+cv2 (scipy optional helpers off)"
    return {"available": True, "reason": f"local FFT/DCT via {extra}"}


def _to_gray(face_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    if face_bgr is None or not isinstance(face_bgr, np.ndarray) \
            or face_bgr.size == 0:
        raise ValueError("empty/corrupt frame")
    gray = (cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            if face_bgr.ndim == 3 else face_bgr)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float64)


def radial_profile(magnitude: np.ndarray, nbins: int = 32) -> np.ndarray:
    """Azimuthally averaged magnitude spectrum (nbins radial bins)."""
    h, w = magnitude.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = r.max()
    edges = np.linspace(0, rmax, nbins + 1)
    prof = np.array([magnitude[(r >= edges[i]) & (r < edges[i + 1])].mean()
                     for i in range(nbins)])
    return np.nan_to_num(prof, nan=0.0, posinf=0.0, neginf=0.0)


def spectral_features(gray: np.ndarray) -> dict:
    """FFT profile features from a square grayscale array."""
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    mag = np.abs(f) + 1e-12
    prof = radial_profile(np.log1p(mag))
    n = len(prof)
    k0 = int(n * HF_CUTOFF)
    total = prof.sum() + 1e-12
    hf_ratio = float(prof[k0:].sum() / total)
    # Log-log slope over the upper half (natural images ~= -2 power law).
    xs = np.log1p(np.arange(k0, n))
    ys = np.log1p(prof[k0:])
    slope = float(np.polyfit(xs, ys, 1)[0]) if n - k0 >= 3 else 0.0
    hf_band = prof[k0:]
    peak_ratio = (float(hf_band.max() / (np.median(hf_band) + 1e-12))
                  if hf_band.size else 0.0)
    return {"hf_ratio": hf_ratio, "spectral_slope": slope,
            "hf_peak_ratio": peak_ratio,
            "radial_profile": [float(v) for v in prof]}


def blockiness(gray: np.ndarray, block: int = 8) -> float:
    """Mean 8-px grid-boundary discontinuity vs interior (higher = blockier)."""
    g = gray.astype(np.float64)
    h, w = g.shape
    # Vertical boundaries at multiples of `block`.
    bv = np.abs(np.diff(g, axis=1))
    cols = np.arange(1, w) % block == 0
    edge_v = bv[:, cols].mean() if cols.any() else 0.0
    inne_v = bv[:, ~cols].mean() if (~cols).any() else 1.0
    bh = np.abs(np.diff(g, axis=0))
    rows = np.arange(1, h) % block == 0
    edge_h = bh[rows, :].mean() if rows.any() else 0.0
    inne_h = bh[~rows, :].mean() if (~rows).any() else 1.0
    return float(((edge_v / (inne_v + 1e-9)) + (edge_h / (inne_h + 1e-9))) / 2.0)


def noise_residual_var(gray: np.ndarray) -> float:
    """Variance of (image - Gaussian blur): over-smoothing vs noise."""
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    return float(np.var(gray - blur))


def _squash(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def frequency_report(face_bgr: np.ndarray) -> dict:
    """Numerical forensic features + heuristic score for one crop/frame."""
    try:
        gray = _to_gray(face_bgr)
    except (ValueError, cv2.error) as exc:
        return {"available": False, "score": None,
                "reason": f"unreadable frame: {exc}"}
    spec = spectral_features(gray)
    blk = blockiness(gray)
    nrv = noise_residual_var(gray)
    # Heuristic blend around typical natural-image operating points.
    # Reference points are fixed, documented, and NOT learned.
    parts = [
        min(1.0, max(0.0, (spec["hf_ratio"] - 0.10) / 0.25)),
        min(1.0, max(0.0, (spec["hf_peak_ratio"] - 3.0) / 9.0)),
        min(1.0, max(0.0, (blk - 1.0) / 1.5)),
        min(1.0, max(0.0, (40.0 - nrv) / 40.0)),
    ]
    score = _squash((float(np.mean(parts)) - 0.35) * 6.0)
    return {"available": True, "score": score,
            "features": {"hf_ratio": spec["hf_ratio"],
                         "spectral_slope": spec["spectral_slope"],
                         "hf_peak_ratio": spec["hf_peak_ratio"],
                         "blockiness": blk,
                         "noise_residual_var": nrv},
            "uncalibrated_heuristic": True,
            "notes": ["Heuristic spectral score; interpret alongside other "
                      "signals, never alone."]}
