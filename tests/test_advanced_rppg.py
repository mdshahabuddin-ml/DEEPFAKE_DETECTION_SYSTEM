"""
tests/test_advanced_rppg.py
Standalone tests for the experimental rPPG module (synthetic clips).
"""
import numpy as np

from advanced_forensics import rppg as RP


def _pulse_clip(n=120, fps=30, bpm=72, noise=1.5, seed=5):
    rng = np.random.RandomState(seed)
    t = np.arange(n) / fps
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)
    frames, boxes = [], []
    for i in range(n):
        frame = np.full((240, 240, 3), 128, dtype=np.uint8)
        # Cheeks/forehead region brightness follows the pulse (BGR green).
        glow = int(8 * pulse[i])
        frame[60:180, 60:180, 1] = np.clip(150 + glow, 0, 255)
        frame = (frame.astype(np.float32)
                 + rng.normal(0, noise, size=frame.shape)).clip(0, 255)
        frames.append(frame.astype(np.uint8))
        boxes.append([60., 60., 180., 180.])
    return frames, boxes


def test_rppg_detects_synthetic_pulse():
    frames, boxes = _pulse_clip()
    rep = RP.rppg_report(frames, boxes, fps=30.0)
    assert rep["available"] is True
    assert rep["experimental"] is True
    assert abs(rep["bpm"] - 72.0) < 12.0  # FFT bin tolerance
    assert 0.0 <= rep["score"] <= 0.5  # capped: absence proves nothing
    assert any("EXPERIMENTAL" in n for n in rep["notes"])


def test_rppg_graceful_degradation():
    frames, boxes = _pulse_clip(n=120)
    assert RP.rppg_report(frames[:10], boxes[:10])["available"] is False
    assert RP.rppg_report([], [])["available"] is False
    assert RP.rppg_report(frames, [None] * len(frames))["available"] is False
    assert RP.rppg_report(frames, boxes, fps=0)["available"] is False
    # Tiny face -> poor quality, still structured output.
    small = [[10., 10., 40., 40.]] * len(frames)
    rep = RP.rppg_report(frames, small)
    if rep["available"]:
        assert rep["quality"] in ("good", "poor")


def test_skin_rois_inside_frame():
    rois = RP.skin_rois([60., 60., 180., 180.])
    assert len(rois) == 3
    for rs, cs in rois:
        assert rs.stop > rs.start and cs.stop > cs.start
