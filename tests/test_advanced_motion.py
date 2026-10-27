"""
tests/test_advanced_motion.py
Standalone tests for advanced_forensics.motion (synthetic tracks only).
"""
import numpy as np

from advanced_forensics import motion as MO


def _track(jitter=0.2, n=12):
    rng = np.random.RandomState(7)
    base = np.array([[80., 90.], [120., 90.], [100., 110.],
                     [85., 135.], [115., 135.]])
    drift = np.linspace(0, 2.0, n)[:, None, None]
    noise = rng.normal(0, jitter, size=(n, 5, 2))
    return [base + drift[i] + noise[i] for i in range(n)]


def test_stable_vs_jittery_tracks():
    calm = MO.motion_report(_track(jitter=0.15))
    wild = MO.motion_report(_track(jitter=4.0))
    assert calm["available"] and wild["available"]
    assert wild["score"] > calm["score"]
    assert 0.0 <= calm["score"] <= 1.0


def test_short_track_unavailable():
    assert MO.jitter_index(_track(n=3))["available"] is False
    assert MO.displacement_stats([])["available"] is False
    r = MO.motion_report([])
    assert r["available"] is False and r["score"] is None


def test_blink_detection_on_synthetic_ear():
    fps, n = 30, 120
    ear = np.full(n, 0.30)
    for start in (20, 60, 100):  # three dips below threshold
        ear[start:start + 3] = 0.12
    out = MO.analyze_blinks(ear, fps=fps)
    assert out["available"] is True
    assert out["blinks"] == 3
    assert 5.0 <= out["blinks_per_min"] <= 60.0


def test_blink_needs_real_ear_sequence():
    assert MO.analyze_blinks([])["available"] is False
    assert MO.analyze_blinks([0.3])["available"] is False
    r = MO.motion_report(_track(), ear_seq=None)
    assert r["blinks"]["available"] is False


def test_openness_stats():
    ok = MO.openness_stats([0.5, 0.55, 0.45, 0.6], "mouth")
    assert ok["available"] and ok["range"] > 0
    assert MO.openness_stats([0.5], "eye")["available"] is False
