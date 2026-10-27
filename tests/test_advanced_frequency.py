"""
tests/test_advanced_frequency.py
Standalone tests for advanced_forensics.frequency (synthetic images).
"""
import numpy as np

from advanced_forensics import frequency as FQ


def _natural_like(seed=3):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, 256)
    img = (np.sin(2 * np.pi * 3 * x)[None, :] * 40 + 120)
    img = np.repeat(img, 256, axis=0) + rng.normal(0, 6, size=(256, 256))
    return np.clip(img, 0, 255).astype(np.uint8)


def _periodic_fake():
    x = np.arange(256)
    grid = ((np.sin(2 * np.pi * x / 8)[:, None]
             + np.sin(2 * np.pi * x / 8)[None, :]) > 0.5)
    return (grid * 255).astype(np.uint8)


def test_periodic_pattern_scores_higher():
    r_nat = FQ.frequency_report(_natural_like())
    r_per = FQ.frequency_report(_periodic_fake())
    assert r_nat["available"] and r_per["available"]
    assert r_per["features"]["hf_peak_ratio"] > r_nat["features"]["hf_peak_ratio"]
    for r in (r_nat, r_per):
        assert 0.0 <= r["score"] <= 1.0
        assert r["uncalibrated_heuristic"] is True
        assert set(r["features"]) >= {"hf_ratio", "spectral_slope",
                                      "hf_peak_ratio", "blockiness",
                                      "noise_residual_var"}


def test_corrupt_and_tiny_inputs():
    assert FQ.frequency_report(None)["available"] is False
    assert FQ.frequency_report(np.zeros((0, 0, 3), np.uint8))["available"] is False
    tiny = FQ.frequency_report(np.full((8, 8, 3), 128, np.uint8))
    assert tiny["available"] is True  # resized internally, still valid


def test_blockiness_sane():
    flat = np.full((64, 64), 128.0)
    assert FQ.blockiness(flat) >= 0.0
