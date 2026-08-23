"""Tests for src/config.py helpers and path setup."""

from src.config import (
    LOGS_DIR, MODELS_DIR, OUTPUTS_DIR, RESULTS_DIR, UPLOADS_DIR,
    confidence_from_prob, label_from_prob,
)


def test_runtime_dirs_exist():
    for d in (MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, LOGS_DIR, RESULTS_DIR):
        assert d.exists() and d.is_dir()


def test_label_from_prob():
    assert label_from_prob(0.9, 0.5) == "FAKE"
    assert label_from_prob(0.1, 0.5) == "REAL"
    assert label_from_prob(0.5, 0.5) == "FAKE"  # boundary -> fake


def test_confidence_from_prob():
    # FAKE side: confidence == prob
    assert confidence_from_prob(0.8, 0.5) == 0.8
    # REAL side: confidence == 1 - prob
    assert abs(confidence_from_prob(0.2, 0.5) - 0.8) < 1e-9
