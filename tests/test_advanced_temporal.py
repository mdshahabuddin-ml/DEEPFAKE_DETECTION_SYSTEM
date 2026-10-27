"""
tests/test_advanced_temporal.py
Standalone tests for advanced_forensics.temporal.

Model-backed tests reuse the EXISTING local checkpoint (skipped when
absent); aggregation math is tested with synthetic tracks (no model).
"""
import numpy as np
import pytest
import torch

from advanced_forensics import temporal as TP
from tests.conftest import requires_model


def test_aggregate_track_math():
    flat = TP.aggregate_track([0.1] * 10, [True] * 10)
    assert flat["available"] and flat["flip_rate"] == 0.0
    flicker = TP.aggregate_track([0.1, 0.9] * 5, [True] * 10)
    assert flicker["flip_rate"] == 1.0
    assert flicker["std"] > flat["std"]
    assert TP.aggregate_track([], [])["available"] is False


def test_lstm_stub_refuses_untrained_inference():
    net = TP.TemporalLSTMAggregator(input_dim=8, hidden_dim=4)
    assert net.trained is False
    with pytest.raises(RuntimeError):
        net(torch.zeros(1, 5, 8))


def test_availability_reports_reason():
    info = TP.availability()
    assert "available" in info and "reason" in info


@requires_model
def test_temporal_report_on_synthetic_frames():
    rng = np.random.RandomState(11)
    frames = [(rng.rand(240, 240, 3) * 255).astype(np.uint8)
              for _ in range(6)]
    rep = TP.temporal_report(frames, max_frames=6)
    # Synthetic noise has no Haar face -> full-frame fallback still runs.
    assert rep["available"] is True
    assert len(rep["frame_probs"]) >= 3
    assert all(0.0 <= p <= 1.0 for p in rep["frame_probs"])
    assert 0.0 <= rep["score"] <= 1.0


@requires_model
def test_temporal_report_short_and_empty():
    assert TP.temporal_report([])["available"] is False
    one = [(np.random.rand(240, 240, 3) * 255).astype(np.uint8)]
    assert TP.temporal_report(one)["available"] is False
