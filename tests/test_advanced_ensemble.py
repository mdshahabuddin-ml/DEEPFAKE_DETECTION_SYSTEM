"""
tests/test_advanced_ensemble.py
Standalone tests for the transparent ensemble (no models needed).
"""
from advanced_forensics import ensemble as EN


def _sig(score, available=True):
    return {"available": available, "score": score}


def test_clear_fake_and_real():
    fake = EN.combine({k: _sig(v) for k, v in
                       [("xception", 0.9), ("frequency", 0.8),
                        ("temporal", 0.85), ("landmark", 0.7),
                        ("motion", 0.75)]})
    assert fake["verdict"] == "FAKE"
    real = EN.combine({k: _sig(v) for k, v in
                       [("xception", 0.1), ("frequency", 0.2),
                        ("temporal", 0.15), ("landmark", 0.3),
                        ("motion", 0.25)]})
    assert real["verdict"] == "REAL"


def test_borderline_is_uncertain_and_uninflated():
    mid = EN.combine({"xception": _sig(0.52), "frequency": _sig(0.48)})
    assert mid["verdict"] == "UNCERTAIN"
    assert mid["confidence"] <= mid["reliability"] <= 1.0


def test_missing_signals_renormalise_and_exclude():
    out = EN.combine({"xception": _sig(0.9),
                      "transformer": {"available": False, "score": None},
                      "rppg": {"available": True, "score": 0.1}})
    assert "transformer" in out["excluded"]
    assert "rppg" in out["excluded"]  # experimental: opt-in only
    assert abs(sum(v["weight"] for v in out["used"].values()) - 1.0) < 1e-6


def test_rppg_opt_in_stays_capped():
    out = EN.combine({"xception": _sig(0.9),
                      "rppg": {"available": True, "score": 0.5}},
                     include_rppg=True)
    assert "rppg" in out["used"]
    assert out["used"]["rppg"]["weight"] <= 0.06


def test_no_signals_gives_uncertain():
    out = EN.combine({})
    assert out["available"] is False
    assert out["verdict"] == "UNCERTAIN"
    assert out["confidence"] == 0.0


def test_confidence_never_exceeds_reliability():
    import random
    rng = random.Random(2)
    for _ in range(25):
        sigs = {k: _sig(rng.random()) for k in
                ["xception", "landmark", "motion", "frequency", "temporal"]}
        out = EN.combine(sigs)
        assert out["confidence"] <= out["reliability"] + 1e-9
