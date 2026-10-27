"""
tests/test_advanced_report.py
Standalone tests for the unified ForensicReport schema.
"""
from advanced_forensics import ensemble as EN
from advanced_forensics.report import ForensicReport


def _signals():
    return {
        "xception": {"available": True, "score": 0.82,
                     "notes": ["local checkpoint"]},
        "landmark": {"available": True, "score": 0.4},
        "motion": {"available": False, "score": None},
        "frequency": {"available": True, "score": 0.7},
        "temporal": {"available": True, "score": 0.6},
        "transformer": {"available": False, "score": None,
                        "notes": ["no local weights"]},
        "rppg": {"available": False, "score": None},
    }


def test_from_signals_marks_gaps():
    signals = _signals()
    ens = EN.combine(signals)
    rep = ForensicReport.from_signals("clip.mp4", signals, ens)
    assert rep.final_verdict in ("REAL", "FAKE", "UNCERTAIN")
    assert "motion" in rep.unavailable
    assert "transformer" in rep.unavailable
    assert rep.transformer_score is None
    assert rep.xception_probability == 0.82
    assert rep.blink_ear_consistency is None
    assert any("partial evidence" in w for w in rep.warnings)
    d = rep.to_dict()
    assert d["source"] == "clip.mp4"
    assert set(d) >= {"ensemble_score", "confidence", "reliability",
                      "unavailable", "warnings", "created_at"}
    assert isinstance(rep.summary(), str) and "clip.mp4" in rep.summary()


def test_empty_report_is_honest():
    rep = ForensicReport.from_signals("x", {}, EN.combine({}))
    assert rep.final_verdict == "UNCERTAIN"
    assert rep.confidence == 0.0
