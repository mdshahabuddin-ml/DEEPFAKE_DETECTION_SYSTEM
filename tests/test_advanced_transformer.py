"""
tests/test_advanced_transformer.py
Standalone tests for the transformer audit + adapter (no downloads).
"""
import pytest

from advanced_forensics import transformer_adapter as TA


def test_audit_reports_unavailable_with_reasons():
    rep = TA.audit_transformers()
    assert rep["available"] is False  # no local weights by repo policy
    assert isinstance(rep["reasons"], list) and len(rep["reasons"]) > 0
    assert rep["local_weights"] == []


def test_audit_on_missing_dir():
    rep = TA.audit_transformers(models_dir="definitely_not_a_dir_xyz")
    assert rep["available"] is False
    assert rep["local_weights"] == []


def test_adapter_never_fabricates_scores():
    adap = TA.TransformerAdapter()
    assert adap.available is False
    out = adap.analyze()
    assert out["available"] is False and out["score"] is None
    with pytest.raises(NotImplementedError):
        adap.load("models/vit_detector.pth")
