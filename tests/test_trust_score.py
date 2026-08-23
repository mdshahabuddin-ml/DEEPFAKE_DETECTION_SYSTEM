"""Tests for the Trust Score engine and PDF report generator."""

import numpy as np

from src.trust_score import (
    TrustReport, aggregate_trust, lip_sync_score, spectral_texture_score,
    temporal_consistency_score, _verdict_and_reco,
)


def test_temporal_consistency_stable_vs_flicker():
    stable = temporal_consistency_score([0.10, 0.12, 0.09, 0.11])
    flicker = temporal_consistency_score([0.10, 0.90, 0.20, 0.85])
    assert stable is not None and flicker is not None
    assert stable > 0.8          # consistent -> high
    assert flicker < 0.2         # flickering -> low


def test_temporal_consistency_needs_two_frames():
    assert temporal_consistency_score([0.5]) is None
    assert temporal_consistency_score([]) is None


def test_lip_sync_agreement():
    assert lip_sync_score(0.2, 0.25) > 0.9      # modalities agree
    assert lip_sync_score(0.1, 0.9) < 0.3       # strong disagreement
    assert lip_sync_score(0.5, None) is None    # no audio


def test_texture_score_bounded():
    img = (np.random.rand(128, 128, 3) * 255).astype("uint8")
    s = spectral_texture_score(img)
    assert 0.0 <= s <= 1.0


def test_aggregate_ignores_none_signals():
    only_face = aggregate_trust({"face_consistency": 0.9, "temporal_consistency": None,
                                 "lip_sync": None, "texture": None})
    assert only_face == 90.0     # single available signal -> its own value * 100


def test_verdict_bands():
    assert _verdict_and_reco(90)[0] == "Likely Authentic"
    assert _verdict_and_reco(55)[0] == "Uncertain"
    assert _verdict_and_reco(20)[0] == "Likely Deepfake"


def test_pdf_report_generation(tmp_path):
    from src.report import generate_pdf_report
    report = TrustReport(
        media_type="video", file_name="clip.mp4", verdict="Likely Deepfake",
        trust_score=38.0, recommendation="Reject / escalate.",
        fake_probability=0.9,
        signals={"face_consistency": 0.1, "temporal_consistency": 0.4,
                 "lip_sync": 0.3, "texture": None},
        faces_detected=2, manipulated_frames=[41, 50, 59], frames_analyzed=64,
        file_hash="deadbeef", inference_time=1.2,
    )
    out = generate_pdf_report(report, output_path=tmp_path / "report.pdf")
    assert out.exists()
    assert out.read_bytes()[:4] == b"%PDF"   # valid PDF magic bytes
