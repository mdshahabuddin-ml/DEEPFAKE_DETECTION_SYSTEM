"""Tests for the SQLite prediction history."""

from src.database import PredictionDB, PredictionRecord


def _sample(pred="FAKE"):
    return PredictionRecord(
        file_name="clip.mp4", media_type="video", prediction=pred,
        confidence=0.87, fake_probability=0.87, inference_time=0.12, source="test",
    )


def test_add_and_recent(tmp_path):
    db = PredictionDB(db_path=tmp_path / "test.db")
    row_id = db.add(_sample("FAKE"))
    assert row_id == 1

    recent = db.recent(limit=10)
    assert len(recent) == 1
    assert recent[0]["prediction"] == "FAKE"
    assert recent[0]["file_name"] == "clip.mp4"
    assert recent[0]["created_at"]  # auto-filled timestamp


def test_stats_and_clear(tmp_path):
    db = PredictionDB(db_path=tmp_path / "test.db")
    db.add(_sample("FAKE"))
    db.add(_sample("REAL"))
    db.add(_sample("FAKE"))

    stats = db.stats()
    assert stats["total"] == 3
    assert stats["fake"] == 2
    assert stats["real"] == 1
    assert 0.0 <= stats["avg_confidence"] <= 1.0

    db.clear()
    assert db.stats()["total"] == 0
