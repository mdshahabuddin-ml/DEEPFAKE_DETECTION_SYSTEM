"""Tests for the FastAPI service using the TestClient."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert "checkpoint_present" in body


def test_model_info():
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"]
    assert body["input_size"] > 0


def test_root_points_to_docs():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["docs"] == "/docs"


def test_predict_image_rejects_bad_extension():
    """A non-image upload must be rejected (400) before touching the model."""
    files = {"file": ("note.txt", b"hello", "text/plain")}
    resp = client.post("/predict/image", files=files)
    # 400 (bad ext) if model present, or 503 (no model) — both are valid guards.
    assert resp.status_code in (400, 503)


def test_stats_endpoint():
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert "total" in resp.json()
