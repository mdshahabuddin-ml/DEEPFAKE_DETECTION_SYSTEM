"""Tests for optional API-key auth and rate limiting."""

from fastapi.testclient import TestClient

import api.main as apimain
from api.main import app

client = TestClient(app)
_TXT = {"file": ("note.txt", b"x", "text/plain")}


def test_api_key_enforced_when_configured(monkeypatch):
    monkeypatch.setattr(apimain, "API_KEY", "secret")

    # Missing key -> 401
    assert client.post("/predict/image", files=_TXT).status_code == 401
    # Wrong key -> 401
    assert client.post("/predict/image", headers={"X-API-Key": "nope"},
                       files=_TXT).status_code == 401
    # Correct key -> passes the guard (then 400 bad-ext or 503 no-model)
    ok = client.post("/predict/image", headers={"X-API-Key": "secret"}, files=_TXT)
    assert ok.status_code in (400, 503)


def test_open_endpoints_never_require_key(monkeypatch):
    monkeypatch.setattr(apimain, "API_KEY", "secret")
    # System routes stay open regardless of the key.
    assert client.get("/health").status_code == 200
    assert client.get("/model/info").status_code == 200


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setattr(apimain, "API_KEY", "")        # disable auth for this test
    monkeypatch.setattr(apimain, "RATE_LIMIT_PER_MIN", 2)
    apimain._request_log.clear()

    codes = [client.post("/predict/image", files=_TXT).status_code for _ in range(3)]
    # First two pass the limiter (then 400/503); the third is throttled.
    assert codes[0] in (400, 503)
    assert codes[1] in (400, 503)
    assert codes[2] == 429
    apimain._request_log.clear()


def test_health_reports_security_flags(monkeypatch):
    monkeypatch.setattr(apimain, "API_KEY", "secret")
    monkeypatch.setattr(apimain, "RATE_LIMIT_PER_MIN", 30)
    body = client.get("/health").json()
    assert body["auth_required"] is True
    assert body["rate_limit_per_min"] == 30
