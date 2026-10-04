import time

import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture
def client(monkeypatch):
    # Without `with`, TestClient skips lifespan, so the LSE stream never starts.
    monkeypatch.setattr(app_module, "last", None)
    monkeypatch.setattr(app_module, "connected", False)
    return TestClient(app_module.app)


def test_health_before_first_tick(client):
    assert client.get("/health").json() == {"ok": True, "connected": False, "has_tick": False}


def test_tick_503_until_first_tick(client):
    assert client.get("/tick").status_code == 503


def test_tick_returns_last_with_age(client, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "last",
        {"symbol": "MANU", "price": 1.5, "bid": 1.4, "ask": 1.6, "volume": 10, "received_at": time.time() - 1},
    )
    body = client.get("/tick").json()
    assert body["price"] == 1.5
    assert body["age_ms"] >= 1000
