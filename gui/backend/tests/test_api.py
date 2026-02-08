"""API tests for the qlab GUI backend."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gui.backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_alphas(client):
    r = client.get("/api/alphas/")
    assert r.status_code == 200
    alphas = r.json()
    assert len(alphas) >= 10
    names = {a["name"] for a in alphas}
    assert "momentum" in names
    assert "mean_reversion_zscore" in names
    assert "low_volatility" in names


def test_get_alpha(client):
    r = client.get("/api/alphas/momentum")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "momentum"
    assert "lookback" in data["params"]


def test_get_alpha_not_found(client):
    r = client.get("/api/alphas/nonexistent")
    assert r.status_code == 404


def test_list_universes(client):
    r = client.get("/api/universes/")
    assert r.status_code == 200
    universes = r.json()
    assert len(universes) >= 3
    names = {u["name"] for u in universes}
    assert "us_large_cap_20" in names


def test_get_universe(client):
    r = client.get("/api/universes/us_large_cap_20")
    assert r.status_code == 200
    data = r.json()
    assert len(data["tickers"]) == 20
    assert "AAPL" in data["tickers"]


def test_get_universe_not_found(client):
    r = client.get("/api/universes/nonexistent")
    assert r.status_code == 404


def test_list_runs_empty(client):
    r = client.get("/api/backtest/runs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_get_run_not_found(client):
    r = client.get("/api/backtest/runs/nonexistent")
    assert r.status_code == 404


def test_get_results_not_found(client):
    r = client.get("/api/backtest/runs/nonexistent/results")
    assert r.status_code == 404
