from fastapi.testclient import TestClient

from stock_factor.api.dependencies import build_application
from stock_factor.api.main import create_app
from tests.test_integration import FixtureContent, FixtureMarket


def test_mining_and_paper_contracts(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_RUNTIME_PROFILE", "test")
    monkeypatch.setenv("FACTOR_PAPER_AUTHORITY", "quant")
    monkeypatch.delenv("ALLOW_LOCAL_PAPER", raising=False)
    application = build_application(f"sqlite:///{tmp_path / 'api.db'}", FixtureMarket(), FixtureContent())
    client = TestClient(create_app(application))
    response = client.post("/api/v1/mining/jobs", json={"symbols": ["600000"]})
    assert response.status_code == 200
    task_id = response.json()["data"]["job_id"]
    assert client.get(f"/api/v1/mining/jobs/{task_id}").json()["data"]["status"] == "PENDING"
    orders = client.post(
        "/api/v1/paper/orders/generate",
        json={
            "scores": [{"symbol": "600000", "score": 0.8}],
            "as_of": "2026-08-12",
            "data_snapshot_id": "snapshot-1",
            "top_k": 1,
        },
    )
    assert orders.status_code == 503
    assert orders.json()["detail"]["code"] == "QUANT_UNAVAILABLE"
    assert client.get("/api/v1/paper/state").status_code == 503


def test_legacy_agent_empty_symbol_request_uses_service_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_RUNTIME_PROFILE", "test")
    monkeypatch.setenv("FACTOR_PAPER_AUTHORITY", "quant")
    monkeypatch.delenv("ALLOW_LOCAL_PAPER", raising=False)
    application = build_application(f"sqlite:///{tmp_path / 'compat.db'}", FixtureMarket(), FixtureContent())
    client = TestClient(create_app(application))
    response = client.post("/api/v1/mining/jobs", json={"rounds": 10, "candidates_per_round": 5})
    assert response.status_code == 200
    assert response.json()["data"]["request"]["symbols"] == ["000001.SH", "399001.SZ"]
