"""On-demand Alpha Score 契约测试（设计文档 §14.2 / §79 / §33）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from stock_factor.adapters.postgres.repositories import PostgresFactorRepository
from stock_factor.api.dependencies import build_application
from stock_factor.api.main import create_app
from stock_factor.domain.factor import FactorDefinition
from tests.test_integration import FixtureContent, FixtureMarket


def _application(tmp_path):
    url = f"sqlite:///{tmp_path / 'alpha.db'}"
    application = build_application(url, FixtureMarket(), FixtureContent())
    return application, url


def test_alpha_score_contract_fields(tmp_path):
    application, url = _application(tmp_path)
    repository = PostgresFactorRepository(application._factors._sessions)  # noqa: SLF001
    repository.save(
        FactorDefinition(
            factor_id="f-1",
            name="reversal",
            rpn=["ret", "ts_mean_5", "neg", "cs_rank"],
            status="ACTIVE",
            candidate_hash="hash-a",
        )
    )
    client = TestClient(create_app(application))
    symbols = [f"6000{index:02d}" for index in range(20)]
    response = client.post("/api/v1/alpha/score", json={"symbols": symbols, "as_of": None, "factor_set": "production"})
    assert response.status_code == 200
    data = response.json()["data"]
    # §14.2 契约字段
    assert data["factor_set_version"].startswith("factor-set-")
    assert data["market_snapshot_id"] == data["data_snapshot_id"]
    assert data["scores"]
    scored = [item for item in data["scores"] if item["score"] is not None]
    assert scored
    ranks = [item["rank"] for item in scored]
    assert sorted(ranks) == list(range(1, len(scored) + 1))
    # 分数降序对应 rank 升序
    by_rank = sorted(scored, key=lambda item: item["rank"])
    scores_by_rank = [item["score"] for item in by_rank]
    assert scores_by_rank == sorted(scores_by_rank, reverse=True)
    evidence = scored[0]["evidence"]
    assert evidence and evidence[0]["factor_id"] == "f-1"
    assert evidence[0]["contribution"] is not None
    # 兼容旧消费方结构保留
    assert len(data["items"]) == len(symbols)


def test_mining_job_idempotency_key(tmp_path):
    application, _ = _application(tmp_path)
    payload = {"symbols": ["600000", "600001"], "candidates": [{"name": "x", "rpn": ["ret", "cs_rank"]}]}
    first = application.create_mining_job(payload, idempotency_key="mine-1")
    second = application.create_mining_job(payload, idempotency_key="mine-1")
    assert second["job_id"] == first["job_id"]
    third = application.create_mining_job(payload, idempotency_key="mine-2")
    assert third["job_id"] != first["job_id"]


def test_distinct_jobs_get_distinct_default_experiment_identity(tmp_path):
    application, _ = _application(tmp_path)
    payload = {"symbols": ["600000"], "candidates": [{"name": "x", "rpn": ["ret", "cs_rank"]}]}
    first = application.create_mining_job(payload)
    second = application.create_mining_job(payload)
    assert first["job_id"] != second["job_id"]
    first_request = application.get_mining_job(first["job_id"])["request"]
    second_request = application.get_mining_job(second["job_id"])["request"]
    assert first_request["experiment_id"] != second_request["experiment_id"]


def test_health_version_endpoint(tmp_path):
    application, _ = _application(tmp_path)
    client = TestClient(create_app(application))
    response = client.get("/health/version")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "stock_factor"
    assert "factor.v1" in payload["contract_versions"]


def test_trace_header_echoed(tmp_path):
    application, _ = _application(tmp_path)
    client = TestClient(create_app(application))
    response = client.get("/healthz", headers={"x-trace-id": "trace-factor-1"})
    assert response.headers.get("x-trace-id") == "trace-factor-1"
