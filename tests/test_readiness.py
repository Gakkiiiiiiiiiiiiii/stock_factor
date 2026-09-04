from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from stock_factor.adapters.postgres import Database
from stock_factor.api.main import create_app
from stock_factor.application.mining import FactorMiningService
from stock_factor.application.readiness import ReadinessAdmissionError, ReadinessService
from stock_factor.application.service import FactorApplication
from stock_factor.config.runtime import RuntimeConfig
from stock_factor.domain.authority import PaperAuthority, RuntimeProfile
from stock_factor.observability.metrics import LowCardinalityMetrics


class _Oos:
    start_or_resume = renew = put_checkpoint = seal = lambda *args, **kwargs: None


class _Jobs:
    def __init__(self):
        self.created = []

    def create(self, job):
        self.created.append(job)
        return job


class _Factors:
    def list_active(self, limit=20):
        return []

    def get(self, factor_id):
        return None


class _Provider:
    def __init__(self):
        self.calls = 0

    def get_daily_bars(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("formal admission must precede market access")


def _config() -> RuntimeConfig:
    return RuntimeConfig.from_env(
        profile=RuntimeProfile.TEST,
        paper_authority=PaperAuthority.QUANT,
        quant_base_url="http://quant",
        required_quant_checksum="sha256:quant",
        required_content_checksum="sha256:content",
    )


def _ready_service() -> ReadinessService:
    database = Database("sqlite:///:memory:")
    database.create_schema()
    now = datetime.now(UTC).isoformat()
    return ReadinessService(
        _config(),
        database=database,
        artifact_store=SimpleNamespace(append_only=True),
        oos_repository=_Oos(),
        market_probe=lambda: {"authority": "quant", "observed_at": now, "as_of": now},
        content_probe=lambda: {
            "authority": "stock_content",
            "contract": "content-factor-signal.v5.1",
            "observed_at": now,
            "as_of": now,
        },
        resource_probe=lambda: {"queue_depth": 1, "deadline_seconds": 60, "memory_mb": 1024},
    )


def test_readiness_states_are_separate_and_evidence_is_frozen(monkeypatch):
    monkeypatch.setattr(
        "stock_factor.application.readiness._contract_inventory",
        lambda: {"revision": "test", "manifest_version": "test", "checksums": {"factor.v1": "x"}, "valid": True},
    )
    service = _ready_service()
    report = service.research()
    assert report.ready is True
    assert report.verify()
    assert report.frozen_at.endswith("+00:00")
    assert report.checks["contracts"]["checksums"]
    with pytest.raises(TypeError):
        report.checks["runtime"] = {}  # type: ignore[index]
    assert service.liveness()["ready"] is True
    assert service.ml().ready is False
    assert service.paper().ready is False
    frozen = report.to_dict()
    assert service.revalidate_oos({"readiness_evidence": frozen}, report.evidence_hash).verify()
    frozen["checks"]["resources"]["memory_mb"] = 1
    with pytest.raises(ReadinessAdmissionError):
        service.revalidate_oos({"readiness_evidence": frozen}, report.evidence_hash)
    frozen = report.to_dict()
    frozen["frozen_at"] = "2020-01-01T00:00:00+00:00"
    with pytest.raises(ReadinessAdmissionError):
        service.revalidate_oos({"readiness_evidence": frozen}, report.evidence_hash)


def test_formal_admission_fails_before_provider_or_job_side_effects():
    jobs = _Jobs()
    market = _Provider()
    content = SimpleNamespace(load_signals=lambda *args, **kwargs: pytest.fail("content must not be loaded"))
    mining = FactorMiningService(market, content, _Factors())
    application = FactorApplication(
        jobs,
        _Factors(),
        mining,
        market,
        content,
        SimpleNamespace(),
        _config(),
        readiness_service=ReadinessService(_config()),
    )
    with pytest.raises(ReadinessAdmissionError) as error:
        application.create_mining_job({"research_mode": "FORMAL", "symbols": ["000001.SH"]})
    assert "MARKET_SNAPSHOT_NOT_READY" in error.value.report.blocking_reasons
    assert market.calls == 0
    assert jobs.created == []


def test_snapshot_freshness_requires_timezone_aware_recent_observation():
    service = ReadinessService(
        _config(),
        market_probe=lambda: {
            "authority": "quant",
            "freshness": "READY",
            "observed_at": "2020-01-01T00:00:00+00:00",
            "as_of": "2020-01-01T00:00:00+00:00",
        },
    )
    report = service.research()
    assert report.checks["market_snapshot_authority"]["freshness"] == "STALE"
    assert "MARKET_SNAPSHOT_NOT_READY" in report.blocking_reasons


def test_snapshot_as_of_must_not_be_after_observation():
    observed = datetime.now(UTC)
    historical = ReadinessService(
        _config(),
        market_probe=lambda: {
            "authority": "quant",
            "observed_at": observed.isoformat(),
            "as_of": (observed - timedelta(hours=1)).isoformat(),
        },
    ).research()
    assert historical.checks["market_snapshot_authority"]["freshness"] == "READY"
    future = ReadinessService(
        _config(),
        market_probe=lambda: {
            "authority": "quant",
            "observed_at": observed.isoformat(),
            "as_of": (observed + timedelta(hours=1)).isoformat(),
        },
    ).research()
    assert future.checks["market_snapshot_authority"]["freshness"] == "STALE"


def test_database_schema_missing_is_not_ready():
    database = Database("sqlite:///:memory:")
    report = ReadinessService(_config(), database=database).research()
    assert report.checks["database"]["status"] == "NOT_READY"
    assert "DATABASE_SCHEMA_NOT_READY" in report.blocking_reasons


def test_contract_manifest_checksum_drift_is_not_ready(monkeypatch):
    monkeypatch.setattr(
        "stock_factor.application.readiness._contract_inventory",
        lambda: {"revision": "test", "checksums": {}, "valid": False, "errors": ["checksum_mismatch"]},
    )
    report = ReadinessService(_config()).research()
    assert "CONTRACT_INVENTORY_MISSING" in report.blocking_reasons


def test_readiness_endpoint_matrix(tmp_path):
    from stock_factor.api.dependencies import build_application

    application = build_application(f"sqlite:///{tmp_path / 'readiness.db'}")
    client = TestClient(create_app(application))
    for path in ("/health/live", "/health/research-ready", "/health/ml-ready", "/health/paper-ready"):
        response = client.get(path)
        assert response.status_code == 200
        assert "ready" in response.json()


def test_metrics_reject_unbounded_labels_and_nonfinite_values():
    metrics = LowCardinalityMetrics()
    metrics.observe("mining_latency_seconds", 1.0, {"stage": "evaluate", "mode": "EXPLORATORY"})
    with pytest.raises(ValueError):
        metrics.observe("mining_latency_seconds", 1.0, {"factor_id": "f-1"})
    with pytest.raises(ValueError):
        metrics.observe("mining_latency_seconds", float("nan"))
