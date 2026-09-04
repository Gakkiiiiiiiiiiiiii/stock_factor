import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stock_factor.adapters.postgres.database import Database
from stock_factor.adapters.postgres.models import ResearchArtifactRow
from stock_factor.adapters.postgres.research_artifact_repository import PostgresResearchArtifactRepository
from stock_factor.api.main import create_app
from stock_factor.api.research_artifacts import create_router
from stock_factor.application.factor_set_service import FactorSetService, InMemoryFactorSetStore
from stock_factor.application.final_oos_evaluation import FinalOosEvaluationService, InMemoryCandidateSealStore
from stock_factor.application.mining import FactorMiningService
from stock_factor.application.research_artifact_service import (
    InMemoryResearchArtifactRepository,
    ResearchArtifactError,
    ResearchArtifactService,
)
from stock_factor.config.runtime import RuntimeConfig
from stock_factor.domain.content_signal_v5 import FormalContentQuery, FormalContentRef
from stock_factor.domain.market import MarketDataSnapshot
from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef
from stock_factor.domain.research_artifact import REQUIRED_CONTRACT_CHECKSUMS, ResearchArtifactV2
from stock_factor.domain.tradability_artifact import (
    ExecutionCostCalibrationRef,
    TradabilityAssessmentArtifact,
    TradabilityAssumptions,
)
from stock_factor.ports.oos_run_repository import InMemoryOosRunRepository


def _payload(**changes):
    payload = {
        "contract_version": "research-artifact.v2",
        "artifact_status": "SEALED",
        "experiment_id": "exp-artifact",
        "research_question": "Does the signal predict returns?",
        "hypothesis": "The signal has positive rank IC.",
        "dataset_manifest": {"universe_identity": "uni-1", "market_ref_hash": "a" * 64, "content_ref_hash": "b" * 64},
        "market_ref": {"ref_hash": "a" * 64, "market_snapshot_id": "market-1"},
        "content_ref": {"ref_hash": "b" * 64, "content_snapshot_id": "content-1"},
        "candidate_set_hash": "a" * 64,
        "statistical_experiment": {
            "multiple_testing": {"passed": True},
            "dsr": {"passed": True},
            "pbo": {"passed": True},
        },
        "final_oos_evidence": {"status": "SEALED", "cohort_artifact_hash": "f" * 64, "run_id": "run-artifact"},
        "tradability_assessment": {
            "reference": "tradability-1",
            "evidence": {"passed": True},
            "formal_eligible": True,
            "gate_result": {"passed": True},
        },
        "promotion_decision": {"passed": True},
        "promotion_policy_version": "promotion_gate_v3",
        "producer_commit": "stock-factor@abc123",
        "dependency_lock_hash": "b" * 64,
        "contract_checksums": {
            "research-artifact.v2": "sha256:" + "c" * 64,
            "factor.v1": "sha256:" + "d" * 64,
            "market-snapshot.v1": "sha256:" + "e" * 64,
            "content-factor-signal.v5.1": "sha256:" + "f" * 64,
        },
        "created_at": "2026-09-04T00:00:00Z",
        "factor_set": {"factors": [{"factor_id": "f-1", "version": 1}]},
    }
    payload.update(changes)
    return payload


def _formal_tradability() -> dict:
    return TradabilityAssessmentArtifact(
        factor_artifact_id="f-1",
        market_snapshot_id="market-1",
        execution_cost_calibration=ExecutionCostCalibrationRef("cal-1", "cost-v1", "a" * 64),
        gross_metrics={},
        net_metrics={},
        turnover=0.0,
        ic_decay={},
        holding_period_sensitivity={},
        limit_hit_rate=0.0,
        halt_exposure=0.0,
        participation_rate=0.0,
        capacity_curve=(),
        neutralized_contribution={},
        assumptions=TradabilityAssumptions(),
        gate_result={"passed": True},
        formal_eligible=True,
        policy_hash="b" * 64,
    ).to_dict()


def test_hash_is_stable_for_key_order_and_utc_equivalent_time():
    first = ResearchArtifactV2.from_payload(_payload())
    reordered = dict(reversed(list(_payload().items())))
    second = ResearchArtifactV2.from_payload(reordered)
    assert first.artifact_id == second.artifact_id
    assert (
        first.artifact_id
        == ResearchArtifactV2.from_payload(_payload(created_at="2026-09-04T08:00:00+08:00")).artifact_id
    )


def test_artifact_rejects_tampering_nonfinite_and_naive_time():
    artifact = ResearchArtifactV2.from_payload(_payload())
    tampered = artifact.to_payload()
    tampered["hypothesis"] = "changed"
    tampered["artifact_id"] = artifact.artifact_id
    with pytest.raises(ValueError, match="hash mismatch"):
        ResearchArtifactV2.from_payload(tampered)
    with pytest.raises(ValueError, match="NaN"):
        ResearchArtifactV2.from_payload(
            _payload(
                final_oos_evidence={
                    "status": "SEALED",
                    "cohort_artifact_hash": "f" * 64,
                    "run_id": "run-artifact",
                    "score": float("nan"),
                }
            )
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        ResearchArtifactV2.from_payload(_payload(created_at=datetime(2026, 9, 4)))


def test_artifact_requires_complete_contract_inventory_and_real_oos_seal():
    for contract_name in REQUIRED_CONTRACT_CHECKSUMS:
        checksums = dict(_payload()["contract_checksums"])
        checksums.pop(contract_name)
        with pytest.raises(ValueError, match="contract checksums"):
            ResearchArtifactV2.from_payload(_payload(contract_checksums=checksums))
    with pytest.raises(ValueError, match="cohort_artifact_hash"):
        ResearchArtifactV2.from_payload(_payload(final_oos_evidence={"status": "SEALED"}))


def test_artifact_evidence_is_deeply_immutable_and_repository_get_isolated():
    service = ResearchArtifactService(InMemoryResearchArtifactRepository())
    artifact = service.seal(_payload())
    with pytest.raises(TypeError):
        artifact.dataset_manifest["tampered"] = "value"
    loaded = service.get(artifact.artifact_id)
    assert loaded is not None
    with pytest.raises(TypeError):
        loaded.final_oos_evidence["status"] = "FAILED"
    assert service.get(artifact.artifact_id).final_oos_evidence["status"] == "SEALED"


def test_append_only_and_verification_mismatch():
    service = ResearchArtifactService(InMemoryResearchArtifactRepository())
    artifact = service.seal(_payload())
    assert service.seal(artifact).artifact_id == artifact.artifact_id
    with pytest.raises(ResearchArtifactError, match="dependency"):
        service.verify(artifact.artifact_id, dependency_lock_hash="other")
    assert service.verify(artifact.artifact_id, dependency_lock_hash="b" * 64)["verified"] is True


def test_formal_factor_set_requires_verified_sealed_artifact():
    artifacts = ResearchArtifactService(InMemoryResearchArtifactRepository())
    factor_sets = FactorSetService(InMemoryFactorSetStore())
    artifact = artifacts.seal(_payload(tradability_assessment=_formal_tradability()))
    formal = factor_sets.publish_formal(artifact.artifact_id, artifacts)
    assert formal.formal_eligible is True
    assert formal.research_artifact_ids == (artifact.artifact_id,)
    with pytest.raises(ResearchArtifactError):
        factor_sets.publish_formal("missing", artifacts)

    legacy = factor_sets.publish_from_factors([{"factor_id": "legacy-1", "version": 1}])
    assert legacy.formal_eligible is False
    assert legacy.research_artifact_ids == ()
    with pytest.raises(ValueError, match="SEALED"):
        artifacts.seal(_payload(artifact_status="DRAFT"))

    with pytest.raises(ValueError, match="promotion decision"):
        artifacts_bad = ResearchArtifactService(InMemoryResearchArtifactRepository())
        failed = artifacts_bad.seal(_payload(promotion_decision={"passed": False}))
        factor_sets.publish_formal(failed.artifact_id, artifacts_bad)


def test_offline_cli_verifies_artifact(tmp_path):
    artifact = ResearchArtifactV2.from_payload(_payload())
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact.to_payload()), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_research_artifact.py",
            str(path),
            "--dependency-lock-hash",
            "b" * 64,
            "--contract-checksum",
            "factor.v1=sha256:" + "d" * 64,
            "--contract-checksum",
            "market-snapshot.v1=sha256:" + "e" * 64,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["verified"] is True


def test_sqlite_repository_retrieves_complete_artifact():
    database = Database("sqlite://")
    database.create_schema()
    service = ResearchArtifactService(PostgresResearchArtifactRepository(database.session_factory))
    artifact = service.seal(_payload())
    loaded = service.get(artifact.artifact_id)
    assert loaded is not None
    assert loaded.to_payload() == artifact.to_payload()


def test_artifact_migration_and_orm_are_append_only_and_timezone_bound():
    migration = (Path(__file__).parents[1] / "migrations" / "012_research_artifacts_v2.sql").read_text(encoding="utf-8")
    assert "artifact_id VARCHAR(64) PRIMARY KEY" in migration
    assert "created_at TIMESTAMPTZ NOT NULL" in migration
    assert "UPDATE" not in migration.upper()
    assert ResearchArtifactRow.__table__.c.artifact_id.primary_key is True
    assert ResearchArtifactRow.__table__.c.created_at.nullable is False


def test_research_artifact_api_seal_retrieve_verify():
    service = ResearchArtifactService(InMemoryResearchArtifactRepository())
    app = FastAPI()
    app.include_router(create_router(service))
    client = TestClient(app)
    sealed = client.post("/api/v1/research-artifacts", json=_payload())
    assert sealed.status_code == 200
    artifact_id = sealed.json()["data"]["artifact_id"]
    retrieved = client.get(f"/api/v1/research-artifacts/{artifact_id}")
    assert retrieved.status_code == 200
    verified = client.post(
        f"/api/v1/research-artifacts/{artifact_id}/verify",
        json={"dependency_lock_hash": "b" * 64},
    )
    assert verified.status_code == 200
    assert verified.json()["data"]["verified"] is True
    assert client.get(f"/api/v1/research-artifacts/{artifact_id}/verify").json()["data"]["verified"] is True
    assert client.get(f"/api/v1/research-artifacts/{artifact_id}/replay").json()["data"]["artifact_id"] == artifact_id


def test_research_artifact_api_rejects_missing_required_contract_checksum():
    service = ResearchArtifactService(InMemoryResearchArtifactRepository())
    app = FastAPI()
    app.include_router(create_router(service))
    client = TestClient(app)
    for contract_name in REQUIRED_CONTRACT_CHECKSUMS:
        checksums = dict(_payload()["contract_checksums"])
        checksums.pop(contract_name)
        response = client.post(
            "/api/v1/research-artifacts",
            json=_payload(contract_checksums=checksums),
        )
        assert response.status_code == 422


def test_mining_api_request_round_trip_preserves_formal_research_evidence_fields():
    captured = {}

    class CaptureApplication:
        runtime_config = RuntimeConfig.from_env(profile="test", paper_authority="quant")
        readiness_service = SimpleNamespace()
        research_artifact_service = None

        def create_mining_job(self, payload, idempotency_key=None):
            captured.update(payload)
            return {"request": payload}

    client = TestClient(create_app(CaptureApplication()))
    response = client.post(
        "/api/v1/mining/jobs",
        json={
            "research_question": "Does this factor predict returns?",
            "hypothesis": "The factor has positive rank IC.",
            "research_mode": "FORMAL",
        },
    )
    assert response.status_code == 200
    assert captured["research_question"] == "Does this factor predict returns?"
    assert captured["hypothesis"] == "The factor has positive rank IC."


def test_formal_mining_seals_real_oos_run_into_artifact_without_cross_run_reuse(monkeypatch):
    monkeypatch.setenv("FACTOR_DEPENDENCY_LOCK_HASH", "1" * 64)
    monkeypatch.setenv("FACTOR_GIT_COMMIT", "stock-factor@test-commit")
    symbols = [f"6000{index:02d}" for index in range(20)]
    start, end = "2024-01-01", "2025-05-14"
    dates = [(datetime(2024, 1, 1) + timedelta(days=index)).date().isoformat() for index in range(500)]
    rng = np.random.default_rng(19)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, (len(symbols), len(dates))), axis=1)
    volume = np.full_like(close, 1_000_000.0)
    market_ref = FormalMarketDatasetRef(
        market_snapshot_id="formal-market-1",
        manifest_hash="2" * 64,
        calendar_version="cn-v1",
        universe_version="universe-v1",
        corporate_action_version="ca-v1",
        tradability_version="trad-v1",
        quality_seal="PASS",
        available_from=datetime(2023, 12, 1, tzinfo=UTC),
        start=start,
        end=end,
    )
    query = FormalContentQuery(
        contract="content-factor-signal.v5.1",
        checksum="sha256:" + "3" * 64,
        content_snapshot_id="content-snapshot-1",
        business_as_of=datetime(2025, 5, 14, tzinfo=UTC),
        knowledge_as_of=datetime(2025, 5, 14, tzinfo=UTC),
        availability_as_of=datetime(2025, 5, 14, tzinfo=UTC),
        pit_mode="PUBLIC_STRICT",
        signal_policy_version="policy-v1",
        min_support=2,
        producer_commit="content@test-commit",
    )
    content_ref = FormalContentRef.from_query(query, "4" * 64)
    calibration = ExecutionCostCalibrationRef("cal-formal", "cost-v1", "5" * 64)

    class Market:
        def get_daily_bars(self, symbols, start, end, adjust, **kwargs):
            return MarketDataSnapshot(
                symbols,
                dates,
                {
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": volume,
                    "amount": volume * close,
                    "turnover": np.ones_like(close),
                },
                "fixture-formal-v1",
                "formal-market-1",
                "quant",
                formal_market_ref=market_ref,
            )

    class Content:
        def load_signals(self, symbols, start, end, **kwargs):
            return []

    class Factors:
        def __init__(self):
            self.items = []

        def save(self, factor):
            self.items.append(factor)
            return factor.to_dict()

    oos_repo = InMemoryOosRunRepository()
    oos = FinalOosEvaluationService(InMemoryCandidateSealStore(), run_repository=oos_repo)
    artifacts = ResearchArtifactService(InMemoryResearchArtifactRepository())
    factors = Factors()
    mining = FactorMiningService(
        Market(),
        Content(),
        factors,
        final_oos_service=oos,
        expected_execution_cost_calibration=calibration,
        research_artifact_service=artifacts,
    )
    request = {
        "symbols": symbols,
        "start": start,
        "end": end,
        "research_mode": "FORMAL",
        "research_question": "Does the formal signal predict returns?",
        "hypothesis": "The signal has positive rank IC.",
        "candidates": [{"name": "formal", "hypothesis": "rank IC", "rpn": ["ret", "cs_rank"]}],
        "candidate_budget": 1,
        "rounds": 1,
        "formal_market_ref": market_ref.__dict__,
        "formal_content_query": query.model_dump(mode="json"),
        "formal_content_ref": content_ref.model_dump(mode="json"),
        "execution_cost_calibration": calibration.to_dict(),
        "readiness_evidence": {
            "ready": True,
            "runtime_profile": "PROD",
            "checks": {"formal_inputs": True},
            "blocking_reasons": [],
            "threshold_version": "readiness_v1",
            "frozen_at": "2026-09-04T00:00:00Z",
            "evidence_hash": "6" * 64,
        },
    }
    result = mining.run(request)
    artifact = artifacts.get(result["research_artifact_id"])
    assert artifact is not None
    assert artifact.final_oos_evidence["run_id"]
    run = oos_repo.get_run(artifact.final_oos_evidence["run_id"])
    assert run is not None and run.cohort_artifact_hash == artifact.final_oos_evidence["cohort_artifact_hash"]
    assert run.status.value == "SEALED"
    assert oos_repo.get_authorization(run.authorization_id).status.value == "CONSUMED"
    assert artifact.statistical_experiment["multiple_testing"]
    assert artifact.statistical_experiment["dsr"]
    assert all("deflated_sharpe" in evidence for evidence in artifact.statistical_experiment["dsr"].values())
    assert all("pbo" in evidence for evidence in artifact.statistical_experiment["pbo"].values())
    assert "candidate_count" not in json.dumps(artifact.to_payload()["statistical_experiment"])
    assert artifact.tradability_assessment["artifact_id"]
    assert TradabilityAssessmentArtifact.verify_payload(artifact.tradability_assessment)
    assert artifact.readiness_evidence["evidence_hash"] == "6" * 64
    schema = json.loads(
        (Path(__file__).parents[1] / "contracts" / "research-artifact.v2.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(artifact.to_payload())

    second = mining.run({**request, "research_question": "A different formal question"})
    assert second["research_artifact_id"] != result["research_artifact_id"]
    second_artifact = artifacts.get(second["research_artifact_id"])
    assert (
        second_artifact.final_oos_evidence["cohort_artifact_hash"]
        != artifact.final_oos_evidence["cohort_artifact_hash"]
    )
