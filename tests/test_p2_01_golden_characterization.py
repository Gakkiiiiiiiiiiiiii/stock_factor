"""Deterministic behavior locks for the P2-01 module split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stock_factor.application.experiments.statistics import cohort_statistics
from stock_factor.application.final_oos_evaluation import (
    FinalOosCandidateInput,
    FinalOosEvaluationService,
    InMemoryCandidateSealStore,
)
from stock_factor.application.mining.screen import canonical_candidates
from stock_factor.domain.experiment import (
    DISCOVERY_COMPLETED,
    DISCOVERY_RUNNING,
    FINALIST_SELECTED,
    FROZEN,
    ResearchExperiment,
)
from stock_factor.domain.oos_run import canonical_candidate_set_hash
from stock_factor.domain.research_artifact import ResearchArtifactV2
from stock_factor.engine.oos_seal import CandidateFreeze
from stock_factor.engine.research_split import FactorResearchSplit
from stock_factor.ports.oos_run_repository import InMemoryOosRunRepository

ROOT = Path(__file__).resolve().parents[1]


def test_split_candidate_screening_preserves_golden_order_and_universe():
    golden = json.loads((ROOT / "tests/golden/p2_01_characterization.json").read_text(encoding="utf-8"))
    candidates = [
        {"name": "ret", "rpn": ["ret", "cs_rank"]},
        {"name": "close", "rpn": ["close", "cs_rank"]},
        {"name": "volume", "rpn": ["volume", "cs_rank"]},
    ]
    screened = canonical_candidates(candidates, budget=3)
    actual = [item["candidate_hash"] for item in screened]
    assert actual == golden["candidate_order"]
    assert actual == golden["multiple_testing_universe"]


def test_split_statistics_and_oos_seal_execute_against_golden_hashes():
    golden = json.loads((ROOT / "tests/golden/p2_01_characterization.json").read_text(encoding="utf-8"))
    candidate_ids = golden["multiple_testing_universe"]
    rng = np.random.default_rng(11)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, (12, 30)), axis=1)
    evaluated = [
        {
            "candidate": {"candidate_hash": candidate_id},
            "values": rng.normal(size=(12, 30)),
            "split": SimpleNamespace(discovery_end=20),
        }
        for candidate_id in candidate_ids
    ]
    statistics = cohort_statistics(evaluated, close, horizon=1)
    statistics_hash = hashlib.sha256(json.dumps(statistics, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert statistics_hash == golden["statistics_hash"]

    candidate_id = "candidate-golden"
    repository = InMemoryOosRunRepository()
    service = FinalOosEvaluationService(InMemoryCandidateSealStore(), run_repository=repository)
    experiment = ResearchExperiment(symbols=["S"], experiment_id="golden-oos", created_at="2026-01-01T00:00:00Z")
    for status in (DISCOVERY_RUNNING, DISCOVERY_COMPLETED, FINALIST_SELECTED, FROZEN):
        experiment.transition(status)
    service.freeze_candidate(
        CandidateFreeze(
            candidate_hash=candidate_id,
            formula=["ret"],
            dsl_version="dsl-v1",
            feature_set_version="features-v1",
            discovery_snapshot_id="disc-1",
            final_oos_snapshot_id="final-1",
            experiment_id=experiment.experiment_id,
            candidate_frozen_at="2026-01-01T00:00:00Z",
        )
    )
    experiment.authorize_oos()
    service.register_authorization(
        experiment.experiment_id,
        "final-1",
        canonical_candidate_set_hash([candidate_id]),
        dataset_ref_hash="d" * 64,
        market_snapshot_id="market-1",
    )
    oos_rng = np.random.default_rng(7)
    values = oos_rng.normal(size=(20, 30))
    closes = 100 * np.cumprod(1 + oos_rng.normal(0.001, 0.01, (20, 30)), axis=1)
    evidence = service.evaluate_cohort_with_evidence(
        experiment,
        [FinalOosCandidateInput(candidate_id, values=values, closes=closes, input_identity="input-1")],
        FactorResearchSplit(0, 10, 20, 20, 30),
        horizon=1,
        run_id="run-golden",
        owner_id="owner-golden",
    )
    assert evidence.run_id == golden["oos_execution"]["run_id"]
    assert evidence.cohort_artifact_hash == golden["oos_execution"]["cohort_artifact_hash"]
    run = repository.get_run(evidence.run_id)
    assert run is not None and run.cohort_artifact_hash == evidence.cohort_artifact_hash


def test_split_golden_keeps_sealed_oos_and_artifact_identity():
    golden = json.loads((ROOT / "tests/golden/p2_01_characterization.json").read_text(encoding="utf-8"))
    fixture = json.loads(
        (ROOT / "tests/contracts/consumer/fixtures/stock_factor_to_stock_agent_research_artifact.json").read_text(
            encoding="utf-8"
        )
    )["response"]
    artifact = ResearchArtifactV2.from_payload(fixture)
    assert artifact.artifact_id == golden["research_artifact_id"]
    assert artifact.final_oos_evidence["status"] == "SEALED"
    assert artifact.final_oos_evidence["run_id"] == golden["oos"]["run_id"]
    assert artifact.final_oos_evidence["cohort_artifact_hash"] == golden["oos"]["cohort_artifact_hash"]
