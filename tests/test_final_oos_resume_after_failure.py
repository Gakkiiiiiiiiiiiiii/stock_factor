from __future__ import annotations

import numpy as np
import pytest

from stock_factor.application.final_oos_evaluation import (
    FinalOosCandidateInput,
    FinalOosEvaluationService,
    InMemoryCandidateSealStore,
)
from stock_factor.application.oos_run_service import OosRunService
from stock_factor.domain.experiment import ResearchExperiment
from stock_factor.domain.oos_run import AuthorizationStatus, canonical_candidate_set_hash
from stock_factor.engine.oos_seal import CandidateFreeze
from stock_factor.engine.research_split import FactorResearchSplit
from stock_factor.ports.oos_run_repository import InMemoryOosRunRepository


def _panel(seed: int):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(8, 120))
    closes = 10 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(8, 120)), axis=1))
    return values, closes


def _service():
    repo = InMemoryOosRunRepository()
    service = OosRunService(repo)
    auth = service.authorize("exp-resume", canonical_candidate_set_hash(["c1", "c2"]), "dataset-ref", "market-1")
    return service, repo, auth


def _single_service():
    repo = InMemoryOosRunRepository()
    service = OosRunService(repo)
    auth = service.authorize("exp-seal", canonical_candidate_set_hash(["c1"]), "dataset-ref", "market-1")
    return service, repo, auth


def test_resume_after_failure_only_evaluates_uncompleted_candidates():
    service, repo, auth = _service()
    candidates = [{"candidate_id": "c1", "values": [1]}, {"candidate_id": "c2", "values": [2]}]
    calls: list[str] = []
    failed = True

    def evaluate(candidate):
        nonlocal failed
        calls.append(candidate["candidate_id"])
        if candidate["candidate_id"] == "c2" and failed:
            failed = False
            raise RuntimeError("temporary evaluator failure")
        return {"candidate_id": candidate["candidate_id"], "score": 1.0}

    with pytest.raises(RuntimeError):
        service.evaluate_cohort(auth.authorization_id, candidates, evaluate, run_id="run-1", owner_id="worker-1")
    result = service.evaluate_cohort(auth.authorization_id, candidates, evaluate, run_id="run-1", owner_id="worker-1")
    assert calls == ["c1", "c2", "c2"]
    assert result["status"] == "SEALED"
    assert repo.get_authorization(auth.authorization_id).status == AuthorizationStatus.CONSUMED
    assert repo.get_run("run-1").cohort_artifact_hash == result["cohort_artifact_hash"]


def test_failed_seal_does_not_consume_authorization_and_can_resume():
    service, repo, auth = _single_service()
    original_seal = repo.seal
    state = {"fail": True}

    def flaky_seal(*args, **kwargs):
        if state["fail"]:
            state["fail"] = False
            raise RuntimeError("persistence unavailable")
        return original_seal(*args, **kwargs)

    repo.seal = flaky_seal
    candidate = [{"candidate_id": "c1", "values": [1]}]
    with pytest.raises(RuntimeError):
        service.evaluate_cohort(auth.authorization_id, candidate, lambda _: {"score": 1}, run_id="run-1", owner_id="w")
    assert repo.get_authorization(auth.authorization_id).status == AuthorizationStatus.EVALUATING_INTERRUPTED
    result = service.evaluate_cohort(
        auth.authorization_id, candidate, lambda _: {"score": 1}, run_id="run-1", owner_id="w"
    )
    assert result["status"] == "SEALED"
    assert repo.get_authorization(auth.authorization_id).status == AuthorizationStatus.CONSUMED


def test_final_service_resume_does_not_reload_completed_candidate():
    repo = InMemoryOosRunRepository()
    service = FinalOosEvaluationService(InMemoryCandidateSealStore(), run_repository=repo)
    experiment = ResearchExperiment(symbols=["000001.SZ"], experiment_id="exp-lazy")
    for status in ("DISCOVERY_RUNNING", "DISCOVERY_COMPLETED", "FINALIST_SELECTED", "FROZEN"):
        experiment.transition(status)
    experiment.authorize_oos()
    for candidate_id in ("c1", "c2"):
        service.freeze_candidate(
            CandidateFreeze(
                candidate_hash=candidate_id,
                formula=["close"],
                dsl_version="factor-dsl.v1",
                feature_set_version="features-1",
                discovery_snapshot_id="discovery-1",
                final_oos_snapshot_id="final-1",
            )
        )
    auth = service.register_authorization(
        experiment.experiment_id,
        "final-1",
        canonical_candidate_set_hash(["c1", "c2"]),
        dataset_ref_hash="final-dataset",
        market_snapshot_id="market-1",
    )
    assert auth is not None
    values, closes = _panel(101)
    loads = {"c1": 0, "c2": 0}
    failed = {"c2": True}

    def loader(candidate_id):
        loads[candidate_id] += 1
        if candidate_id == "c2" and failed["c2"]:
            failed["c2"] = False
            raise RuntimeError("temporary OOS load failure")
        return values, closes

    split = FactorResearchSplit(
        warmup_start=0, discovery_start=10, discovery_end=80, final_oos_start=80, final_oos_end=100
    )
    candidates = [
        FinalOosCandidateInput(
            candidate_hash=candidate_id,
            loader=lambda candidate_id=candidate_id: loader(candidate_id),
            input_identity=f"input-{candidate_id}-final-dataset",
        )
        for candidate_id in ("c1", "c2")
    ]
    with pytest.raises(RuntimeError):
        service.evaluate_cohort_for_experiment(experiment, candidates, split, horizon=5, run_id="run-lazy")
    service.evaluate_cohort_for_experiment(experiment, candidates, split, horizon=5, run_id="run-lazy")
    assert loads == {"c1": 1, "c2": 2}
