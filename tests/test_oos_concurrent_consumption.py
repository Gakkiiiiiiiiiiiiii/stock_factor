"""Final OOS 数据库级一次性授权（详细修改方案 P0-2）。

- AUTHORIZED -> CONSUMED 事务原子：并发消费只有一个成功；
- 已消费/缺失/失效的授权禁止再次评估；
- FinalOosEvaluationService 接入授权存储后双重评估被拒绝。
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from stock_factor.adapters.postgres.models import Base
from stock_factor.application.final_oos_evaluation import (
    FinalOosAuthorizationError,
    FinalOosEvaluationService,
    InMemoryCandidateSealStore,
)
from stock_factor.application.oos_authorization import (
    CONSUMED,
    InMemoryOosAuthorizationStore,
    OosAuthorizationConsumedError,
    OosAuthorizationMissingError,
    PostgresOosAuthorizationStore,
)
from stock_factor.domain.experiment import ResearchExperiment
from stock_factor.engine.oos_seal import CandidateFreeze
from stock_factor.engine.research_split import FactorResearchSplit


@pytest.fixture()
def pg_store(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{(tmp_path / 'factor.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return PostgresOosAuthorizationStore(sessionmaker(engine, expire_on_commit=False))


def test_authorization_lifecycle(pg_store):
    record = pg_store.authorize("exp-1", "mds-oos-1", "sethash-1")
    assert record["status"] == "AUTHORIZED"
    consumed = pg_store.consume("exp-1")
    assert consumed["status"] == CONSUMED and consumed["consumed_at"]
    with pytest.raises(OosAuthorizationConsumedError):
        pg_store.consume("exp-1")
    with pytest.raises(OosAuthorizationMissingError):
        pg_store.consume("exp-none")


def test_concurrent_consumption_single_winner(pg_store):
    pg_store.authorize("exp-race", "mds-oos-race", "sethash-race")
    results: list[str] = []
    lock = threading.Lock()

    def worker():
        try:
            pg_store.consume("exp-race")
            outcome = "consumed"
        except OosAuthorizationConsumedError:
            outcome = "rejected"
        with lock:
            results.append(outcome)

    barrier = threading.Barrier(8)

    def race():
        barrier.wait()
        worker()

    threads = [threading.Thread(target=race) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count("consumed") == 1, f"并发双重消费未被阻止: {results}"
    assert results.count("rejected") == 7


def _experiment_at_authorized() -> ResearchExperiment:
    experiment = ResearchExperiment(symbols=["000001.SZ"])
    experiment.transition("DISCOVERY_RUNNING")
    experiment.transition("DISCOVERY_COMPLETED")
    experiment.transition("FINALIST_SELECTED")
    experiment.transition("FROZEN")
    experiment.authorize_oos()
    return experiment


def test_evaluation_service_consumes_authorization_once():
    seal = InMemoryCandidateSealStore()
    authorizations = InMemoryOosAuthorizationStore()
    service = FinalOosEvaluationService(seal, authorizations=authorizations)
    experiment = _experiment_at_authorized()
    authorizations.authorize(experiment.experiment_id, "mds-oos-x", "set-x")

    freeze = CandidateFreeze(
        candidate_hash="cand-1",
        formula=["close"],
        dsl_version="factor-dsl.v1",
        feature_set_version="features-1",
        discovery_snapshot_id="mds-discovery-1",
        final_oos_snapshot_id="mds-oos-x",
    )
    service.freeze_candidate(freeze)
    split = FactorResearchSplit(warmup_start=0, discovery_start=10, discovery_end=80, final_oos_start=80, final_oos_end=100)
    rng = np.random.default_rng(3)
    values = rng.normal(size=(20, 120))
    closes = 10 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(20, 120)), axis=1))

    metrics = service.evaluate_for_experiment(experiment, "cand-1", values, closes, split, horizon=5)
    assert metrics["final_oos_snapshot_id"] == "mds-oos-x"
    assert authorizations.get(experiment.experiment_id)["status"] == CONSUMED

    # 第二个 finalist（同一实验）：授权已消费 -> 拒绝评估。
    other = _experiment_at_authorized()
    other.experiment_id = experiment.experiment_id
    with pytest.raises(FinalOosAuthorizationError):
        service.evaluate_for_experiment(other, "cand-1", values, closes, split, horizon=5)


def test_evaluation_without_authorization_record_rejected():
    seal = InMemoryCandidateSealStore()
    service = FinalOosEvaluationService(seal, authorizations=InMemoryOosAuthorizationStore())
    experiment = _experiment_at_authorized()
    split = FactorResearchSplit(warmup_start=0, discovery_start=10, discovery_end=80, final_oos_start=80, final_oos_end=100)
    values = np.ones((20, 120))
    closes = np.linspace(10, 12, 120)[None, :].repeat(20, axis=0)
    with pytest.raises(FinalOosAuthorizationError):
        service.evaluate_for_experiment(experiment, "cand-x", values, closes, split, horizon=5)
