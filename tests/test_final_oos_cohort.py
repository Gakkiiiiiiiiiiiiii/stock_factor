"""P0 F-02：Final OOS cohort 一次性授权语义测试。

一个 Experiment → 一个 Finalist Cohort → 一次打开/消费 Final OOS authorization
→ 同一个 FinalOosDatasetRef → 评估整个 cohort → 全部成功后 OOS_EVALUATED。
"""

from __future__ import annotations

import numpy as np
import pytest

from stock_factor.application.final_oos_evaluation import (
    FinalOosAuthorizationError,
    FinalOosCandidateInput,
    FinalOosEvaluationService,
    InMemoryCandidateSealStore,
)
from stock_factor.application.oos_authorization import (
    CONSUMED,
    InMemoryOosAuthorizationStore,
)
from stock_factor.domain.experiment import ResearchExperiment
from stock_factor.engine.oos_seal import CandidateFreeze, OosWindowInvalidatedError
from stock_factor.engine.research_split import FactorResearchSplit

OOS_SNAPSHOT_ID = "mds-oos-cohort"


class CountingAuthorizationStore(InMemoryOosAuthorizationStore):
    """记录 consume 调用次数，验证 cohort 只消费一次授权。"""

    def __init__(self) -> None:
        super().__init__()
        self.consume_calls = 0

    def consume(self, experiment_id: str) -> dict:
        self.consume_calls += 1
        return super().consume(experiment_id)


def _experiment_at_authorized() -> ResearchExperiment:
    experiment = ResearchExperiment(symbols=["000001.SZ"])
    experiment.transition("DISCOVERY_RUNNING")
    experiment.transition("DISCOVERY_COMPLETED")
    experiment.transition("FINALIST_SELECTED")
    experiment.transition("FROZEN")
    experiment.authorize_oos()
    return experiment


def _freeze(candidate_hash: str, oos_snapshot_id: str = OOS_SNAPSHOT_ID) -> CandidateFreeze:
    return CandidateFreeze(
        candidate_hash=candidate_hash,
        formula=["close"],
        dsl_version="factor-dsl.v1",
        feature_set_version="features-1",
        discovery_snapshot_id="mds-discovery-1",
        final_oos_snapshot_id=oos_snapshot_id,
    )


def _split() -> FactorResearchSplit:
    return FactorResearchSplit(
        warmup_start=0, discovery_start=10, discovery_end=80, final_oos_start=80, final_oos_end=100
    )


def _panel(seed: int):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(20, 120))
    closes = 10 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(20, 120)), axis=1))
    return values, closes


def _service() -> tuple[FinalOosEvaluationService, InMemoryCandidateSealStore, CountingAuthorizationStore]:
    seal = InMemoryCandidateSealStore()
    authorizations = CountingAuthorizationStore()
    return FinalOosEvaluationService(seal, authorizations=authorizations), seal, authorizations


def test_single_finalist_consumes_authorization_once():
    service, seal, authorizations = _service()
    experiment = _experiment_at_authorized()
    authorizations.authorize(experiment.experiment_id, OOS_SNAPSHOT_ID, "set-1")
    service.freeze_candidate(_freeze("cand-1"))
    values, closes = _panel(11)

    results = service.evaluate_cohort_for_experiment(
        experiment, [FinalOosCandidateInput("cand-1", values, closes)], _split(), horizon=5
    )

    assert len(results) == 1
    assert results[0]["final_oos_snapshot_id"] == OOS_SNAPSHOT_ID
    assert authorizations.consume_calls == 1
    assert authorizations.get(experiment.experiment_id)["status"] == CONSUMED
    assert experiment.status == "OOS_EVALUATED"
    # cohort 大小越界直接拒绝，且不消耗授权。
    fresh = _experiment_at_authorized()
    authorizations.authorize(fresh.experiment_id, OOS_SNAPSHOT_ID, "set-size")
    before = authorizations.consume_calls
    with pytest.raises(ValueError):
        service.evaluate_cohort_for_experiment(fresh, [], _split(), horizon=5)
    assert authorizations.consume_calls == before


def test_three_finalists_consume_one_authorization():
    service, seal, authorizations = _service()
    experiment = _experiment_at_authorized()
    authorizations.authorize(experiment.experiment_id, OOS_SNAPSHOT_ID, "set-3")
    cohort = []
    for index in range(1, 4):
        candidate_hash = f"cand-{index}"
        service.freeze_candidate(_freeze(candidate_hash))
        values, closes = _panel(index)
        cohort.append(FinalOosCandidateInput(candidate_hash, values, closes))

    results = service.evaluate_cohort_for_experiment(experiment, cohort, _split(), horizon=5)

    assert len(results) == 3
    # 三个 finalist 共享同一次授权消费，禁止每 candidate 一份 authorization。
    assert authorizations.consume_calls == 1
    assert authorizations.get(experiment.experiment_id)["status"] == CONSUMED
    # 全部成功后实验才进入 OOS_EVALUATED。
    assert experiment.status == "OOS_EVALUATED"
    for result in results:
        assert result["final_oos_snapshot_id"] == OOS_SNAPSHOT_ID


def test_all_finalists_use_same_final_oos_snapshot():
    service, seal, authorizations = _service()
    experiment = _experiment_at_authorized()
    authorizations.authorize(experiment.experiment_id, OOS_SNAPSHOT_ID, "set-mixed")
    service.freeze_candidate(_freeze("cand-a", OOS_SNAPSHOT_ID))
    service.freeze_candidate(_freeze("cand-b", "mds-oos-other"))
    values_a, closes_a = _panel(21)
    values_b, closes_b = _panel(22)
    cohort = [
        FinalOosCandidateInput("cand-a", values_a, closes_a),
        FinalOosCandidateInput("cand-b", values_b, closes_b),
    ]

    with pytest.raises(FinalOosAuthorizationError):
        service.evaluate_cohort_for_experiment(experiment, cohort, _split(), horizon=5)

    # 校验失败发生在 consume 之前：授权未被消费，实验状态不变。
    assert authorizations.consume_calls == 0
    assert authorizations.get(experiment.experiment_id)["status"] == "AUTHORIZED"
    assert experiment.status == "OOS_AUTHORIZED"


def test_second_worker_cannot_consume_same_experiment():
    service, seal, authorizations = _service()
    experiment = _experiment_at_authorized()
    authorizations.authorize(experiment.experiment_id, OOS_SNAPSHOT_ID, "set-race")
    service.freeze_candidate(_freeze("cand-1"))
    values, closes = _panel(31)
    cohort = [FinalOosCandidateInput("cand-1", values, closes)]

    service.evaluate_cohort_for_experiment(experiment, cohort, _split(), horizon=5)
    assert authorizations.consume_calls == 1

    # 第二个 worker 以同一 experiment_id 再次消费：必须被拒绝。
    second = _experiment_at_authorized()
    second.experiment_id = experiment.experiment_id
    with pytest.raises(FinalOosAuthorizationError):
        service.evaluate_cohort_for_experiment(second, cohort, _split(), horizon=5)
    assert authorizations.consume_calls == 2  # 尝试了但失败，未产生第二次成功消费
    assert authorizations.get(experiment.experiment_id)["status"] == CONSUMED


def test_partial_failure_does_not_mark_experiment_evaluated():
    service, seal, authorizations = _service()
    experiment = _experiment_at_authorized()
    authorizations.authorize(experiment.experiment_id, OOS_SNAPSHOT_ID, "set-partial")
    service.freeze_candidate(_freeze("cand-1"))
    service.freeze_candidate(_freeze("cand-2"))
    # 中途失效第二个 candidate 的 OOS 窗口：consume 后才暴露。
    service.report_feedback_into_search("cand-2")
    values_1, closes_1 = _panel(41)
    values_2, closes_2 = _panel(42)
    cohort = [
        FinalOosCandidateInput("cand-1", values_1, closes_1),
        FinalOosCandidateInput("cand-2", values_2, closes_2),
    ]

    with pytest.raises(OosWindowInvalidatedError):
        service.evaluate_cohort_for_experiment(experiment, cohort, _split(), horizon=5)

    # 中途失败不得留下“实验已成功”的假状态。
    assert experiment.status == "OOS_AUTHORIZED"
    # 已成功的第一个 candidate 评估已确定性记录（partial resume 可复用）。
    assert seal.get_evaluation("cand-1", "mds-discovery-1") is not None
    assert seal.get_evaluation("cand-2", "mds-discovery-1") is None

    # 未冻结的 candidate：校验在 consume 之前失败，不消耗授权。
    fresh = _experiment_at_authorized()
    authorizations.authorize(fresh.experiment_id, OOS_SNAPSHOT_ID, "set-unfrozen")
    before = authorizations.consume_calls
    with pytest.raises(Exception):
        service.evaluate_cohort_for_experiment(
            fresh, [FinalOosCandidateInput("cand-unfrozen", values_1, closes_1)], _split(), horizon=5
        )
    assert authorizations.consume_calls == before
    assert fresh.status == "OOS_AUTHORIZED"


def test_replay_does_not_reopen_oos_dataset():
    service, seal, authorizations = _service()
    experiment = _experiment_at_authorized()
    authorizations.authorize(experiment.experiment_id, OOS_SNAPSHOT_ID, "set-replay")
    service.freeze_candidate(_freeze("cand-1"))
    values, closes = _panel(51)

    results = service.evaluate_cohort_for_experiment(
        experiment, [FinalOosCandidateInput("cand-1", values, closes)], _split(), horizon=5
    )
    consumed_at = authorizations.get(experiment.experiment_id)["consumed_at"]

    # 确定性重放：直接返回已记录结果，不重新打开/消费 OOS。
    replayed = service.evaluate("cand-1", values, closes, _split(), horizon=5)
    assert replayed == results[0]
    assert authorizations.consume_calls == 1
    assert authorizations.get(experiment.experiment_id)["consumed_at"] == consumed_at

    # 实验已 OOS_EVALUATED：再次发起 cohort 评估被拒绝，不会重开 OOS。
    with pytest.raises(FinalOosAuthorizationError):
        service.evaluate_cohort_for_experiment(
            experiment, [FinalOosCandidateInput("cand-1", values, closes)], _split(), horizon=5
        )
