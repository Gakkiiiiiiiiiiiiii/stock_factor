"""Final OOS 真隔离测试（设计文档 §13 / §78 / §86 / 验收标准 Factor 部分）。"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from stock_factor.application.final_oos_evaluation import (
    FinalOosEvaluationService,
    InMemoryCandidateSealStore,
)
from stock_factor.application.mining import FactorMiningService
from stock_factor.domain.market import MarketDataSnapshot
from stock_factor.engine.oos_seal import (
    CandidateFreeze,
    CandidateUnfrozenError,
    derive_snapshot_refs,
)
from stock_factor.engine.research_split import build_research_split
from stock_factor.research_config import get_research_config

DAYS = 520
SYMBOLS = [f"6000{index:02d}" for index in range(20)]
CANDIDATES = [{"name": "reversal", "hypothesis": "mean reversion", "rpn": ["ret", "ts_mean_5", "neg", "cs_rank"]}]


class LongMarket:
    def get_daily_bars(self, symbols, start, end, adjust):
        rng = np.random.default_rng(7)
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, (len(symbols), DAYS)), axis=1)
        volume = rng.uniform(1e6, 2e6, close.shape)
        dates = [(datetime.now(UTC).date() - timedelta(days=DAYS - index)).isoformat() for index in range(DAYS)]
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
            "fixture-v1",
            "snapshot-long",
            "fixture",
        )


class EmptyContent:
    def load_signals(self, symbols, start, end):
        return []


class FixtureFactors:
    def save(self, factor):
        return factor.to_dict()


def _build(store=None):
    service_store = store or InMemoryCandidateSealStore()
    oos_service = FinalOosEvaluationService(service_store)
    mining = FactorMiningService(LongMarket(), EmptyContent(), FixtureFactors(), final_oos_service=oos_service)
    return mining, oos_service


def test_mining_freezes_candidate_and_separates_snapshot_refs():
    mining, oos_service = _build()
    result = mining.run({"symbols": SYMBOLS, "candidates": CANDIDATES})
    factor = result["factors"][0]
    metrics = factor["metrics"]
    # §86：discovery 与 final OOS 快照引用必须分别保存且互不相同
    assert metrics["discovery_snapshot_id"].startswith("mds-discovery-")
    assert metrics["final_oos_snapshot_id"].startswith("mds-final-oos-")
    assert metrics["discovery_snapshot_id"] != metrics["final_oos_snapshot_id"]
    # §13.3：冻结记录必须存在且包含 formula/dsl/feature-set 版本
    assert metrics["candidate_frozen_at"]
    assert metrics["dsl_version"]
    assert metrics["feature_set_version"].startswith("features-")
    assert metrics["candidate_search_window"] == "discovery_only"
    freeze = oos_service._seal.get_freeze(factor["candidate_hash"])  # noqa: SLF001
    assert freeze is not None
    assert freeze.formula == CANDIDATES[0]["rpn"]


def test_preliminary_fitness_is_restricted_to_discovery_window():
    mining, _ = _build()
    result = mining.run({"symbols": SYMBOLS, "candidates": CANDIDATES})
    metrics = result["factors"][0]["metrics"]
    config = get_research_config()
    # Preliminary Fitness 不得覆盖 Final OOS 区间（§13.2）
    assert metrics["in_sample"]["evaluated_days"] <= config.data_split.discovery_days


def test_snapshot_refs_are_deterministic_and_window_scoped():
    first = derive_snapshot_refs("mds-x", (10, 20), (20, 30))
    second = derive_snapshot_refs("mds-x", (10, 20), (20, 30))
    other = derive_snapshot_refs("mds-x", (11, 20), (20, 30))
    assert first == second
    assert first.discovery_snapshot_id != other.discovery_snapshot_id
    assert first.final_oos_snapshot_id != first.discovery_snapshot_id


def test_final_oos_evaluation_requires_frozen_candidate():
    service = FinalOosEvaluationService(InMemoryCandidateSealStore())
    split = build_research_split(DAYS, get_research_config().data_split, 5)
    values = np.random.default_rng(1).normal(size=(len(SYMBOLS), DAYS))
    closes = np.random.default_rng(2).normal(size=(len(SYMBOLS), DAYS)) + 100
    with pytest.raises(CandidateUnfrozenError):
        service.evaluate("missing-hash", values, closes, split, 5)


def test_final_oos_replay_returns_recorded_evaluation():
    store = InMemoryCandidateSealStore()
    service = FinalOosEvaluationService(store)
    split = build_research_split(DAYS, get_research_config().data_split, 5)
    refs = derive_snapshot_refs("mds-x", (split.discovery_start, split.discovery_end), (split.final_oos_start, split.final_oos_end))
    freeze = CandidateFreeze(
        candidate_hash="cand-1",
        formula=["ret", "cs_rank"],
        dsl_version="factor-dsl.v1",
        feature_set_version="features-x",
        discovery_snapshot_id=refs.discovery_snapshot_id,
        final_oos_snapshot_id=refs.final_oos_snapshot_id,
    )
    service.freeze_candidate(freeze)
    values = np.random.default_rng(1).normal(size=(len(SYMBOLS), DAYS))
    closes = np.cumprod(1 + np.random.default_rng(2).normal(0.001, 0.02, (len(SYMBOLS), DAYS)), axis=1) * 100
    first = service.evaluate("cand-1", values, closes, split, 5)
    replayed = service.evaluate("cand-1", values, closes, split, 5)
    assert first == replayed
    assert first["final_oos_snapshot_id"] == refs.final_oos_snapshot_id


def test_feedback_into_search_invalidates_oos_window():
    """§13.4：OOS 结果反馈进入下一轮搜索后，该 OOS 区间立即失效。"""
    store = InMemoryCandidateSealStore()
    mining, oos_service = _build(store)
    request = {"symbols": SYMBOLS, "candidates": CANDIDATES}
    first_run = mining.run(request)
    candidate_hash = first_run["factors"][0]["candidate_hash"]
    assert first_run["factors"][0]["metrics"]["final_oos"].get("reason") != "OOS_WINDOW_INVALIDATED"

    oos_service.report_feedback_into_search(candidate_hash)

    replay = FactorMiningService(LongMarket(), EmptyContent(), FixtureFactors(), final_oos_service=oos_service)
    second_run = replay.run(request)
    final_oos = second_run["factors"][0]["metrics"]["final_oos"]
    assert final_oos == {"passed": False, "reason": "OOS_WINDOW_INVALIDATED"}
    assert store.oos_window_status(candidate_hash) == "INVALIDATED"
