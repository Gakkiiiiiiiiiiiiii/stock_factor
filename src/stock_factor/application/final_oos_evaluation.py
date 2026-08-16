"""FinalOosEvaluationService（设计文档 §13.4 / §78）。

独立的 Final OOS 评估应用服务：

- 输入：Frozen Candidate + Final OOS Snapshot + Evaluation Config
- 输出：OOS Evaluation，一次性使用
- 如果 OOS 结果反馈进入下一轮搜索：该 OOS 区间立即失效，自动降级为
  Discovery 数据（后续评估必须报告 OOS_WINDOW_INVALIDATED）。
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from stock_factor.engine.fitness import evaluate_factor_range
from stock_factor.engine.oos_seal import (
    CandidateFreeze,
    CandidateUnfrozenError,
    OosWindowInvalidatedError,
)
from stock_factor.engine.research_split import FactorResearchSplit


class CandidateSealStore(Protocol):
    """Candidate Freeze 与 OOS 评估记录的持久化端口。"""

    def save_freeze(self, freeze: CandidateFreeze) -> None: ...

    def get_freeze(self, candidate_hash: str) -> CandidateFreeze | None: ...

    def record_evaluation(self, candidate_hash: str, discovery_snapshot_id: str, metrics: dict) -> None: ...

    def get_evaluation(self, candidate_hash: str, discovery_snapshot_id: str) -> dict | None: ...

    def invalidate_oos_window(self, candidate_hash: str, reason: str) -> None: ...

    def oos_window_status(self, candidate_hash: str) -> str: ...


class InMemoryCandidateSealStore:
    """无持久化场景（单元测试/内存运行）的冻结记录。"""

    def __init__(self) -> None:
        self._freezes: dict[str, CandidateFreeze] = {}
        self._evaluations: dict[tuple[str, str], dict] = {}
        self._invalidated: dict[str, str] = {}

    def save_freeze(self, freeze: CandidateFreeze) -> None:
        self._freezes[freeze.candidate_hash] = freeze

    def get_freeze(self, candidate_hash: str) -> CandidateFreeze | None:
        return self._freezes.get(candidate_hash)

    def record_evaluation(self, candidate_hash: str, discovery_snapshot_id: str, metrics: dict) -> None:
        self._evaluations[(candidate_hash, discovery_snapshot_id)] = metrics

    def get_evaluation(self, candidate_hash: str, discovery_snapshot_id: str) -> dict | None:
        return self._evaluations.get((candidate_hash, discovery_snapshot_id))

    def invalidate_oos_window(self, candidate_hash: str, reason: str) -> None:
        self._invalidated[candidate_hash] = reason

    def oos_window_status(self, candidate_hash: str) -> str:
        return "INVALIDATED" if candidate_hash in self._invalidated else "SEALED"


class FinalOosEvaluationService:
    """一次性 Final OOS 评估。"""

    def __init__(self, seal_store: CandidateSealStore) -> None:
        self._seal = seal_store

    def freeze_candidate(self, freeze: CandidateFreeze) -> CandidateFreeze:
        """进入 Final OOS 前冻结候选（§13.3）。"""
        self._seal.save_freeze(freeze)
        return freeze

    def evaluate(
        self,
        candidate_hash: str,
        values: np.ndarray,
        closes: np.ndarray,
        split: FactorResearchSplit,
        horizon: int,
    ) -> dict[str, Any]:
        freeze = self._seal.get_freeze(candidate_hash)
        if freeze is None:
            raise CandidateUnfrozenError(f"candidate {candidate_hash} 未冻结，禁止进行 Final OOS 评估")
        status = self._seal.oos_window_status(candidate_hash)
        if status == "INVALIDATED":
            raise OosWindowInvalidatedError(
                f"candidate {candidate_hash} 的 Final OOS 区间已失效（曾被反馈进入搜索），降级为 Discovery 数据"
            )
        previous = self._seal.get_evaluation(candidate_hash, freeze.discovery_snapshot_id)
        if previous is not None:
            # 一次性使用：同一冻结快照下只允许一次真实评估；
            # 完全相同输入的确定性重放直接返回已记录结果（§4.5 可重放）。
            return previous
        metrics = evaluate_factor_range(
            values, closes, split.final_oos_start, split.final_oos_end, horizon=horizon
        )
        metrics = {
            **metrics,
            "final_oos_snapshot_id": freeze.final_oos_snapshot_id,
            "discovery_snapshot_id": freeze.discovery_snapshot_id,
            "candidate_frozen_at": freeze.candidate_frozen_at,
        }
        self._seal.record_evaluation(candidate_hash, freeze.discovery_snapshot_id, metrics)
        return metrics

    def report_feedback_into_search(self, candidate_hash: str, reason: str = "OOS_RESULT_USED_IN_SEARCH") -> None:
        """OOS 结果进入下一轮搜索时调用：OOS 区间立即失效（§13.4）。"""
        self._seal.invalidate_oos_window(candidate_hash, reason)


__all__ = [
    "CandidateSealStore",
    "FinalOosEvaluationService",
    "InMemoryCandidateSealStore",
]
