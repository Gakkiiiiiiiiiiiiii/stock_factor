"""FinalOosEvaluationService（设计文档 §13.4 / §78）。

独立的 Final OOS 评估应用服务：

- 输入：Frozen Candidate + Final OOS Snapshot + Evaluation Config
- 输出：OOS Evaluation，一次性使用
- 如果 OOS 结果反馈进入下一轮搜索：该 OOS 区间立即失效，自动降级为
  Discovery 数据（后续评估必须报告 OOS_WINDOW_INVALIDATED）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from stock_factor.application.finalist_selection import MAX_FINALIST_COUNT
from stock_factor.application.oos.authorize import OosAuthorizationError, OosAuthorizationStore
from stock_factor.application.oos_run_service import OosRunService
from stock_factor.engine.fitness import evaluate_factor_range
from stock_factor.engine.oos_seal import (
    CandidateFreeze,
    CandidateUnfrozenError,
    OosWindowInvalidatedError,
)
from stock_factor.engine.research_split import FactorResearchSplit
from stock_factor.ports.oos_run_repository import OosIdentityError


class FinalOosAuthorizationError(RuntimeError):
    """收尾文档 §23：实验未处于 OOS_AUTHORIZED 时禁止读取/评估 Final OOS。

    详细修改方案 P0-2：数据库级授权异常（缺失/已消费/已失效）会被转为
    本异常抛出，调用方按同一策略处理。
    """


@dataclass(frozen=True)
class FinalOosCandidateInput:
    """P0 F-01：cohort 内单个 finalist 的最小评估输入。"""

    candidate_hash: str
    values: np.ndarray | None = None
    closes: np.ndarray | None = None
    loader: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None
    input_identity: str | None = None

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        if self.loader is not None:
            return self.loader()
        if self.values is None or self.closes is None:
            raise ValueError("Final OOS candidate input requires values/closes or a loader")
        return self.values, self.closes


@dataclass(frozen=True)
class FinalOosCohortResult:
    """Sealed per-call OOS evidence; never stored as service-global state."""

    results: list[dict[str, Any]]
    run_id: str | None
    cohort_artifact_hash: str | None


class FinalOosDataProvider:
    """收尾文档 §21/§23：Final OOS 数据只在实验授权后才允许加载。

    Discovery / Candidate Search 进程中 Final OOS 数据不得进入内存；
    只有完成 Finalist Selection + Candidate Freeze 并授权后才允许 load。

    OOS Warmup（§22）：返回的 dataset 覆盖 warmup + Final OOS 窗口
    （即完整 values/close 面板），但统计只允许 [final_oos_start, final_oos_end)。
    """

    def load(self, experiment, dataset):
        status = getattr(experiment, "status", None)
        if status != "OOS_AUTHORIZED":
            raise FinalOosAuthorizationError(f"实验状态 {status} 不允许加载 Final OOS 数据（需 OOS_AUTHORIZED）")
        return dataset


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

    def __init__(
        self,
        seal_store: CandidateSealStore,
        authorizations: OosAuthorizationStore | None = None,
        run_repository=None,
    ) -> None:
        self._seal = seal_store
        self._authorizations = authorizations
        self._run_service = OosRunService(run_repository) if run_repository is not None else None

    def freeze_candidate(self, freeze: CandidateFreeze) -> CandidateFreeze:
        """进入 Final OOS 前冻结候选（§13.3）。"""
        self._seal.save_freeze(freeze)
        return freeze

    def register_authorization(
        self,
        experiment_id: str,
        final_oos_snapshot_id: str = "",
        candidate_set_hash: str = "",
        *,
        dataset_ref_hash: str | None = None,
        market_snapshot_id: str | None = None,
        content_ref_hash: str | None = None,
    ) -> dict | None:
        """P0-2：实验授权 OOS 时同步落库授权记录（无持久化存储时为 no-op）。"""
        if self._run_service is not None:
            authorization = self._run_service.authorize(
                experiment_id,
                candidate_set_hash,
                dataset_ref_hash or final_oos_snapshot_id,
                market_snapshot_id or final_oos_snapshot_id,
                content_ref_hash=content_ref_hash,
            )
            return {
                "authorization_id": authorization.authorization_id,
                "experiment_id": authorization.experiment_id,
                "status": authorization.status.value,
                "identity_hash": authorization.identity_hash,
            }
        if self._authorizations is None:
            return None
        return self._authorizations.authorize(experiment_id, final_oos_snapshot_id, candidate_set_hash)

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
        metrics = evaluate_factor_range(values, closes, split.final_oos_start, split.final_oos_end, horizon=horizon)
        metrics = {
            **metrics,
            "final_oos_snapshot_id": freeze.final_oos_snapshot_id,
            "discovery_snapshot_id": freeze.discovery_snapshot_id,
            "candidate_frozen_at": freeze.candidate_frozen_at,
        }
        self._seal.record_evaluation(candidate_hash, freeze.discovery_snapshot_id, metrics)
        return metrics

    def evaluate_raw(
        self,
        candidate_hash: str,
        values: np.ndarray,
        closes: np.ndarray,
        split: FactorResearchSplit,
        horizon: int,
    ) -> dict[str, Any]:
        """Evaluate without consulting the legacy candidate/discovery cache."""
        freeze = self._seal.get_freeze(candidate_hash)
        if freeze is None:
            raise CandidateUnfrozenError(f"candidate {candidate_hash} 未冻结，禁止进行 Final OOS 评估")
        if self._seal.oos_window_status(candidate_hash) == "INVALIDATED":
            raise OosWindowInvalidatedError(f"candidate {candidate_hash} 的 Final OOS 区间已失效")
        return {
            **evaluate_factor_range(values, closes, split.final_oos_start, split.final_oos_end, horizon=horizon),
            "final_oos_snapshot_id": freeze.final_oos_snapshot_id,
            "discovery_snapshot_id": freeze.discovery_snapshot_id,
            "candidate_frozen_at": freeze.candidate_frozen_at,
        }

    def report_feedback_into_search(self, candidate_hash: str, reason: str = "OOS_RESULT_USED_IN_SEARCH") -> None:
        """OOS 结果进入下一轮搜索时调用：OOS 区间立即失效（§13.4 / 收尾文档 §24）。"""
        self._seal.invalidate_oos_window(candidate_hash, reason)

    def evaluate_cohort_with_evidence(
        self,
        experiment,
        candidates: list[FinalOosCandidateInput],
        split: FactorResearchSplit,
        horizon: int,
        *,
        run_id: str | None = None,
        owner_id: str = "oos-worker",
        lease_seconds: int = 900,
        evaluator_version: str = "final-oos-v1",
    ) -> FinalOosCohortResult:
        """P0 F-01：一个 Experiment → 一个 Finalist Cohort → 一次授权消费。

        语义（不得改成每 candidate 一份 authorization）：
        1. 验证 experiment.status == OOS_AUTHORIZED；
        2. 验证 1 <= cohort size <= MAX_FINALIST_COUNT；
        3. 所有 candidate 必须已 Freeze，且 Freeze 指向同一个预注册 Final OOS Snapshot；
        4. resumable repository path only consumes after cohort artifact sealing;
        5. cohort 全部 candidate 评估成功后才 transition("OOS_EVALUATED")；
        6. retry 语义：deterministic partial resume —— completed checkpoints
           直接复用，未完成的 candidate 可继续评估。
        """
        if getattr(experiment, "status", None) != "OOS_AUTHORIZED":
            raise FinalOosAuthorizationError(
                f"实验 {getattr(experiment, 'experiment_id', '?')} 未授权 Final OOS（当前 {getattr(experiment, 'status', '?')}）"
            )
        # 授权记录缺失快速失败（在 consume 之前，语义等同数据库级拒绝）。
        if self._authorizations is not None:
            record = self._authorizations.get(getattr(experiment, "experiment_id", ""))
            if record is None:
                raise FinalOosAuthorizationError(
                    f"实验 {getattr(experiment, 'experiment_id', '?')} 没有 Final OOS 授权记录"
                )
        if not 1 <= len(candidates) <= MAX_FINALIST_COUNT:
            raise ValueError(f"Final OOS cohort 大小必须满足 1 <= size <= {MAX_FINALIST_COUNT}，实际 {len(candidates)}")
        if self._run_service is not None:
            authorization_id = f"oosa-{getattr(experiment, 'experiment_id', '')}"
            payload = [
                {
                    "candidate_id": candidate.candidate_hash,
                    "final_oos_start": split.final_oos_start,
                    "final_oos_end": split.final_oos_end,
                    "horizon": horizon,
                    **({"input_identity": candidate.input_identity} if candidate.input_identity is not None else {}),
                    **(
                        {"loader": candidate.loader}
                        if candidate.loader is not None
                        else {"values": candidate.values, "closes": candidate.closes}
                    ),
                }
                for candidate in candidates
            ]

            def preflight() -> None:
                for candidate in candidates:
                    if candidate.loader is not None and not candidate.input_identity:
                        raise OosIdentityError(
                            f"candidate {candidate.candidate_hash} lazy input requires a frozen input identity"
                        )
                    if candidate.loader is None and (candidate.values is None or candidate.closes is None):
                        raise OosIdentityError(f"candidate {candidate.candidate_hash} lacks immutable OOS input")
                freezes = []
                for candidate in candidates:
                    freeze = self._seal.get_freeze(candidate.candidate_hash)
                    if freeze is None:
                        raise CandidateUnfrozenError(
                            f"candidate {candidate.candidate_hash} 未冻结，禁止进行 Final OOS 评估"
                        )
                    freezes.append(freeze)
                snapshot_ids = {freeze.final_oos_snapshot_id for freeze in freezes}
                if len(snapshot_ids) != 1:
                    raise OosIdentityError(
                        f"cohort 内 Freeze 必须指向同一个预注册 Final OOS Snapshot，实际 {sorted(snapshot_ids)}"
                    )

            def evaluator(item):
                values, closes = item["loader"]() if "loader" in item else (item["values"], item["closes"])
                return self.evaluate_raw(item["candidate_id"], values, closes, split, horizon)

            result = self._run_service.evaluate_cohort(
                authorization_id,
                payload,
                evaluator,
                run_id=run_id,
                owner_id=owner_id,
                lease_seconds=lease_seconds,
                evaluator_version=evaluator_version,
                preflight=preflight,
            )
            if experiment.status == "OOS_AUTHORIZED":
                experiment.transition("OOS_EVALUATED")
            return FinalOosCohortResult(
                results=result["results"],
                run_id=result["run_id"],
                cohort_artifact_hash=result["cohort_artifact_hash"],
            )
        # Legacy compatibility path keeps its eager validation and cache semantics.
        freezes: list[CandidateFreeze] = []
        for candidate in candidates:
            freeze = self._seal.get_freeze(candidate.candidate_hash)
            if freeze is None:
                raise CandidateUnfrozenError(f"candidate {candidate.candidate_hash} 未冻结，禁止进行 Final OOS 评估")
            freezes.append(freeze)
        snapshot_ids = {freeze.final_oos_snapshot_id for freeze in freezes}
        if len(snapshot_ids) != 1:
            raise FinalOosAuthorizationError(
                f"cohort 内 Freeze 必须指向同一个预注册 Final OOS Snapshot，实际 {sorted(snapshot_ids)}"
            )
        # 一次性消费：整个 cohort 共享同一次 authorization consume。
        if self._authorizations is not None:
            try:
                self._authorizations.consume(getattr(experiment, "experiment_id", ""))
            except OosAuthorizationError as exc:
                raise FinalOosAuthorizationError(str(exc)) from exc
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            # evaluate() 内部对已记录结果确定性重放（partial resume），
            # 中途失败直接抛出：experiment 保持 OOS_AUTHORIZED，不产生假成功状态。
            results.append(self.evaluate(candidate.candidate_hash, candidate.values, candidate.closes, split, horizon))
        if experiment.status == "OOS_AUTHORIZED":
            experiment.transition("OOS_EVALUATED")
        return FinalOosCohortResult(results=results, run_id=None, cohort_artifact_hash=None)

    def evaluate_cohort_for_experiment(
        self,
        experiment,
        candidates: list[FinalOosCandidateInput],
        split: FactorResearchSplit,
        horizon: int,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Compatibility wrapper returning only result metrics."""
        return self.evaluate_cohort_with_evidence(experiment, candidates, split, horizon, **kwargs).results

    def evaluate_for_experiment(
        self,
        experiment,
        candidate_hash: str,
        values: np.ndarray,
        closes: np.ndarray,
        split: FactorResearchSplit,
        horizon: int,
    ) -> dict[str, Any]:
        """兼容入口：只允许封装单 candidate cohort，复用同一套消费语义。

        不得独立 consume()；多 finalist 场景必须改用 evaluate_cohort_for_experiment。
        """
        cohort = [FinalOosCandidateInput(candidate_hash=candidate_hash, values=values, closes=closes)]
        return self.evaluate_cohort_for_experiment(experiment, cohort, split, horizon)[0]


__all__ = [
    "CandidateSealStore",
    "FinalOosCandidateInput",
    "FinalOosCohortResult",
    "FinalOosEvaluationService",
    "FinalOosDataProvider",
    "FinalOosAuthorizationError",
    "InMemoryCandidateSealStore",
]
