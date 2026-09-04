"""ResearchExperiment（收尾文档 §19）。

一次因子研究实验的唯一权威记录：Discovery → Finalist → Freeze → OOS，
状态机保证 Final OOS 只能在 FROZEN 且授权之后执行一次。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

CREATED = "CREATED"
DISCOVERY_RUNNING = "DISCOVERY_RUNNING"
DISCOVERY_COMPLETED = "DISCOVERY_COMPLETED"
FINALIST_SELECTED = "FINALIST_SELECTED"
FROZEN = "FROZEN"
OOS_AUTHORIZED = "OOS_AUTHORIZED"
OOS_EVALUATED = "OOS_EVALUATED"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
OOS_INVALIDATED = "OOS_INVALIDATED"

# 状态机（§19）：FAILED / OOS_INVALIDATED 可以从任何非终态进入。
_TRANSITIONS: dict[str, set[str]] = {
    CREATED: {DISCOVERY_RUNNING},
    DISCOVERY_RUNNING: {DISCOVERY_COMPLETED},
    DISCOVERY_COMPLETED: {FINALIST_SELECTED},
    # 无可冻结 finalist（如研究历史不足）时允许直接完结。
    FINALIST_SELECTED: {FROZEN, COMPLETED},
    FROZEN: {OOS_AUTHORIZED},
    OOS_AUTHORIZED: {OOS_EVALUATED},
    OOS_EVALUATED: {COMPLETED},
}
_TERMINAL = {COMPLETED, FAILED, OOS_INVALIDATED}

SELECTION_POLICY_VERSION = "finalist_selection_v1"
RESEARCH_CODE_VERSION = "stock_factor@local"


class ExperimentStateError(RuntimeError):
    """非法状态迁移。"""


@dataclass
class ResearchExperiment:
    symbols: list[str]
    discovery_snapshot_id: str | None = None
    final_oos_snapshot_id: str | None = None
    market_ref_hash: str | None = None
    content_ref_hash: str | None = None
    content_manifest: dict | None = None
    final_oos_dataset_ref_hash: str | None = None
    final_oos_dataset_ref: dict | None = None
    config_hash: str = ""
    candidate_budget: int = 50
    round_limit: int = 1
    finalist_count: int = 1
    selection_policy_version: str = SELECTION_POLICY_VERSION
    code_version: str = RESEARCH_CODE_VERSION
    experiment_id: str = field(default_factory=lambda: f"exp-{uuid4().hex[:12]}")
    status: str = CREATED
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    frozen_at: str | None = None
    oos_evaluated_at: str | None = None
    # Immutable admission evidence captured before formal data loading/OOS.
    readiness_evidence: dict | None = None
    readiness_evidence_hash: str | None = None
    readiness_frozen_at: str | None = None
    readiness_threshold_version: str | None = None

    def transition(self, new_status: str) -> None:
        if self.status in _TERMINAL:
            raise ExperimentStateError(f"experiment {self.experiment_id} 已处于终态 {self.status}")
        allowed = _TRANSITIONS.get(self.status, set()) | {FAILED, OOS_INVALIDATED}
        if new_status not in allowed:
            raise ExperimentStateError(f"非法状态迁移 {self.status} -> {new_status}")
        self.status = new_status
        if new_status == FROZEN:
            self.frozen_at = datetime.now(UTC).isoformat(timespec="seconds")
        if new_status == OOS_EVALUATED:
            self.oos_evaluated_at = datetime.now(UTC).isoformat(timespec="seconds")

    def authorize_oos(self) -> None:
        """§23：只有 FROZEN 的实验才能授权 Final OOS。"""
        if self.status != FROZEN:
            raise ExperimentStateError(f"实验必须先 FROZEN 才能授权 OOS（当前 {self.status}）")
        self.transition(OOS_AUTHORIZED)

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = [
    "ResearchExperiment",
    "ExperimentStateError",
    "SELECTION_POLICY_VERSION",
    "RESEARCH_CODE_VERSION",
    "CREATED",
    "DISCOVERY_RUNNING",
    "DISCOVERY_COMPLETED",
    "FINALIST_SELECTED",
    "FROZEN",
    "OOS_AUTHORIZED",
    "OOS_EVALUATED",
    "COMPLETED",
    "FAILED",
    "OOS_INVALIDATED",
]
