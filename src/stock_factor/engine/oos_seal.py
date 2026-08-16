"""Candidate Freeze 与快照引用（设计文档 §13.3 / §86）。

进入 Final OOS 前必须冻结候选：冻结后不可修改 Formula、不可重新搜索参数、
不可根据 OOS 结果重新 Mutation。
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

DSL_VERSION = "factor-dsl.v1"


@dataclass(frozen=True)
class MarketSnapshotRef:
    """Experiment 的分窗口快照引用（§86）。

    discovery 与 final OOS 使用从数据快照内容派生的独立 ID，
    保证审计时可以区分候选搜索与一次性 OOS 评估各自看到的数据区域。
    """

    data_snapshot_id: str
    discovery_snapshot_id: str
    final_oos_snapshot_id: str


def derive_snapshot_refs(data_snapshot_id: str, discovery_range: tuple[int, int], final_oos_range: tuple[int, int]) -> MarketSnapshotRef:
    def _suffix(label: str, window: tuple[int, int]) -> str:
        material = f"{data_snapshot_id}:{label}:{window[0]}:{window[1]}"
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    return MarketSnapshotRef(
        data_snapshot_id=data_snapshot_id,
        discovery_snapshot_id=f"mds-discovery-{_suffix('discovery', discovery_range)}",
        final_oos_snapshot_id=f"mds-final-oos-{_suffix('final-oos', final_oos_range)}",
    )


def feature_set_version(panel_keys: list[str]) -> str:
    digest = hashlib.sha256("|".join(sorted(panel_keys)).encode()).hexdigest()[:12]
    return f"features-{digest}"


@dataclass(frozen=True)
class CandidateFreeze:
    """§13.3 Candidate Freeze 记录（收尾文档 §20 扩展）。"""

    candidate_hash: str
    formula: list[str]
    dsl_version: str
    feature_set_version: str
    discovery_snapshot_id: str
    final_oos_snapshot_id: str
    candidate_frozen_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    # 收尾文档 §20 扩展字段
    experiment_id: str | None = None
    discovery_config_hash: str | None = None
    selection_policy_version: str | None = None
    selection_rank: int | None = None
    research_code_version: str | None = None
    selected_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class CandidateUnfrozenError(RuntimeError):
    """在 Final OOS 评估前候选未被冻结。"""


class OosWindowInvalidatedError(RuntimeError):
    """Final OOS 区间已因反馈进入搜索而失效（降级为 Discovery 数据）。"""


__all__ = [
    "DSL_VERSION",
    "CandidateFreeze",
    "CandidateUnfrozenError",
    "MarketSnapshotRef",
    "OosWindowInvalidatedError",
    "derive_snapshot_refs",
    "feature_set_version",
]
