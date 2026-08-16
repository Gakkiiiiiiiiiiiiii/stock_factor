"""FactorSet 正式版本化（详细修改方案 §11）。

Agent 使用 factor_set_id 而不是临时查询"当前所有 active factor"。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

FACTOR_SET_STATUS = ("DRAFT", "ACTIVE", "SUPERSEDED", "RETIRED")


@dataclass(frozen=True)
class FactorSet:
    factor_ids: tuple[str, ...]
    factor_versions: tuple[int, ...]
    weights: tuple[float, ...]
    research_experiment_ids: tuple[str, ...] = ()
    promotion_policy_version: str = "promotion_gate_v2"
    valid_from: str | None = None
    valid_to: str | None = None
    status: str = "ACTIVE"
    code_sha: str | None = None
    config_hash: str | None = None
    factor_set_id: str = field(init=False, default="")
    factor_set_version: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if len(self.factor_ids) != len(self.factor_versions) or len(self.factor_ids) != len(self.weights):
            raise ValueError("factor_ids / factor_versions / weights 长度必须一致")
        material = {
            "factor_ids": list(self.factor_ids),
            "factor_versions": list(self.factor_versions),
            "weights": list(self.weights),
            "research_experiment_ids": list(self.research_experiment_ids),
            "promotion_policy_version": self.promotion_policy_version,
        }
        digest = hashlib.sha256(
            json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "factor_set_id", f"fs-{digest[:16]}")
        object.__setattr__(self, "factor_set_version", f"factor-set-{digest[:12]}")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["created_at"] = None
        return payload


def factor_set_from_factors(
    factors: list[dict],
    research_experiment_ids: tuple[str, ...] = (),
    promotion_policy_version: str = "promotion_gate_v2",
) -> FactorSet:
    """从因子列表（含 factor_id/version）构建等权 FactorSet。"""
    ordered = sorted(factors, key=lambda item: str(item.get("factor_id")))
    count = max(len(ordered), 1)
    return FactorSet(
        factor_ids=tuple(str(item.get("factor_id")) for item in ordered),
        factor_versions=tuple(int(item.get("version", 1)) for item in ordered),
        weights=tuple(round(1.0 / count, 8) for _ in ordered),
        research_experiment_ids=research_experiment_ids,
        promotion_policy_version=promotion_policy_version,
        valid_from=datetime.now(UTC).isoformat(timespec="seconds"),
    )


__all__ = ["FACTOR_SET_STATUS", "FactorSet", "factor_set_from_factors"]
