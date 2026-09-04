"""Versioned policy and content hash for the economic implementation gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stock_factor.config.schema import CONFIG_ROOT, load_config


@dataclass(frozen=True)
class TradabilityPolicy:
    policy_version: str
    assessment_version: str
    report_only: bool
    annualization_days: int
    limits: dict[str, float]
    policy_hash: str

    def __post_init__(self) -> None:
        if not self.policy_version or not self.assessment_version:
            raise ValueError("tradability policy requires version identifiers")
        if self.annualization_days != 250:
            raise ValueError("tradability policy requires 250-day annualization")
        required = {
            "min_average_daily_amount",
            "max_limit_hit_rate",
            "max_halt_exposure",
            "max_turnover",
            "max_participation_rate",
            "min_net_annualized_return",
        }
        if not required.issubset(self.limits):
            raise ValueError("tradability policy is missing required limits")
        if len(self.policy_hash) != 64 or any(char not in "0123456789abcdef" for char in self.policy_hash.lower()):
            raise ValueError("tradability policy hash must be sha256")
        if float(self.limits["min_average_daily_amount"]) <= 0:
            raise ValueError("minimum average daily amount must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "assessment_version": self.assessment_version,
            "report_only": self.report_only,
            "annualization_days": self.annualization_days,
            "limits": dict(self.limits),
            "policy_hash": self.policy_hash,
        }


def load_tradability_policy(path: str | Path | None = None) -> TradabilityPolicy:
    policy_path = Path(path) if path is not None else CONFIG_ROOT / "promotion" / "tradability_v1.yaml"
    loaded = load_config(policy_path)
    payload = loaded.payload
    limits = {str(key): float(value) for key, value in (payload.get("limits") or {}).items()}
    return TradabilityPolicy(
        policy_version=str(payload.get("policy_version") or ""),
        assessment_version=str(payload.get("assessment_version") or ""),
        report_only=bool(payload.get("report_only", False)),
        annualization_days=int(payload.get("annualization_days", 0)),
        limits=limits,
        policy_hash=loaded.content_hash.removeprefix("sha256:"),
    )


__all__ = ["TradabilityPolicy", "load_tradability_policy"]
