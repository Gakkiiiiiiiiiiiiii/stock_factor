"""Immutable, content-addressed economic implementation evidence.

The existing ``compute_capacity_proxy`` diagnostic remains a discovery-only
proxy.  These models are the formal assessment boundary used by promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ExecutionCostCalibrationRef:
    """A fixed Quant-owned execution-cost calibration identity."""

    calibration_id: str
    version: str
    checksum: str
    commission_bps: float = 0.0
    spread_bps: float = 0.0
    impact_bps: float = 0.0
    stamp_tax_bps: float = 0.0
    contract: str = "execution-cost-calibration.v1"

    def __post_init__(self) -> None:
        if not self.calibration_id or not self.version or not self.checksum:
            raise ValueError("execution cost calibration requires id, version and checksum")
        if self.contract != "execution-cost-calibration.v1":
            raise ValueError("execution cost calibration requires execution-cost-calibration.v1")
        if len(self.checksum) != 64 or any(char not in "0123456789abcdef" for char in self.checksum.lower()):
            raise ValueError("execution cost calibration checksum must be sha256")
        for name in ("commission_bps", "spread_bps", "impact_bps", "stamp_tax_bps"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExecutionCostCalibrationRef":
        return cls(
            calibration_id=str(payload.get("calibration_id") or payload.get("artifact_id") or ""),
            version=str(payload.get("version") or payload.get("execution_model_version") or ""),
            checksum=str(
                payload.get("checksum") or payload.get("calibration_checksum") or payload.get("manifest_hash") or ""
            ),
            commission_bps=float(payload.get("commission_bps", 0.0)),
            spread_bps=float(payload.get("spread_bps", 0.0)),
            impact_bps=float(payload.get("impact_bps", 0.0)),
            stamp_tax_bps=float(payload.get("stamp_tax_bps", 0.0)),
            contract=str(payload.get("contract") or "execution-cost-calibration.v1"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "calibration_id": self.calibration_id,
            "version": self.version,
            "checksum": self.checksum,
            "commission_bps": self.commission_bps,
            "spread_bps": self.spread_bps,
            "impact_bps": self.impact_bps,
            "stamp_tax_bps": self.stamp_tax_bps,
        }


@dataclass(frozen=True)
class TradabilityAssumptions:
    annualization_days: int = 250
    rebalance: str = "DAILY"
    horizon_days: int = 1
    rounding: str = "NEAREST_SHARE"
    cash_assumption: str = "FULLY_INVESTED_WITH_RESIDUAL_CASH"
    fillability: str = "NEXT_OPEN_IF_TRADABLE"
    capital: float = 1_000_000.0

    def __post_init__(self) -> None:
        if self.annualization_days != 250:
            raise ValueError("formal tradability assessment requires 250-day annualization")
        if self.horizon_days <= 0 or self.capital <= 0:
            raise ValueError("horizon_days and capital must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "annualization_days": self.annualization_days,
            "rebalance": self.rebalance,
            "horizon_days": self.horizon_days,
            "rounding": self.rounding,
            "cash_assumption": self.cash_assumption,
            "fillability": self.fillability,
            "capital": self.capital,
        }


@dataclass(frozen=True)
class CapacityAssessmentArtifact:
    factor_artifact_id: str
    market_snapshot_id: str
    execution_cost_calibration: ExecutionCostCalibrationRef
    capacity_curve: tuple[Mapping[str, Any], ...]
    assumptions: TradabilityAssumptions
    gate_result: Mapping[str, Any]
    policy_version: str = "tradability_v1"
    policy_hash: str = ""
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.factor_artifact_id or not self.market_snapshot_id:
            raise ValueError("capacity artifact requires factor and market snapshot identities")
        if self.policy_hash and (
            len(self.policy_hash) != 64 or any(char not in "0123456789abcdef" for char in self.policy_hash.lower())
        ):
            raise ValueError("capacity policy hash must be sha256")
        curve = tuple(_freeze(item) for item in self.capacity_curve)
        object.__setattr__(self, "capacity_curve", curve)
        object.__setattr__(self, "gate_result", _freeze(self.gate_result))
        payload = self.to_dict(include_id=False)
        object.__setattr__(self, "artifact_id", f"capacity-{canonical_hash(payload)[:32]}")

    @property
    def passed(self) -> bool:
        return bool(self.gate_result.get("passed", False))

    def verify(self) -> bool:
        return self.artifact_id == f"capacity-{canonical_hash(self.to_dict(include_id=False))[:32]}"

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "artifact_type": "CapacityAssessmentArtifact",
            "factor_artifact_id": self.factor_artifact_id,
            "market_snapshot_id": self.market_snapshot_id,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "execution_model_version": self.execution_cost_calibration.version,
            "execution_cost_calibration": self.execution_cost_calibration.to_dict(),
            "capacity_curve": [_jsonable(item) for item in self.capacity_curve],
            "assumptions": self.assumptions.to_dict(),
            "gate_result": _jsonable(self.gate_result),
        }
        if include_id:
            payload["artifact_id"] = self.artifact_id
        return payload


@dataclass(frozen=True)
class TradabilityAssessmentArtifact:
    factor_artifact_id: str
    market_snapshot_id: str
    execution_cost_calibration: ExecutionCostCalibrationRef
    gross_metrics: Mapping[str, Any]
    net_metrics: Mapping[str, Any]
    turnover: float
    ic_decay: Mapping[str, Any]
    holding_period_sensitivity: Mapping[str, Any]
    limit_hit_rate: float
    halt_exposure: float
    participation_rate: float
    capacity_curve: tuple[Mapping[str, Any], ...]
    neutralized_contribution: Mapping[str, Any]
    assumptions: TradabilityAssumptions
    gate_result: Mapping[str, Any]
    capacity_artifact: CapacityAssessmentArtifact | None = None
    formal_eligible: bool = True
    policy_version: str = "tradability_v1"
    policy_hash: str = ""
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.factor_artifact_id or not self.market_snapshot_id:
            raise ValueError("tradability artifact requires factor and market snapshot identities")
        if self.formal_eligible and (
            len(self.policy_hash) != 64 or any(char not in "0123456789abcdef" for char in self.policy_hash.lower())
        ):
            raise ValueError("tradability policy hash must be sha256")
        for name in ("turnover", "limit_hit_rate", "halt_exposure", "participation_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if float(self.limit_hit_rate) > 1 or float(self.halt_exposure) > 1 or float(self.participation_rate) > 1:
            raise ValueError("rate metrics must be in [0, 1]")
        for name in (
            "gross_metrics",
            "net_metrics",
            "ic_decay",
            "holding_period_sensitivity",
            "neutralized_contribution",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "capacity_curve", tuple(_freeze(item) for item in self.capacity_curve))
        object.__setattr__(self, "gate_result", _freeze(self.gate_result))
        payload = self.to_dict(include_id=False)
        object.__setattr__(self, "artifact_id", f"tradability-{canonical_hash(payload)[:32]}")

    @property
    def passed(self) -> bool:
        return bool(self.formal_eligible and self.gate_result.get("passed", False))

    @property
    def execution_model_version(self) -> str:
        return self.execution_cost_calibration.version

    def verify(self) -> bool:
        return self.artifact_id == f"tradability-{canonical_hash(self.to_dict(include_id=False))[:32]}"

    @staticmethod
    def verify_payload(payload: Mapping[str, Any]) -> bool:
        supplied = str(payload.get("artifact_id") or "")
        material = {key: value for key, value in payload.items() if key != "artifact_id"}
        return supplied == f"tradability-{canonical_hash(material)[:32]}"

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "artifact_type": "TradabilityAssessmentArtifact",
            "factor_artifact_id": self.factor_artifact_id,
            "market_snapshot_id": self.market_snapshot_id,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "execution_model_version": self.execution_cost_calibration.version,
            "execution_calibration_checksum": self.execution_cost_calibration.checksum,
            "execution_cost_calibration": self.execution_cost_calibration.to_dict(),
            "gross_metrics": _jsonable(self.gross_metrics),
            "net_metrics": _jsonable(self.net_metrics),
            "turnover": self.turnover,
            "ic_decay": _jsonable(self.ic_decay),
            "holding_period_sensitivity": _jsonable(self.holding_period_sensitivity),
            "limit_hit_rate": self.limit_hit_rate,
            "halt_exposure": self.halt_exposure,
            "participation_rate": self.participation_rate,
            "capacity_curve": [_jsonable(item) for item in self.capacity_curve],
            "neutralized_contribution": _jsonable(self.neutralized_contribution),
            "assumptions": self.assumptions.to_dict(),
            "gate_result": _jsonable(self.gate_result),
            "capacity_artifact_id": self.capacity_artifact.artifact_id if self.capacity_artifact else None,
            "formal_eligible": self.formal_eligible,
        }
        if include_id:
            payload["artifact_id"] = self.artifact_id
        return payload


__all__ = [
    "CapacityAssessmentArtifact",
    "ExecutionCostCalibrationRef",
    "TradabilityAssessmentArtifact",
    "TradabilityAssumptions",
    "canonical_hash",
]
