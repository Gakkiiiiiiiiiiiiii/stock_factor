"""Deterministic capacity assessment for formal factor research."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from stock_factor.application.tradability_policy import TradabilityPolicy, load_tradability_policy
from stock_factor.domain.tradability_artifact import (
    CapacityAssessmentArtifact,
    ExecutionCostCalibrationRef,
    TradabilityAssumptions,
)


def _array(value: Any, shape: tuple[int, int], *, default: float = np.nan) -> np.ndarray:
    if value is None:
        return np.full(shape, default, dtype=float)
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"market input shape {result.shape} does not match {shape}")
    return result


class CapacityAssessmentService:
    """Calculate a replayable capacity curve from actual amount/volume inputs.

    ``capacity`` is expressed as the capital that can be deployed before the
    selected participation cap is breached.  The curve's conservative
    ``capacity`` value is intentionally non-increasing as participation rises;
    ``max_capital`` retains the raw notional reference for operators.
    """

    def assess(
        self,
        *,
        factor_artifact_id: str,
        market_snapshot_id: str,
        factor_values: Any,
        amount: Any | None = None,
        volume: Any | None = None,
        prices: Any | None = None,
        tradability_flags: Mapping[str, Any] | None = None,
        execution_cost_calibration: ExecutionCostCalibrationRef,
        assumptions: TradabilityAssumptions | None = None,
        participation_rates: Sequence[float] = (0.01, 0.025, 0.05, 0.10, 0.20),
        policy: TradabilityPolicy | None = None,
        gross_daily_return: float = 0.0,
    ) -> CapacityAssessmentArtifact:
        values = np.asarray(factor_values, dtype=float)
        if values.ndim != 2:
            raise ValueError("factor_values must be a 2D array")
        shape = values.shape
        amount_values = _array(amount, shape)
        if amount is None and volume is not None and prices is not None:
            amount_values = _array(volume, shape) * _array(prices, shape)
        flags = tradability_flags or {}
        tradable = flags.get("tradable", flags.get("is_tradable", flags.get("can_trade")))
        tradable_values = _array(tradable, shape, default=np.nan)
        valid_amount = np.isfinite(amount_values) & (amount_values > 0)
        if tradable is not None:
            valid_amount &= tradable_values > 0
        usable_daily = np.where(valid_amount, amount_values, np.nan)
        daily_amount = np.nanmean(usable_daily, axis=0) if np.isfinite(usable_daily).any() else np.asarray([])
        average_amount = float(np.nanmean(daily_amount)) if daily_amount.size else 0.0
        policy = policy or load_tradability_policy()
        curve: list[dict[str, float]] = []
        turnover = self._turnover(values)
        cost_base = (
            execution_cost_calibration.commission_bps
            + execution_cost_calibration.spread_bps
            + execution_cost_calibration.impact_bps
            + execution_cost_calibration.stamp_tax_bps
        )
        for raw_rate in participation_rates:
            rate = float(raw_rate)
            if not np.isfinite(rate) or rate <= 0 or rate > 1:
                raise ValueError("participation rates must be in (0, 1]")
            max_capital = average_amount * rate / max(turnover, 1e-12)
            impact_multiplier = max(1.0, (rate / 0.01) ** 0.5)
            estimated_cost_bps = cost_base * impact_multiplier
            curve.append(
                {
                    "participation_rate": rate,
                    "max_capital": round(max_capital, 8),
                    "capacity": round(max_capital, 8),
                    "estimated_cost_bps": round(estimated_cost_bps, 8),
                    "net_annualized_return": round(
                        min(gross_daily_return, gross_daily_return - turnover * estimated_cost_bps / 10_000.0)
                        * policy.annualization_days,
                        12,
                    ),
                    "capacity_unit": "deployable_aum_before_participation_cap",
                }
            )
        gate_passed = bool(
            curve and average_amount >= policy.limits["min_average_daily_amount"] and np.isfinite(usable_daily).any()
        )
        return CapacityAssessmentArtifact(
            factor_artifact_id=factor_artifact_id,
            market_snapshot_id=market_snapshot_id,
            execution_cost_calibration=execution_cost_calibration,
            capacity_curve=tuple(curve),
            assumptions=assumptions or TradabilityAssumptions(),
            gate_result={
                "passed": gate_passed,
                "reasons": []
                if gate_passed
                else ["CAPACITY_INPUT_MISSING" if average_amount <= 0 else "CAPACITY_BELOW_POLICY_MINIMUM"],
                "assessment_version": "capacity_assessment_v1",
            },
            policy_version=policy.policy_version,
            policy_hash=policy.policy_hash,
        )

    @staticmethod
    def _turnover(values: np.ndarray) -> float:
        positions = np.where(np.isfinite(values), values, 0.0)
        positions = positions - np.mean(positions, axis=0, keepdims=True)
        scale = np.sum(np.abs(positions), axis=0, keepdims=True)
        positions = np.divide(positions, scale, out=np.zeros_like(positions), where=scale > 0)
        if positions.shape[1] < 2:
            return 1.0
        return float(np.mean(np.sum(np.abs(np.diff(positions, axis=1)), axis=0) / 2.0))


__all__ = ["CapacityAssessmentService", "assess_capacity"]


def assess_capacity(**kwargs: Any) -> CapacityAssessmentArtifact:
    """Functional facade retained for callers that do not need a service object."""
    return CapacityAssessmentService().assess(**kwargs)
