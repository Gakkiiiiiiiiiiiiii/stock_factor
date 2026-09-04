"""Formal tradability, fillability and implementation-cost assessment."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from stock_factor.application.capacity_assessment import CapacityAssessmentService
from stock_factor.application.tradability_policy import TradabilityPolicy, load_tradability_policy
from stock_factor.domain.tradability_artifact import (
    ExecutionCostCalibrationRef,
    TradabilityAssessmentArtifact,
    TradabilityAssumptions,
)
from stock_factor.engine.diagnostics import compute_ic_decay


def _array(value: Any, shape: tuple[int, int], *, default: float = np.nan) -> np.ndarray:
    if value is None:
        return np.full(shape, default, dtype=float)
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"market input shape {result.shape} does not match {shape}")
    return result


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


class TradabilityAssessmentService:
    def __init__(
        self,
        capacity_service: CapacityAssessmentService | None = None,
        policy: TradabilityPolicy | None = None,
        expected_execution_cost_calibration: ExecutionCostCalibrationRef | Mapping[str, Any] | None = None,
    ) -> None:
        self._capacity = capacity_service or CapacityAssessmentService()
        self._policy = policy or load_tradability_policy()
        self._expected_calibration = expected_execution_cost_calibration

    def assess(
        self,
        *,
        factor_artifact_id: str,
        market_snapshot_id: str,
        factor_values: Any,
        closes: Any,
        amount: Any | None = None,
        volume: Any | None = None,
        execution_cost_calibration: ExecutionCostCalibrationRef | Mapping[str, Any] | None = None,
        tradability_flags: Mapping[str, Any] | None = None,
        assumptions: TradabilityAssumptions | None = None,
        holding_periods: Sequence[int] = (1, 5, 10),
        participation_rates: Sequence[float] = (0.01, 0.025, 0.05, 0.10, 0.20),
        neutralization: Mapping[str, Any] | None = None,
        market_snapshot_ref: Mapping[str, Any] | Any | None = None,
        expected_execution_cost_calibration: ExecutionCostCalibrationRef | Mapping[str, Any] | None = None,
        formal: bool = True,
    ) -> TradabilityAssessmentArtifact:
        values = np.asarray(factor_values, dtype=float)
        prices = np.asarray(closes, dtype=float)
        if values.ndim != 2 or prices.shape != values.shape:
            raise ValueError("factor_values and closes must be same-shaped 2D arrays")
        shape = values.shape
        flags = tradability_flags or {}
        calibration = self._calibration(execution_cost_calibration)
        expected_payload = (
            expected_execution_cost_calibration
            if expected_execution_cost_calibration is not None
            else self._expected_calibration
        )
        expected_calibration = self._calibration(expected_payload)
        assumptions = assumptions or TradabilityAssumptions(horizon_days=1)
        valid = np.isfinite(values) & np.isfinite(prices) & (prices > 0)
        tradable = flags.get("tradable", flags.get("is_tradable", flags.get("can_trade")))
        if tradable is None:
            valid &= False
            tradable_values = np.full(shape, np.nan)
        else:
            tradable_values = _array(tradable, shape)
            valid &= tradable_values > 0
        halted = flags.get("halted", flags.get("is_halted", flags.get("suspended", flags.get("is_suspended"))))
        halted_values = _array(halted, shape, default=0.0)
        limit = flags.get("limit", flags.get("limit_hit"))
        if limit is None:
            limit = np.maximum(
                _array(flags.get("limit_up"), shape, default=0.0),
                _array(flags.get("limit_down"), shape, default=0.0),
            )
        limit_values = _array(limit, shape, default=0.0)
        returns = np.full(shape, np.nan)
        returns[:, :-1] = np.divide(prices[:, 1:], prices[:, :-1]) - 1.0
        forward_valid = valid & np.isfinite(returns)
        direction = np.sign(np.where(np.isfinite(values), values, 0.0))
        gross_series = np.where(forward_valid, direction * returns, np.nan)
        daily_counts = np.sum(np.isfinite(gross_series), axis=0)
        daily_totals = np.nansum(gross_series, axis=0)
        gross_daily = np.divide(
            daily_totals,
            daily_counts,
            out=np.full_like(daily_totals, np.nan),
            where=daily_counts > 0,
        )
        gross_daily = gross_daily[np.isfinite(gross_daily)]
        gross_return = float(np.nanmean(gross_daily)) if gross_daily.size else 0.0
        turnover = self._turnover(values, valid)
        cost_bps = (
            calibration.commission_bps + calibration.spread_bps + calibration.impact_bps + calibration.stamp_tax_bps
        )
        daily_cost = turnover * cost_bps / 10_000.0
        net_return = min(gross_return, gross_return - daily_cost)
        participation_amount = amount
        if participation_amount is None and volume is not None:
            participation_amount = _array(volume, shape) * prices
        participation = self._participation(values, participation_amount, valid, turnover)
        limit_hit_rate = float(np.mean(limit_values[forward_valid] > 0)) if forward_valid.any() else 1.0
        halt_exposure = float(np.mean(halted_values[valid] > 0)) if valid.any() else 1.0
        capacity = self._capacity.assess(
            factor_artifact_id=factor_artifact_id,
            market_snapshot_id=market_snapshot_id,
            factor_values=values,
            amount=amount,
            volume=volume,
            prices=prices,
            tradability_flags=flags,
            execution_cost_calibration=calibration,
            assumptions=assumptions,
            participation_rates=participation_rates,
            policy=self._policy,
            gross_daily_return=gross_return,
        )
        missing = []
        if execution_cost_calibration is None:
            missing.append("EXECUTION_COST_CALIBRATION_MISSING")
        if formal and expected_payload is None:
            missing.append("EXPECTED_EXECUTION_COST_CALIBRATION_MISSING")
        if expected_payload is not None and calibration != expected_calibration:
            missing.append("EXECUTION_COST_CALIBRATION_MISMATCH")
        if tradable is None:
            missing.append("TRADABILITY_FLAGS_MISSING")
        if amount is None and volume is None:
            missing.append("VOLUME_OR_AMOUNT_MISSING")
        reasons = list(missing)
        if limit_hit_rate > self._policy.limits["max_limit_hit_rate"]:
            reasons.append("LIMIT_FILLABILITY_FAILED")
        if halt_exposure > self._policy.limits["max_halt_exposure"]:
            reasons.append("HALT_EXPOSURE_FAILED")
        if turnover > self._policy.limits["max_turnover"]:
            reasons.append("HIGH_TURNOVER_FAILED")
        if participation > self._policy.limits["max_participation_rate"]:
            reasons.append("PARTICIPATION_RATE_FAILED")
        if formal:
            if market_snapshot_ref is None:
                missing.append("MARKET_SNAPSHOT_REF_MISSING")
            else:
                ref_snapshot_id = getattr(market_snapshot_ref, "market_snapshot_id", None)
                ref_hash = getattr(market_snapshot_ref, "ref_hash", None)
                contract = getattr(market_snapshot_ref, "contract", None)
                if isinstance(market_snapshot_ref, Mapping):
                    ref_snapshot_id = market_snapshot_ref.get("market_snapshot_id")
                    ref_hash = market_snapshot_ref.get("ref_hash")
                    contract = market_snapshot_ref.get("contract")
                if ref_snapshot_id != market_snapshot_id or not ref_hash or contract != "market-snapshot.v1":
                    missing.append("MARKET_SNAPSHOT_MISMATCH")
        reasons = list(missing) + [reason for reason in reasons if reason not in missing]
        if not capacity.passed:
            reasons.append("CAPACITY_FAILED")
        gross_metrics = _finite_or_none(
            {
                "mean_daily_return": round(gross_return, 12),
                "annualized_return": round(gross_return * assumptions.annualization_days, 12),
                "metric_scope": "FORMAL_RESEARCH" if formal else "EXPLORATORY_REPORT",
            }
        )
        net_metrics = _finite_or_none(
            {
                "mean_daily_return": round(net_return, 12),
                "annualized_return": round(net_return * assumptions.annualization_days, 12),
                "cost_drag_bps_per_day": round(daily_cost * 10_000.0, 8),
                "metric_scope": "FORMAL_RESEARCH" if formal else "EXPLORATORY_REPORT",
            }
        )
        if gross_metrics["mean_daily_return"] is None or net_metrics["mean_daily_return"] is None:
            reasons.append("NONFINITE_METRIC")
        if net_metrics["annualized_return"] < self._policy.limits["min_net_annualized_return"]:
            reasons.append("NET_RETURN_BELOW_POLICY_MINIMUM")
        gate_passed = not reasons
        # Exploratory assessments are useful diagnostics but can never be
        # evidence for formal promotion, even when all local inputs exist.
        eligible = bool(formal and not missing)
        return TradabilityAssessmentArtifact(
            factor_artifact_id=factor_artifact_id,
            market_snapshot_id=market_snapshot_id,
            execution_cost_calibration=calibration,
            gross_metrics=gross_metrics,
            net_metrics=net_metrics,
            turnover=round(turnover, 12),
            ic_decay={str(key): value for key, value in compute_ic_decay(values, prices, max_horizon=5).items()},
            holding_period_sensitivity=self._holding_sensitivity(values, prices, valid, holding_periods),
            limit_hit_rate=round(limit_hit_rate, 12),
            halt_exposure=round(halt_exposure, 12),
            participation_rate=round(participation, 12),
            capacity_curve=capacity.capacity_curve,
            neutralized_contribution=dict(neutralization or {"available": False, "reason": "not_supplied"}),
            assumptions=assumptions,
            gate_result={
                "passed": bool(gate_passed and eligible),
                "reasons": reasons,
                "assessment_version": "tradability_assessment_v1",
                "report_only": not formal,
            },
            capacity_artifact=capacity,
            formal_eligible=bool(eligible),
            policy_version=self._policy.policy_version,
            policy_hash=self._policy.policy_hash,
        )

    @staticmethod
    def _calibration(value: ExecutionCostCalibrationRef | Mapping[str, Any] | None) -> ExecutionCostCalibrationRef:
        if isinstance(value, ExecutionCostCalibrationRef):
            return value
        if isinstance(value, Mapping):
            return ExecutionCostCalibrationRef.from_payload(value)
        # Keep the object serializable even when exploratory mode has no
        # calibration; the resulting artifact remains ineligible/report-only.
        return ExecutionCostCalibrationRef("missing", "missing", "0" * 64)

    @staticmethod
    def _turnover(values: np.ndarray, valid: np.ndarray) -> float:
        positions = np.where(valid, values, 0.0)
        positions -= np.mean(positions, axis=0, keepdims=True)
        scale = np.sum(np.abs(positions), axis=0, keepdims=True)
        positions = np.divide(positions, scale, out=np.zeros_like(positions), where=scale > 0)
        if positions.shape[1] < 2:
            return 1.0
        return float(np.mean(np.sum(np.abs(np.diff(positions, axis=1)), axis=0) / 2.0))

    @classmethod
    def _participation(cls, values: np.ndarray, amount: Any | None, valid: np.ndarray, turnover: float) -> float:
        if amount is None:
            return 0.0
        amounts = _array(amount, values.shape)
        positions = np.where(valid, values, 0.0)
        scale = np.sum(np.abs(positions), axis=0, keepdims=True)
        weights = np.divide(np.abs(positions), scale, out=np.zeros_like(positions), where=scale > 0)
        traded = np.sum(weights * np.nan_to_num(amounts), axis=0)
        available = np.nansum(np.where(valid, amounts, np.nan), axis=0)
        rates = np.divide(traded, available, out=np.zeros_like(traded), where=available > 0)
        # Only traded weight contributes to participation; the current
        # portfolio allocation itself is not daily market volume.
        return float(np.nanmean(rates) * turnover) if rates.size else 0.0

    @staticmethod
    def _holding_sensitivity(
        values: np.ndarray, prices: np.ndarray, valid: np.ndarray, horizons: Sequence[int]
    ) -> dict[str, float]:
        output: dict[str, float] = {}
        direction = np.sign(np.where(np.isfinite(values), values, 0.0))
        for raw_horizon in horizons:
            horizon = int(raw_horizon)
            if horizon <= 0 or horizon >= prices.shape[1]:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                forward = prices[:, horizon:] / prices[:, :-horizon] - 1.0
            valid_window = valid[:, :-horizon] & np.isfinite(forward)
            series = np.where(valid_window, direction[:, :-horizon] * forward, np.nan)
            output[str(horizon)] = round(float(np.nanmean(series)) if np.isfinite(series).any() else 0.0, 12)
        return output


__all__ = ["TradabilityAssessmentService", "assess_tradability"]


def assess_tradability(**kwargs: Any) -> TradabilityAssessmentArtifact:
    """Functional facade retained for simple batch/replay callers."""
    return TradabilityAssessmentService().assess(**kwargs)
