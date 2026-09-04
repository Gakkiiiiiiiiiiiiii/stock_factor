from __future__ import annotations

import numpy as np
import pytest

from stock_factor.api.main import MiningJobRequest
from stock_factor.application.tradability_assessment import TradabilityAssessmentService
from stock_factor.domain.tradability_artifact import ExecutionCostCalibrationRef
from stock_factor.engine.promotion_gate import evaluate_promotion_gate


def _assessment(**overrides):
    values = np.array([[1.0, 2.0, 1.0, 2.0, 1.0], [2.0, 1.0, 2.0, 1.0, 2.0], [0.5, 0.4, 0.6, 0.3, 0.7]])
    prices = np.array([[10.0, 10.2, 10.1, 10.3, 10.4], [10.0, 9.8, 9.9, 9.7, 9.6], [5.0, 5.1, 5.0, 5.2, 5.1]])
    amount = np.full(values.shape, 1_000_000.0)
    flags = {"tradable": np.ones(values.shape)}
    kwargs = {
        "factor_artifact_id": "factor-1",
        "market_snapshot_id": "market-1",
        "factor_values": values,
        "closes": prices,
        "amount": amount,
        "volume": amount / prices,
        "execution_cost_calibration": ExecutionCostCalibrationRef("quant-cost", "v1", "0" * 64),
        "expected_execution_cost_calibration": ExecutionCostCalibrationRef("quant-cost", "v1", "0" * 64),
        "tradability_flags": flags,
        "market_snapshot_ref": {
            "market_snapshot_id": "market-1",
            "ref_hash": "0" * 64,
            "contract": "market-snapshot.v1",
        },
    }
    kwargs.update(overrides)
    return TradabilityAssessmentService().assess(**kwargs)


def test_formal_artifact_is_content_addressed_and_cost_versioned():
    first = _assessment()
    second = _assessment(execution_cost_calibration=ExecutionCostCalibrationRef("quant-cost", "v2", "1" * 64))
    assert first.artifact_id != second.artifact_id
    assert first.to_dict()["capacity_artifact_id"] == first.capacity_artifact.artifact_id
    assert first.formal_eligible is True


def test_missing_formal_inputs_fail_closed_and_exploratory_is_report_only():
    missing = _assessment(
        execution_cost_calibration=None,
        expected_execution_cost_calibration=ExecutionCostCalibrationRef("quant-cost", "v1", "0" * 64),
        tradability_flags={},
    )
    assert missing.passed is False
    assert missing.formal_eligible is False
    report = _assessment(formal=False, execution_cost_calibration=None, tradability_flags={})
    assert report.passed is False
    assert report.to_dict()["gate_result"]["report_only"] is True


def test_cost_calibration_contract_and_checksum_are_strict():
    with pytest.raises(ValueError, match="execution-cost-calibration.v1"):
        ExecutionCostCalibrationRef("id", "v1", "0" * 64, contract="local-cost.v1")
    with pytest.raises(ValueError, match="checksum"):
        ExecutionCostCalibrationRef("id", "v1", "not-a-sha256")


def test_public_mining_request_cannot_supply_trusted_expected_calibration():
    assert "expected_execution_cost_calibration" not in MiningJobRequest.model_fields


def test_mismatched_expected_calibration_fails_closed():
    assessment = _assessment(
        expected_execution_cost_calibration=ExecutionCostCalibrationRef("quant-cost", "v2", "1" * 64)
    )
    assert assessment.formal_eligible is False
    assert "EXECUTION_COST_CALIBRATION_MISMATCH" in assessment.gate_result["reasons"]


def test_limits_and_halts_are_untradeable():
    flags = {
        "tradable": np.ones((3, 5)),
        "halted": np.zeros((3, 5)),
        "limit_hit": np.zeros((3, 5)),
    }
    flags["halted"][0, 1] = 1
    flags["limit_hit"][1, 1] = 1
    assessment = _assessment(tradability_flags=flags)
    assert assessment.halt_exposure > 0
    assert assessment.limit_hit_rate > 0
    assert assessment.passed is False


def test_low_liquidity_fails_versioned_capacity_policy():
    assessment = _assessment(amount=np.full((3, 5), 1.0), volume=np.full((3, 5), 1.0))
    assert assessment.capacity_artifact.gate_result["passed"] is False
    assert "CAPACITY_BELOW_POLICY_MINIMUM" in assessment.capacity_artifact.gate_result["reasons"]
    assert assessment.to_dict()["policy_version"] == "tradability_v1"
    assert len(assessment.to_dict()["policy_hash"]) == 64


def test_net_metrics_cannot_exceed_gross_and_capacity_is_decreasing():
    assessment = _assessment(
        execution_cost_calibration=ExecutionCostCalibrationRef(
            "quant-cost", "v1", "0" * 64, commission_bps=2.0, spread_bps=3.0, impact_bps=4.0
        )
    )
    assert assessment.net_metrics["mean_daily_return"] <= assessment.gross_metrics["mean_daily_return"]
    curve = assessment.capacity_curve
    assert [item["max_capital"] for item in curve] == sorted(item["max_capital"] for item in curve)
    assert curve[0]["max_capital"] == pytest.approx(15986.09904431)
    assert curve[-1]["max_capital"] == pytest.approx(319721.98088619)
    assert curve[-1]["estimated_cost_bps"] >= curve[0]["estimated_cost_bps"]


def test_formal_promotion_requires_economic_gate_in_addition_to_oos():
    assessment = _assessment().to_dict()
    inputs = {
        "walkforward": {"passed": True, "window_pass_ratio": 0.9},
        "statistics": {"passed": True},
        "final_oos": {"passed": True},
        "oos_audit": {"audit_status": "PASSED"},
        "diagnostics": {"ic_decay": {1: 0.2, 5: 0.1}},
        "exposure": {"liquidity_exposure": 0.1},
        "capacity": {"daily_notional_capacity_proxy": 10.0},
        "recent_alpha": {"passed": True},
        "formal_research": True,
        "tradability_assessment": assessment,
    }
    assert evaluate_promotion_gate(**inputs).passed is True
    assessment["gate_result"]["passed"] = False
    assert evaluate_promotion_gate(**inputs).passed is False
    assert "TRADABILITY_CAPACITY_FAILED" in evaluate_promotion_gate(**inputs).reject_reasons
