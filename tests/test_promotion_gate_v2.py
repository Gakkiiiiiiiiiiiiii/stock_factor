"""Promotion Gate v2 与生命周期状态机（详细修改方案 §10 / §14）。"""
from __future__ import annotations

import pytest

from stock_factor.engine.lifecycle import FactorResearchStatus, validate_research_transition
from stock_factor.engine.promotion_gate import GATE_ORDER, evaluate_promotion_gate_v2


def _passing_inputs() -> dict:
    return {
        "data_quality": {"critical_flags": []},
        "walkforward": {"passed": True, "window_pass_ratio": 0.8},
        "stability": {"ic_sign_consistency": 0.9, "worst_regime_ic": 0.02},
        "statistics": {"passed": True},
        "final_oos": {"passed": True},
        "oos_audit": {"audit_status": "PASSED"},
        "cost_sensitivity": {"break_even_cost_bps": 30.0},
        "exposure": {"liquidity_exposure": 0.2},
        "neutralization": {"ic_retained_ratio": 0.8},
        "governance": {"experiment_id": "exp-1", "candidate_freeze": {"candidate_hash": "c-1"}},
    }


def test_gate_v2_passes_with_full_evidence():
    result = evaluate_promotion_gate_v2(**_passing_inputs())
    assert result["passed"] is True
    assert result["failed_gates"] == []
    assert result["gates_evaluated"] == list(GATE_ORDER)


def test_gate_v2_reports_every_missing_gate_not_boolean_only():
    result = evaluate_promotion_gate_v2()
    assert result["passed"] is False
    failed = {item["gate"] for item in result["failed_gates"]}
    assert failed == set(GATE_ORDER)
    assert all(item["reason"] for item in result["failed_gates"])


def test_gate_v2_specific_failure_reasons():
    inputs = _passing_inputs()
    inputs["statistics"] = {"passed": False}
    inputs["cost_sensitivity"] = {"break_even_cost_bps": 1.0}
    result = evaluate_promotion_gate_v2(**inputs)
    gates = {item["gate"]: item["reason"] for item in result["failed_gates"]}
    assert "MULTIPLE_TESTING" in gates
    assert "FDR" in gates["MULTIPLE_TESTING"]
    assert "COST_SENSITIVITY" in gates


def test_research_status_machine():
    validate_research_transition("DRAFT", "DISCOVERY_CANDIDATE")
    validate_research_transition("DISCOVERY_CANDIDATE", "DISCOVERY_PASSED")
    validate_research_transition("DISCOVERY_PASSED", "FINALIST")
    validate_research_transition("FINALIST", "OOS_PASSED")
    validate_research_transition("OOS_PASSED", "PROMOTED")
    validate_research_transition("PROMOTED", "ACTIVE")
    validate_research_transition("ACTIVE", "DEGRADED")
    validate_research_transition("DEGRADED", "RETIRED")
    with pytest.raises(ValueError):
        validate_research_transition("FINALIST", "ACTIVE")  # 不允许跳级
    with pytest.raises(ValueError):
        validate_research_transition("RETIRED", "ACTIVE")  # 终态


def test_paper_eligible_removed_from_lifecycle():
    # §10：交易执行迁往 Quant 后不得使用 PAPER_ELIGIBLE
    assert not hasattr(FactorResearchStatus, "PAPER_ELIGIBLE")
    with pytest.raises(ValueError):
        validate_research_transition("OOS_PASSED", "PAPER_ELIGIBLE")
