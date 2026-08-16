"""Single, persisted gate for moving a factor into paper trading.

Keeping this decision separate from the miner prevents a caller from treating a
successful walk-forward run as an implicit production approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromotionGateResult:
    passed: bool
    metrics: dict[str, Any]
    reject_reasons: list[str]

    def model_dump(self) -> dict:
        return {"passed": self.passed, "metrics": self.metrics, "reject_reasons": self.reject_reasons}


def evaluate_promotion_gate(
    *,
    walkforward: dict | None,
    statistics: dict | None,
    final_oos: dict | None = None,
    oos_audit: dict | None = None,
    diagnostics: dict | None = None,
    exposure: dict | None = None,
    capacity: dict | None = None,
    recent_alpha: dict | None = None,
    min_window_pass_ratio: float = 0.60,
    min_capacity_proxy: float = 1.0,
    max_abs_liquidity_exposure: float = 0.80,
    gate_version: str = "promotion_gate_v2",
    statistical_validation_version: str = "factor_statistics_v1",
    data_snapshot_id: str | None = None,
) -> PromotionGateResult:
    """Deterministic promotion gate with explicit diagnostic provenance."""
    walkforward, statistics = walkforward or {}, statistics or {}
    final_oos, oos_audit = final_oos or {}, oos_audit or {}
    diagnostics, exposure, capacity, recent_alpha = (
        diagnostics or {},
        exposure or {},
        capacity or {},
        recent_alpha or {},
    )
    reasons: list[str] = []
    if not walkforward.get("passed"):
        reasons.append("WALKFORWARD_FAILED")
    if float(walkforward.get("window_pass_ratio", 0.0)) < min_window_pass_ratio:
        reasons.append("WALKFORWARD_COVERAGE_FAILED")
    if not statistics.get("passed"):
        reasons.append("STATISTICAL_VALIDATION_FAILED")
    if not final_oos.get("passed"):
        reasons.append("FINAL_OOS_FAILED")
    if oos_audit.get("audit_status") != "PASSED":
        reasons.append("OOS_AUDIT_FAILED")
    if not diagnostics:
        reasons.append("DIAGNOSTICS_MISSING")
    else:
        decay = diagnostics.get("ic_decay") or {}
        day_1 = decay.get(1, decay.get("1"))
        day_5 = decay.get(5, decay.get("5"))
        if day_1 is not None and day_5 is not None and abs(float(day_1)) > 1e-8 and abs(float(day_5)) <= 1e-8:
            reasons.append("IC_DECAY_FAILED")
    if not exposure:
        reasons.append("EXPOSURE_MISSING")
    elif (
        exposure.get("liquidity_exposure") is not None
        and abs(float(exposure["liquidity_exposure"])) > max_abs_liquidity_exposure
    ):
        reasons.append("LIQUIDITY_EXPOSURE_FAILED")
    if not capacity:
        reasons.append("CAPACITY_MISSING")
    elif float(capacity.get("daily_notional_capacity_proxy", 0.0)) < min_capacity_proxy:
        reasons.append("CAPACITY_FAILED")
    if not recent_alpha or not recent_alpha.get("passed"):
        reasons.append("RECENT_ALPHA_FAILED")
    return PromotionGateResult(
        passed=not reasons,
        metrics={
            "walkforward": walkforward,
            "statistics": statistics,
            "final_oos": final_oos,
            "oos_audit": oos_audit,
            "diagnostics": diagnostics,
            "exposure": exposure,
            "capacity": capacity,
            "recent_alpha": recent_alpha,
            "promotion_gate_version": gate_version,
            "statistical_validation_version": statistical_validation_version,
            "data_snapshot_id": data_snapshot_id,
        },
        reject_reasons=reasons,
    )


PROMOTION_GATE_V2_VERSION = "promotion_gate_v3"

# 详细修改方案 §14：八道正式 Gate。
GATE_ORDER = (
    "DATA_QUALITY",
    "DISCOVERY_IC",
    "STABILITY",
    "MULTIPLE_TESTING",
    "FINAL_OOS",
    "COST_SENSITIVITY",
    "EXPOSURE_NEUTRALIZATION",
    "RESEARCH_GOVERNANCE",
)


def evaluate_promotion_gate_v2(
    *,
    data_quality: dict | None = None,
    walkforward: dict | None = None,
    stability: dict | None = None,
    statistics: dict | None = None,
    final_oos: dict | None = None,
    oos_audit: dict | None = None,
    cost_sensitivity: dict | None = None,
    exposure: dict | None = None,
    neutralization: dict | None = None,
    governance: dict | None = None,
    min_window_pass_ratio: float = 0.60,
    min_sign_consistency: float = 0.55,
    min_break_even_cost_bps: float = 5.0,
    max_abs_liquidity_exposure: float = 0.80,
    min_ic_retained_ratio: float = 0.30,
) -> dict:
    """详细修改方案 §14：禁止只返回 boolean，必须给出每道门的完整原因。"""
    data_quality = data_quality or {}
    walkforward = walkforward or {}
    stability = stability or {}
    statistics = statistics or {}
    final_oos = final_oos or {}
    oos_audit = oos_audit or {}
    cost_sensitivity = cost_sensitivity or {}
    exposure = exposure or {}
    neutralization = neutralization or {}
    governance = governance or {}

    failed_gates: list[dict[str, str]] = []

    def _fail(gate: str, reason: str) -> None:
        failed_gates.append({"gate": gate, "reason": reason})

    # Gate 1 - Data Quality
    critical_flags = data_quality.get("critical_flags") or []
    if not data_quality:
        _fail("DATA_QUALITY", "data quality report missing")
    elif critical_flags:
        _fail("DATA_QUALITY", f"critical quality flags: {', '.join(critical_flags)}")

    # Gate 2 - Discovery IC
    if not walkforward.get("passed"):
        _fail("DISCOVERY_IC", "walk-forward discovery gate failed")
    elif float(walkforward.get("window_pass_ratio", 0.0)) < min_window_pass_ratio:
        _fail("DISCOVERY_IC", f"window pass ratio < {min_window_pass_ratio}")

    # Gate 3 - Stability
    if not stability:
        _fail("STABILITY", "stability report missing")
    else:
        consistency = float(stability.get("ic_sign_consistency", 0.0))
        if consistency < min_sign_consistency:
            _fail("STABILITY", f"ic sign consistency {consistency} < {min_sign_consistency}")
        worst = stability.get("worst_regime_ic")
        if worst is not None and float(worst) <= 0:
            _fail("STABILITY", f"worst regime ic <= 0 ({worst})")

    # Gate 4 - Multiple Testing
    if not statistics.get("passed"):
        _fail("MULTIPLE_TESTING", "FDR adjusted p-value > threshold (or DSR/PBO failed)")

    # Gate 5 - Final OOS
    if not final_oos.get("passed"):
        _fail("FINAL_OOS", final_oos.get("reason") or "final oos evaluation failed")
    if oos_audit.get("audit_status") not in (None, "PASSED"):
        _fail("FINAL_OOS", f"oos audit status {oos_audit.get('audit_status')}")

    # Gate 6 - Cost Sensitivity
    if not cost_sensitivity:
        _fail("COST_SENSITIVITY", "cost sensitivity report missing")
    elif float(cost_sensitivity.get("break_even_cost_bps", 0.0)) < min_break_even_cost_bps:
        _fail("COST_SENSITIVITY", f"break-even cost < {min_break_even_cost_bps}bps")

    # Gate 7 - Exposure / Neutralization
    if not exposure:
        _fail("EXPOSURE_NEUTRALIZATION", "exposure report missing")
    else:
        liquidity = exposure.get("liquidity_exposure")
        if liquidity is not None and abs(float(liquidity)) > max_abs_liquidity_exposure:
            _fail("EXPOSURE_NEUTRALIZATION", f"liquidity exposure {liquidity} exceeds limit")
        retained = neutralization.get("ic_retained_ratio")
        if retained is not None and float(retained) < min_ic_retained_ratio:
            _fail("EXPOSURE_NEUTRALIZATION", f"neutralized ic retained ratio {retained} < {min_ic_retained_ratio}")

    # Gate 8 - Research Governance
    if not governance:
        _fail("RESEARCH_GOVERNANCE", "governance evidence missing (experiment lineage / freeze required)")
    else:
        for required in ("experiment_id", "candidate_freeze"):
            if not governance.get(required):
                _fail("RESEARCH_GOVERNANCE", f"{required} missing")

    return {
        "passed": not failed_gates,
        "gate_version": PROMOTION_GATE_V2_VERSION,
        "gates_evaluated": list(GATE_ORDER),
        "failed_gates": failed_gates,
    }
