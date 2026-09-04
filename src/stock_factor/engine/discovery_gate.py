"""Discovery Gate（收尾文档 §17）。

Discovery 阶段的准入门：只允许 WalkForward / Statistics / Diagnostics /
Exposure / Capacity / Recent Alpha 输入 —— 函数签名中不得出现 final_oos，
从类型层面保证 Final OOS 不参与候选选择（§14-§18）。
"""

from __future__ import annotations

DISCOVERY_GATE_VERSION = "discovery_gate_v1"


def evaluate_discovery_gate(
    *,
    walkforward: dict | None,
    statistics: dict | None,
    diagnostics: dict | None,
    exposure: dict | None,
    capacity: dict | None,
    recent_alpha: dict | None,
    min_window_pass_ratio: float = 0.60,
    min_capacity_proxy: float = 1.0,
    max_abs_liquidity_exposure: float = 0.80,
) -> dict:
    walkforward, statistics = walkforward or {}, statistics or {}
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
    if not diagnostics:
        reasons.append("DIAGNOSTICS_MISSING")
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
    return {
        "passed": not reasons,
        "reasons": reasons,
        "gate_version": DISCOVERY_GATE_VERSION,
    }


__all__ = ["evaluate_discovery_gate", "DISCOVERY_GATE_VERSION"]
