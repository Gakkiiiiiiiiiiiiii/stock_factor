from __future__ import annotations

from enum import Enum


class FactorLifecycleStatus(str, Enum):
    DRAFT = "DRAFT"
    COMPUTABLE = "COMPUTABLE"
    IN_SAMPLE_PASS = "IN_SAMPLE_PASS"
    RECENT_ALPHA = "RECENT_ALPHA"
    OOS_PASS = "OOS_PASS"
    PAPER_TRADING = "PAPER_TRADING"
    APPROVED = "APPROVED"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    RETIRED = "RETIRED"
    LEGACY_UNVERIFIED = "LEGACY_UNVERIFIED"


class FactorResearchStatus(str, Enum):
    """详细修改方案 §10：因子研究治理状态机（交易执行已迁往 Quant，不再有 PAPER_ELIGIBLE）。"""

    DRAFT = "DRAFT"
    DISCOVERY_CANDIDATE = "DISCOVERY_CANDIDATE"
    DISCOVERY_PASSED = "DISCOVERY_PASSED"
    FINALIST = "FINALIST"
    OOS_PASSED = "OOS_PASSED"
    OOS_FAILED = "OOS_FAILED"
    PROMOTED = "PROMOTED"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    RETIRED = "RETIRED"
    INVALIDATED = "INVALIDATED"


RESEARCH_STATUS_TRANSITIONS: dict[FactorResearchStatus, set[FactorResearchStatus]] = {
    FactorResearchStatus.DRAFT: {FactorResearchStatus.DISCOVERY_CANDIDATE, FactorResearchStatus.INVALIDATED},
    FactorResearchStatus.DISCOVERY_CANDIDATE: {
        FactorResearchStatus.DISCOVERY_PASSED,
        FactorResearchStatus.FINALIST,
        FactorResearchStatus.RETIRED,
        FactorResearchStatus.INVALIDATED,
    },
    FactorResearchStatus.DISCOVERY_PASSED: {
        FactorResearchStatus.FINALIST,
        FactorResearchStatus.RETIRED,
        FactorResearchStatus.INVALIDATED,
    },
    FactorResearchStatus.FINALIST: {
        FactorResearchStatus.OOS_PASSED,
        FactorResearchStatus.OOS_FAILED,
        FactorResearchStatus.INVALIDATED,
    },
    FactorResearchStatus.OOS_PASSED: {FactorResearchStatus.PROMOTED, FactorResearchStatus.INVALIDATED},
    FactorResearchStatus.OOS_FAILED: {FactorResearchStatus.RETIRED, FactorResearchStatus.INVALIDATED},
    FactorResearchStatus.PROMOTED: {
        FactorResearchStatus.ACTIVE,
        FactorResearchStatus.DEGRADED,
        FactorResearchStatus.INVALIDATED,
    },
    FactorResearchStatus.ACTIVE: {FactorResearchStatus.DEGRADED, FactorResearchStatus.RETIRED},
    FactorResearchStatus.DEGRADED: {FactorResearchStatus.ACTIVE, FactorResearchStatus.RETIRED},
    FactorResearchStatus.RETIRED: set(),
    FactorResearchStatus.INVALIDATED: set(),
}


def validate_research_transition(current: str, target: str) -> None:
    current_status = FactorResearchStatus(current)
    target_status = FactorResearchStatus(target)
    if target_status not in RESEARCH_STATUS_TRANSITIONS.get(current_status, set()):
        raise ValueError(f"非法因子状态迁移: {current} -> {target}")


ACTIVATION_ALLOWED = {FactorLifecycleStatus.APPROVED.value, FactorLifecycleStatus.ACTIVE.value}


def can_activate(status: str) -> bool:
    return status in ACTIVATION_ALLOWED
