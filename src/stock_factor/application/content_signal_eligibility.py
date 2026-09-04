"""Point-in-time eligibility policy for v5.1 content signals."""

from __future__ import annotations

from dataclasses import dataclass

from stock_factor.domain.content_signal_v5 import ContentSignalV5, FormalContentQuery


@dataclass(frozen=True)
class ContentEligibilityResult:
    accepted: tuple[ContentSignalV5, ...]
    rejected: tuple[dict, ...]

    @property
    def formal_eligible(self) -> bool:
        return not self.rejected


def evaluate_content_eligibility(
    signals: list[ContentSignalV5], query: FormalContentQuery, *, allow_proxy: bool = False
) -> ContentEligibilityResult:
    accepted: list[ContentSignalV5] = []
    rejected: list[dict] = []
    for signal in signals:
        reasons: list[str] = []
        if signal.source_available_at > query.availability_as_of:
            reasons.append("source_available_after_availability_cutoff")
        if signal.available_from > query.availability_as_of:
            reasons.append("available_from_after_availability_cutoff")
        if signal.knowledge_projection_at > query.knowledge_as_of:
            reasons.append("knowledge_projection_after_knowledge_cutoff")
        if signal.lifecycle_as_of > query.knowledge_as_of:
            reasons.append("lifecycle_after_knowledge_cutoff")
        if signal.occurrence_at > query.business_as_of or signal.business_as_of > query.business_as_of:
            reasons.append("business_temporal_binding_after_business_cutoff")
        if signal.source_availability_quality == "UNKNOWN":
            reasons.append("unknown_source_availability")
        if signal.source_availability_quality == "PROXY" and not (allow_proxy or query.allow_proxy):
            reasons.append("proxy_source_not_allowed")
        if signal.truth_status.upper() not in {"EXTERNALLY_VERIFIED", "VERIFIED"}:
            reasons.append("unknown_truth_status")
        if signal.support_count is None or signal.support_count < query.min_support:
            reasons.append("insufficient_or_unknown_support")
        if signal.producer_commit != query.producer_commit:
            reasons.append("producer_commit_mismatch")
        if reasons:
            rejected.append({"signal_id": signal.signal_id, "reasons": reasons})
        else:
            accepted.append(signal)
    return ContentEligibilityResult(tuple(accepted), tuple(rejected))


__all__ = ["ContentEligibilityResult", "evaluate_content_eligibility"]
