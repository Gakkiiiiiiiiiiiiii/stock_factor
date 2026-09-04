"""Assembly of immutable formal artifact evidence from completed use cases."""

from __future__ import annotations

from datetime import UTC, datetime

from stock_factor.domain.oos_run import canonical_candidate_set_hash
from stock_factor.domain.research_artifact import ResearchArtifactV2


def assemble_research_artifact(
    *,
    experiment,
    request: dict,
    accepted: list,
    frozen_items: list[dict],
    formal_ref,
    formal_content_ref,
    cohort_statistics: dict,
    cohort_evidence,
    contract_checksums: dict[str, str],
    dependency_lock_hash: str,
    producer_commit: str,
) -> ResearchArtifactV2:
    """Build a V2 artifact only from a real sealed cohort result."""

    def factor_field(item, name, default=None):
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    def factor_metrics(item):
        return factor_field(item, "metrics", {}) or {}

    primary = next((item for item in accepted if factor_metrics(item).get("is_finalist")), None)
    if primary is None or not cohort_evidence.cohort_artifact_hash:
        raise ValueError("formal artifact requires a real sealed OOS cohort")
    research_question = str(request.get("research_question") or "").strip()
    hypothesis = str(request.get("hypothesis") or "").strip()
    producer_commit = str(producer_commit or "").strip()
    if not research_question or not hypothesis or not producer_commit:
        raise ValueError("formal artifact requires research_question, hypothesis, and producer commit")
    primary_metrics = factor_metrics(primary)
    final_oos_metrics = primary_metrics.get("final_oos") or {}
    promotion_decision = primary_metrics.get("promotion_gate") or {"passed": False}
    tradability_assessment = primary_metrics.get("tradability_assessment") or {"formal_eligible": False}
    final_oos_evidence = {
        "status": "SEALED",
        "passed": bool(final_oos_metrics.get("passed")),
        "cohort_artifact_hash": cohort_evidence.cohort_artifact_hash,
        "run_id": cohort_evidence.run_id,
        "result": final_oos_metrics,
    }
    promoted_factors = [
        {"factor_id": factor_field(factor, "factor_id"), "version": factor_field(factor, "version")}
        for factor in accepted
        if factor_metrics(factor).get("is_finalist")
        and (factor_metrics(factor).get("final_oos") or {}).get("passed")
        and (factor_metrics(factor).get("promotion_gate") or {}).get("passed")
    ]
    return ResearchArtifactV2(
        experiment_id=experiment.experiment_id,
        research_question=research_question,
        hypothesis=hypothesis,
        dataset_manifest={
            "final_oos_dataset_ref": experiment.final_oos_dataset_ref,
            "final_oos_dataset_ref_hash": experiment.final_oos_dataset_ref_hash,
            "universe_identity": formal_ref.universe_version if formal_ref else experiment.symbols,
        },
        market_ref=formal_ref.__dict__ if formal_ref else {},
        content_ref=formal_content_ref.model_dump(mode="json") if formal_content_ref else {},
        candidate_set_hash=canonical_candidate_set_hash([item["candidate"]["candidate_hash"] for item in frozen_items]),
        statistical_experiment={
            "multiple_testing": cohort_statistics,
            "dsr": {
                candidate_id: {
                    "deflated_sharpe": evidence.get("deflated_sharpe"),
                    "passed": evidence.get("passed"),
                }
                for candidate_id, evidence in cohort_statistics.items()
            },
            "pbo": {
                candidate_id: {
                    "pbo": evidence.get("pbo"),
                    "passed": evidence.get("passed_pbo"),
                }
                for candidate_id, evidence in cohort_statistics.items()
            },
        },
        final_oos_evidence=final_oos_evidence,
        tradability_assessment=tradability_assessment,
        promotion_decision=promotion_decision,
        promotion_policy_version=promotion_decision.get("gate_version") or "promotion_gate_v2",
        producer_commit=producer_commit,
        dependency_lock_hash=dependency_lock_hash,
        contract_checksums=contract_checksums,
        created_at=datetime.now(UTC),
        factor_set={"factors": promoted_factors} if promoted_factors else None,
        readiness_evidence=experiment.readiness_evidence,
    )


__all__ = ["assemble_research_artifact"]
