"""Candidate-level tradability and promotion orchestration."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import uuid4

from stock_factor.application.promotion.gate import evaluate_candidate_promotion
from stock_factor.domain.factor import FactorDefinition
from stock_factor.engine.oos_audit import audit_final_oos
from stock_factor.engine.oos_seal import DSL_VERSION, CandidateFreeze


def promote_candidates(
    *,
    service: Any,
    evaluated: list[dict],
    finalists: list[dict],
    finalist_hashes: set[str],
    cohort_statistics: dict,
    discovery_gates: dict,
    snapshot: Any,
    panel: dict,
    formal_ref: Any,
    formal_content_ref: Any,
    formal_content_query: Any,
    market_ref_hash: str | None,
    experiment: Any,
    research_mode: str,
    horizon: int,
    panel_feature_version: str,
    refs: Any,
    request: dict,
    progress: Any,
) -> list[dict]:
    """Persist factors after all discovery/OOS evidence has been assembled."""
    accepted: list[dict] = []
    for index, item in enumerate(evaluated):
        candidate = item["candidate"]
        rpn = list(candidate["rpn"])
        if hashlib.sha256(" ".join(rpn).encode()).hexdigest() != candidate["candidate_hash"]:
            raise ValueError("candidate_hash does not match the candidate RPN")
        values = item["values"]
        preliminary = item["preliminary"]
        walkforward = item["walkforward"]
        final_oos = item["final_oos"]
        diagnostics = item["diagnostics"]
        exposure = item["exposure"]
        capacity = item["capacity"]
        recent_alpha = item["recent_alpha"]
        split = item["split"]
        freeze = item.get("freeze")
        is_finalist = candidate["candidate_hash"] in finalist_hashes
        statistics = cohort_statistics[candidate["candidate_hash"]]
        discovery_gate = discovery_gates[candidate["candidate_hash"]]
        oos_audit = audit_final_oos(
            split=split,
            final_oos=final_oos,
            data_snapshot_id=snapshot.data_snapshot_id,
        )
        factor_id = uuid4().hex
        tradability_flags = {
            key: snapshot.bars[key]
            for key in (
                "tradable",
                "is_tradable",
                "can_trade",
                "halted",
                "is_halted",
                "suspended",
                "is_suspended",
                "limit",
                "limit_hit",
                "limit_up",
                "limit_down",
            )
            if key in snapshot.bars
        }
        tradability = service._tradability.assess(
            factor_artifact_id=factor_id,
            market_snapshot_id=snapshot.data_snapshot_id,
            factor_values=values,
            closes=panel["close"],
            amount=panel.get("amount"),
            volume=panel.get("volume"),
            execution_cost_calibration=request.get("execution_cost_calibration"),
            expected_execution_cost_calibration=service._expected_execution_cost_calibration,
            tradability_flags=tradability_flags,
            market_snapshot_ref=formal_ref,
            formal=research_mode == "FORMAL",
            neutralization={"available": bool(exposure), "exposure": exposure},
        )
        tradability_payload = tradability.to_dict()
        promotion = evaluate_candidate_promotion(
            walkforward=walkforward,
            statistics=statistics,
            final_oos=final_oos,
            oos_audit=oos_audit,
            diagnostics=diagnostics,
            exposure=exposure,
            capacity=capacity,
            recent_alpha=recent_alpha,
            tradability_assessment=tradability_payload,
            formal_research=research_mode == "FORMAL",
            data_snapshot_id=snapshot.data_snapshot_id,
        )
        if is_finalist:
            if final_oos and final_oos.get("passed") and promotion["passed"]:
                status = "OOS_PASSED"
            elif final_oos and final_oos.get("passed"):
                status = "DISCOVERY_PASSED"
            else:
                status = "OOS_FAILED"
        elif discovery_gate["passed"]:
            status = "DISCOVERY_PASSED"
        else:
            status = "DISCOVERY_CANDIDATE"
        freeze_dict = freeze.to_dict() if isinstance(freeze, CandidateFreeze) else freeze
        metrics = {
            "in_sample": preliminary,
            "walkforward": walkforward,
            "final_oos": final_oos,
            "oos_audit": oos_audit,
            "statistics": statistics,
            "diagnostics": diagnostics,
            "exposure": exposure,
            "capacity": capacity,
            "tradability_assessment": tradability_payload,
            "tradability_assessment_hash": tradability.artifact_id,
            "tradability_policy_version": tradability.policy_version,
            "tradability_policy_hash": tradability.policy_hash,
            "capacity_artifact_id": tradability.capacity_artifact.artifact_id
            if tradability.capacity_artifact
            else None,
            "recent_alpha": recent_alpha,
            "promotion_gate": promotion,
            "discovery_gate": discovery_gate,
            "is_finalist": is_finalist,
            "selection_rank": item.get("selection_rank"),
            "research_split": split.diagnostics(horizon, values.shape[1]) if split else None,
            "data_version": snapshot.data_version,
            "data_snapshot_id": snapshot.data_snapshot_id,
            "discovery_snapshot_id": freeze_dict["discovery_snapshot_id"]
            if freeze_dict
            else (refs.discovery_snapshot_id if refs else None),
            "final_oos_snapshot_id": freeze_dict["final_oos_snapshot_id"]
            if freeze_dict
            else (refs.final_oos_snapshot_id if refs else None),
            "candidate_frozen_at": freeze_dict["candidate_frozen_at"] if freeze_dict else None,
            "dsl_version": DSL_VERSION,
            "feature_set_version": panel_feature_version,
            "candidate_search_window": "discovery_only",
            "generation_round": candidate.get("generation_round", 1),
            "parent_candidate_id": candidate.get("parent_candidate_hash"),
            "generation_strategy": candidate.get("generation_strategy", "seed"),
            "generation_feedback": candidate.get("generation_feedback") or {},
            "research_experiment_id": experiment.experiment_id,
            "final_oos_dataset_ref_hash": experiment.final_oos_dataset_ref_hash,
            "market_ref_hash": market_ref_hash,
            "content_ref_hash": formal_content_ref.ref_hash if formal_content_ref else None,
            "content_snapshot_id": formal_content_query.content_snapshot_id if formal_content_query else None,
            "content_policy_version": formal_content_query.signal_policy_version if formal_content_query else None,
            "content_manifest": formal_content_ref.model_dump(mode="json") if formal_content_ref else None,
        }
        definition = FactorDefinition(
            factor_id=factor_id,
            name=candidate.get("name") or candidate["candidate_hash"][:12],
            rpn=rpn,
            hypothesis=candidate.get("hypothesis", ""),
            status=status,
            metrics=metrics,
            candidate_hash=candidate["candidate_hash"],
        )
        accepted.append(service._factors.save(definition))
        progress("promotion", 75 + int((index + 1) * 20 / max(len(evaluated), 1)))
    return accepted


__all__ = ["promote_candidates"]
