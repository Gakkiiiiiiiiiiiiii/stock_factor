"""Final OOS scheduling: freeze-admission, dataset binding, and evaluation."""

from __future__ import annotations

import hashlib
import json

from stock_factor.application.oos.evaluate import FinalOosCandidateInput
from stock_factor.domain.datasets import FinalOosDatasetRef
from stock_factor.domain.oos_run import canonical_candidate_set_hash


def schedule_final_oos(
    *,
    oos_service,
    oos_data,
    experiment,
    frozen_items: list[dict],
    snapshot,
    panel: dict,
    start: str,
    content_ref,
    formal_ref,
    research_mode: str,
    feature_set_version_value: str,
    experiment_config_hash: str,
    horizon: int,
):
    """Open one cohort and return its immutable dataset/evidence pair."""
    candidate_set = [item["candidate"]["candidate_hash"] for item in frozen_items]
    final_split = frozen_items[0]["split"]
    final_start = (
        snapshot.dates[final_split.final_oos_start] if final_split.final_oos_start < len(snapshot.dates) else start
    )
    final_end_index = max(final_split.final_oos_start, final_split.final_oos_end - 1)
    final_end = snapshot.dates[final_end_index] if final_end_index < len(snapshot.dates) else snapshot.dates[-1]
    final_dataset_ref = FinalOosDatasetRef(
        market_snapshot_id=(formal_ref.market_snapshot_id if formal_ref is not None else snapshot.data_snapshot_id),
        source_data_version=snapshot.data_version,
        universe_snapshot_id=(formal_ref.universe_version if formal_ref is not None else snapshot.data_snapshot_id),
        feature_schema_version=feature_set_version_value,
        start=final_start,
        end=final_end,
        warmup_start=snapshot.dates[0] if snapshot.dates else start,
        formal_market_ref=formal_ref,
        formal_content_ref=content_ref,
    )
    if research_mode == "FORMAL" and not final_dataset_ref.formal_eligible:
        raise ValueError("FORMAL Final OOS dataset must have market and content references")
    dataset_ref_hash = final_dataset_ref.dataset_hash
    experiment.final_oos_dataset_ref_hash = dataset_ref_hash
    experiment.final_oos_dataset_ref = final_dataset_ref.to_dict()
    oos_service.register_authorization(
        experiment.experiment_id,
        snapshot.data_snapshot_id,
        canonical_candidate_set_hash(candidate_set),
        dataset_ref_hash=dataset_ref_hash,
        market_snapshot_id=(formal_ref.market_snapshot_id if formal_ref else snapshot.data_snapshot_id),
        content_ref_hash=(content_ref.ref_hash if content_ref else None),
    )
    cohort_inputs: list[FinalOosCandidateInput] = []
    for item in frozen_items:
        input_identity = hashlib.sha256(
            json.dumps(
                {
                    "candidate_hash": item["candidate"]["candidate_hash"],
                    "final_oos_dataset_ref_hash": final_dataset_ref.dataset_hash,
                    "final_oos_start": final_split.final_oos_start,
                    "final_oos_end": final_split.final_oos_end,
                    "horizon": horizon,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        candidate_hash = item["candidate"]["candidate_hash"]
        if getattr(oos_service, "_run_service", None) is not None:
            cohort_inputs.append(
                FinalOosCandidateInput(
                    candidate_hash=candidate_hash,
                    loader=lambda item=item: oos_data.load(experiment, (item["values"], panel["close"])),
                    input_identity=input_identity,
                )
            )
        else:
            values_oos, closes_oos = oos_data.load(experiment, (item["values"], panel["close"]))
            cohort_inputs.append(
                FinalOosCandidateInput(candidate_hash=candidate_hash, values=values_oos, closes=closes_oos)
            )
    evidence = oos_service.evaluate_cohort_with_evidence(experiment, cohort_inputs, final_split, horizon)
    return final_dataset_ref, evidence


__all__ = ["schedule_final_oos"]
