"""Discovery cohort statistics and multiple-testing gate inputs."""

from __future__ import annotations

import numpy as np

from stock_factor.application.statistical_experiment import rank_ic_series, validate_statistical_experiment
from stock_factor.engine.discovery_gate import evaluate_discovery_gate


def cohort_statistics(evaluated: list[dict], close: np.ndarray, horizon: int) -> dict:
    """Build the immutable multiple-testing universe from discovery windows only."""
    return validate_statistical_experiment(
        {
            item["candidate"]["candidate_hash"]: rank_ic_series(
                item["values"][:, : item["split"].discovery_end] if item["split"] else item["values"],
                close[:, : item["split"].discovery_end] if item["split"] else close,
                horizon,
            )
            for item in evaluated
        }
    )


def discovery_gates(evaluated: list[dict], statistics: dict) -> dict:
    return {
        item["candidate"]["candidate_hash"]: evaluate_discovery_gate(
            walkforward=item["walkforward"],
            statistics=statistics[item["candidate"]["candidate_hash"]],
            diagnostics=item["diagnostics"],
            exposure=item["exposure"],
            capacity=item["capacity"],
            recent_alpha=item["recent_alpha"],
        )
        for item in evaluated
    }


__all__ = ["cohort_statistics", "discovery_gates"]
