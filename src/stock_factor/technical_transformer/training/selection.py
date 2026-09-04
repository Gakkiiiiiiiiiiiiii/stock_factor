from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..evaluation.composite import technical_components


@dataclass(frozen=True)
class StageSelectionResult:
    stage: str
    score: float
    components: dict[str, float]
    valid: bool


def select_stage_score(stage: str, metrics: dict[str, Any]) -> StageSelectionResult:
    """Apply the frozen research selection policy for one training stage."""
    name = str(stage)
    if name == "masked_pretraining":
        loss = metrics.get("mask", metrics.get("mask_loss", metrics.get("loss")))
        valid = loss is not None
        score = -float(loss) if valid else float("-inf")
        return StageSelectionResult(name, score, {"mask_loss": float(loss)} if valid else {}, valid)
    components = technical_components(metrics)
    if name == "ma_bollinger":
        score = 0.50 * components["ma"] + 0.50 * components["boll"]
        selected = {"ma": components["ma"], "boll": components["boll"]}
    elif name == "wyckoff_primitives":
        score = 0.20 * components["ma"] + 0.20 * components["boll"] + 0.60 * components["primitive"]
        selected = {"ma": components["ma"], "boll": components["boll"], "primitive": components["primitive"]}
    elif name == "wyckoff_phase_events":
        score = (
            0.10 * components["ma"]
            + 0.10 * components["boll"]
            + 0.20 * components["primitive"]
            + 0.25 * components["phase"]
            + 0.35 * components["event"]
        )
        selected = components
    else:
        raise ValueError(f"unknown training stage: {stage}")
    valid = all(value is not None for value in selected.values())
    return StageSelectionResult(
        name, float(score) if valid else float("-inf"), {key: float(value) for key, value in selected.items()}, valid
    )
