from __future__ import annotations

from typing import Callable

import torch

from ..data.schemas import FEATURE_GROUPS, FEATURE_NAMES


def occlude_feature_group(x: torch.Tensor, group: str, *, value: float = 0.0) -> torch.Tensor:
    if group not in FEATURE_GROUPS:
        raise ValueError(f"unknown feature group: {group}")
    result = x.clone()
    indices = [FEATURE_NAMES.index(name) for name in FEATURE_GROUPS[group]]
    result[..., indices] = value
    return result


def run_feature_group_occlusion(
    model: torch.nn.Module,
    x: torch.Tensor,
    evaluate: Callable[[dict[str, torch.Tensor]], dict[str, float]],
) -> dict[str, dict[str, float]]:
    model.eval()
    result = {"full": evaluate(model(x))}
    for group in FEATURE_GROUPS:
        result[group] = evaluate(model(occlude_feature_group(x, group)))
    return result
