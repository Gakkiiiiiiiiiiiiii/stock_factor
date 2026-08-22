from __future__ import annotations

import torch
from torch.nn import functional as F

from ..data.schemas import LABEL_SCHEMA


def _masked_huber(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=1.0)


def _soft_cross_entropy(logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
    return (-soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    stage: str,
    mask_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    slices = LABEL_SCHEMA.slices
    losses: dict[str, torch.Tensor] = {}
    if stage in {"masked_pretraining", "ma_bollinger", "wyckoff_primitives", "wyckoff_phase_events"}:
        if stage == "masked_pretraining":
            if mask_positions is None or not mask_positions.any():
                losses["mask"] = outputs["mask_prediction"].sum() * 0.0
            else:
                # Pretraining targets are supplied separately by the trainer;
                # the placeholder is replaced in compute_mask_loss below.
                losses["mask"] = outputs["mask_prediction"].sum() * 0.0
        if stage in {"ma_bollinger", "wyckoff_primitives", "wyckoff_phase_events"}:
            losses["ma"] = _masked_huber(outputs["ma"], labels[:, slices["ma"]])
            losses["bollinger"] = _masked_huber(outputs["bollinger"], labels[:, slices["bollinger"]])
        if stage in {"wyckoff_primitives", "wyckoff_phase_events"}:
            losses["wyckoff_primitives"] = _masked_huber(outputs["wyckoff_primitives"], labels[:, slices["wyckoff_primitives"]])
        if stage == "wyckoff_phase_events":
            losses["phase"] = _soft_cross_entropy(outputs["phase"], labels[:, slices["phase"]])
            losses["events"] = F.binary_cross_entropy_with_logits(outputs["events"], labels[:, slices["events"]])
    weights = {"mask": 1.0, "ma": 1.0, "bollinger": 1.0, "wyckoff_primitives": 1.5, "phase": 0.5, "events": 0.5}
    total = sum(losses[name] * weights[name] for name in losses)
    return total, {name: float(value.detach().cpu()) for name, value in losses.items()}


def compute_mask_loss(prediction: torch.Tensor, target: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if not positions.any():
        return prediction.sum() * 0.0
    return F.huber_loss(prediction[positions], target[positions], delta=1.0)
