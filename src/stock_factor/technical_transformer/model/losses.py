from __future__ import annotations

import torch
from torch.nn import functional as F

from ..data.schemas import LABEL_SCHEMA


def _masked_huber(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    if valid is None:
        return F.huber_loss(pred, target, delta=1.0)
    mask = valid.to(dtype=torch.bool)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(pred)
    if not mask.any():
        return pred.sum() * 0.0
    return F.huber_loss(pred[mask], target[mask], delta=1.0)


def _soft_cross_entropy(logits: torch.Tensor, soft_target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    values = (-soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    if valid is not None:
        values = values[valid.to(dtype=torch.bool)]
    return values.mean() if values.numel() else logits.sum() * 0.0


def event_pos_weight(labels: torch.Tensor, *, cap: float = 20.0) -> torch.Tensor:
    """Compute capped negative/positive weights without balancing eval data."""
    positives = labels.sum(dim=0).clamp_min(1.0)
    negatives = labels.shape[0] - positives
    return (negatives / positives).clamp(min=1.0, max=float(cap))


def compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    stage: str,
    mask_positions: torch.Tensor | None = None,
    active_heads: tuple[str, ...] | None = None,
    label_valid: torch.Tensor | None = None,
    event_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    slices = LABEL_SCHEMA.slices
    losses: dict[str, torch.Tensor] = {}
    active = set(active_heads or ())
    if stage == "masked_pretraining":
        losses["mask"] = outputs["mask_prediction"].sum() * 0.0
    else:
        requested = {"ma", "bollinger"} if stage == "ma_bollinger" else set()
        if stage in {"wyckoff_primitives", "wyckoff_phase_events"}:
            requested.update({"ma", "bollinger", "wyckoff_primitives"})
        if stage == "wyckoff_phase_events":
            requested.update({"phase", "events"})
        if active:
            requested &= active
        if "ma" in requested:
            losses["ma"] = _masked_huber(outputs["ma"], labels[:, slices["ma"]], label_valid)
        if "bollinger" in requested:
            losses["bollinger"] = _masked_huber(outputs["bollinger"], labels[:, slices["bollinger"]], label_valid)
        if "wyckoff_primitives" in requested:
            losses["wyckoff_primitives"] = _masked_huber(outputs["wyckoff_primitives"], labels[:, slices["wyckoff_primitives"]], label_valid)
        if "phase" in requested:
            losses["phase"] = _soft_cross_entropy(outputs["phase"], labels[:, slices["phase"]], label_valid)
        if "events" in requested:
            losses["events"] = F.binary_cross_entropy_with_logits(
                outputs["events"], labels[:, slices["events"]], pos_weight=event_weight,
            )
    weights = {"mask": 1.0, "ma": 1.0, "bollinger": 1.0, "wyckoff_primitives": 1.5, "phase": 0.5, "events": 0.5}
    total = sum((losses[name] * weights[name] for name in losses), outputs[next(iter(outputs))].sum() * 0.0)
    return total, {name: float(value.detach().cpu()) for name, value in losses.items()}


def compute_mask_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    positions: torch.Tensor,
    *,
    target_indices: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if target_indices is not None:
        target = target[..., list(target_indices)]
        if positions.ndim == 3:
            positions = positions[..., list(target_indices)]
    if positions.ndim == 3 and positions.shape[-1] != prediction.shape[-1]:
        raise ValueError("mask positions and reconstruction prediction have incompatible feature dimensions")
    if positions.ndim == 3:
        mask = positions
    elif positions.ndim == 2:
        mask = positions.unsqueeze(-1).expand_as(prediction)
    else:
        raise ValueError("positions must be [batch, sequence] or [batch, sequence, features]")
    if not mask.any():
        return prediction.sum() * 0.0
    return F.huber_loss(prediction[mask], target[mask], delta=1.0)
