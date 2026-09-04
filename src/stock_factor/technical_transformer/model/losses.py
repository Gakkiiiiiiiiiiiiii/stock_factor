from __future__ import annotations

import torch
from torch.nn import functional as F

from ..data.schemas import LABEL_SCHEMA


def _masked_huber(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    if valid is None:
        return F.huber_loss(pred, target, delta=1.0)
    mask = valid.to(dtype=torch.bool)
    if mask.shape != pred.shape:
        raise ValueError(f"valid mask shape {tuple(mask.shape)} does not match prediction {tuple(pred.shape)}")
    if not mask.any():
        return pred.sum() * 0.0
    return F.huber_loss(pred[mask], target[mask], delta=1.0)


def _soft_cross_entropy(
    logits: torch.Tensor, soft_target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    values = (-soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    if valid is not None:
        mask = valid.to(dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask.all(dim=-1)
        values = values[mask]
    return values.mean() if values.numel() else logits.sum() * 0.0


def event_pos_weight(
    labels: torch.Tensor, label_valid: torch.Tensor | None = None, *, cap: float = 20.0
) -> torch.Tensor:
    """Compute capped negative/positive weights without balancing eval data."""
    if label_valid is None:
        label_valid = torch.ones_like(labels, dtype=torch.bool)
    valid = label_valid.to(dtype=torch.bool)
    positives = (labels * valid.to(dtype=labels.dtype)).sum(dim=0)
    observations = valid.sum(dim=0).to(dtype=labels.dtype)
    negatives = observations - positives
    positives = positives.clamp_min(1.0)
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
            valid = label_valid[:, slices["ma"]] if label_valid is not None else None
            losses["ma"] = _masked_huber(outputs["ma"], labels[:, slices["ma"]], valid)
        if "bollinger" in requested:
            valid = label_valid[:, slices["bollinger"]] if label_valid is not None else None
            losses["bollinger"] = _masked_huber(outputs["bollinger"], labels[:, slices["bollinger"]], valid)
        if "wyckoff_primitives" in requested:
            valid = label_valid[:, slices["wyckoff_primitives"]] if label_valid is not None else None
            losses["wyckoff_primitives"] = _masked_huber(
                outputs["wyckoff_primitives"], labels[:, slices["wyckoff_primitives"]], valid
            )
        if "phase" in requested:
            valid = label_valid[:, slices["phase"]] if label_valid is not None else None
            losses["phase"] = _soft_cross_entropy(outputs["phase"], labels[:, slices["phase"]], valid)
        if "events" in requested:
            valid = label_valid[:, slices["events"]] if label_valid is not None else None
            values = F.binary_cross_entropy_with_logits(
                outputs["events"],
                labels[:, slices["events"]],
                pos_weight=event_weight,
                reduction="none",
            )
            if valid is not None:
                values = values[valid.to(dtype=torch.bool)]
            losses["events"] = values.mean() if values.numel() else outputs["events"].sum() * 0.0
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
