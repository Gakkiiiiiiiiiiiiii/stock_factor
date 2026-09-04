from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class TrainingStage:
    name: str
    epochs: int
    encoder_lr: float
    head_lr: float
    mask_mode: str | None = None
    active_heads: tuple[str, ...] = field(default_factory=tuple)
    patience: int = 3

    @classmethod
    def from_mapping(cls, value: dict, *, default_lr: float = 1e-4) -> "TrainingStage":
        head_lr = float(value.get("head_lr", value.get("lr", default_lr)))
        return cls(
            name=str(value["name"]),
            epochs=int(value.get("epochs", 1)),
            encoder_lr=float(value.get("encoder_lr", head_lr)),
            head_lr=head_lr,
            mask_mode=value.get("mask_mode"),
            active_heads=tuple(str(item) for item in value.get("active_heads", ())),
            patience=int(value.get("patience", 3)),
        )


def build_optimizer(model: torch.nn.Module, stage: TrainingStage, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """Build explicit encoder/head parameter groups with auditable LRs."""
    encoder = list(model.encoder.parameters())
    heads = list(model.heads.parameters())
    if not encoder or not heads:
        raise ValueError("model must expose non-empty encoder and heads modules")
    return torch.optim.AdamW(
        [
            {"name": "encoder", "params": encoder, "lr": stage.encoder_lr},
            {"name": "heads", "params": heads, "lr": stage.head_lr},
        ],
        weight_decay=float(weight_decay),
    )


def optimizer_group_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {str(group.get("name", index)): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}
