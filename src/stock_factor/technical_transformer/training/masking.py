from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.schemas import CONTINUOUS_FEATURES, FEATURE_GROUPS, FEATURE_NAMES


@dataclass(frozen=True)
class MaskedBatch:
    input: torch.Tensor
    target: torch.Tensor
    positions: torch.Tensor
    groups: torch.Tensor


_GROUP_IDS = {name: index + 1 for index, name in enumerate(FEATURE_GROUPS)}
_CONTINUOUS_INDICES = tuple(FEATURE_NAMES.index(name) for name in CONTINUOUS_FEATURES)
_GROUP_INDICES = {
    name: tuple(FEATURE_NAMES.index(feature) for feature in features)
    for name, features in FEATURE_GROUPS.items()
}


def _one_mask(
    positions: torch.Tensor,
    groups: torch.Tensor,
    batch_index: int,
    *,
    mode: str,
    generator: torch.Generator,
    valid_days: torch.Tensor | None = None,
) -> None:
    steps = positions.shape[1]
    candidates = torch.arange(steps, dtype=torch.long)
    if valid_days is not None:
        candidates = candidates[valid_days[batch_index].detach().cpu()]
    if candidates.numel() == 0:
        return
    if mode == "day":
        count = min(candidates.numel(), max(1, int(round(candidates.numel() * 0.125))))
        days = candidates[torch.randperm(candidates.numel(), generator=generator)[:count]]
        positions[batch_index, days[:, None], torch.tensor(_CONTINUOUS_INDICES) ] = True
        groups[batch_index, days[:, None], torch.tensor(_CONTINUOUS_INDICES)] = 0
        return
    if mode == "group":
        count = min(candidates.numel(), max(1, int(round(candidates.numel() * 0.125))))
        days = candidates[torch.randperm(candidates.numel(), generator=generator)[:count]]
        names = list(_GROUP_INDICES)
        group_name = names[int(torch.randint(len(names), (1,), generator=generator))]
        indices = torch.tensor(_GROUP_INDICES[group_name])
        positions[batch_index, days[:, None], indices] = True
        groups[batch_index, days[:, None], indices] = _GROUP_IDS[group_name]
        return
    if mode == "span":
        length = int(torch.randint(2, 6, (1,), generator=generator))
        start = int(torch.randint(max(1, steps - length + 1), (1,), generator=generator))
        positions[batch_index, start:start + length, torch.tensor(_CONTINUOUS_INDICES)] = True
        groups[batch_index, start:start + length, torch.tensor(_CONTINUOUS_INDICES)] = 0
        return
    raise ValueError(f"unknown mask mode: {mode}")


def apply_mask(
    x: torch.Tensor,
    *,
    valid_days: torch.Tensor | None = None,
    feature_validity: torch.Tensor | None = None,
    mode: str = "mixed",
    seed: int = 0,
) -> MaskedBatch:
    """Apply the one canonical structured mask used by train and validation.

    ``positions`` is feature-level so reconstruction can be restricted to the
    selected primitives. ``groups`` contains ``-1`` for unmasked features,
    ``0`` for day/span masks, and a stable positive id for group masks.
    """
    if x.ndim != 3 or x.shape[2] != len(FEATURE_NAMES):
        raise ValueError(f"expected [batch, sequence, {len(FEATURE_NAMES)}], got {tuple(x.shape)}")
    target = x.detach().clone()
    if valid_days is not None:
        if valid_days.shape != x.shape[:2]:
            raise ValueError("valid_days must have shape [batch, sequence]")
        valid_days = valid_days.to(device=x.device, dtype=torch.bool)
    if feature_validity is not None:
        if feature_validity.shape != x.shape:
            raise ValueError("feature_validity must have shape [batch, sequence, features]")
        feature_validity = feature_validity.to(device=x.device, dtype=torch.bool)
    # Mask sampling uses a CPU generator for deterministic cross-device masks;
    # apply device tensors only after the sampling pass.
    valid_days_cpu = valid_days.detach().cpu() if valid_days is not None else None
    feature_validity_cpu = feature_validity.detach().cpu() if feature_validity is not None else None
    masked_input = x.detach().clone()
    positions = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype=torch.bool)
    groups = torch.full((x.shape[0], x.shape[1], x.shape[2]), fill_value=-1, dtype=torch.int64)
    if mode in {"none", "off"}:
        return MaskedBatch(masked_input, target, positions.to(x.device), groups.to(x.device))
    if mode not in {"mixed", "day", "group", "span"}:
        raise ValueError(f"unknown mask mode: {mode}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for batch_index in range(x.shape[0]):
        if mode == "mixed":
            draw = float(torch.rand((), generator=generator))
            selected = "day" if draw < 0.50 else ("group" if draw < 0.80 else "span")
        else:
            selected = mode
        _one_mask(
            positions,
            groups,
            batch_index,
            mode=selected,
            generator=generator,
            valid_days=valid_days_cpu,
        )
    if valid_days is not None:
        positions &= valid_days_cpu.unsqueeze(-1)
    if feature_validity is not None:
        positions &= feature_validity_cpu
    positions = positions.to(x.device)
    groups = groups.to(x.device)
    # Zero is only the transport value.  The encoder receives positions and
    # adds a learnable mask embedding, distinguishing it from a real zero.
    masked_input = masked_input.masked_fill(positions, 0.0)
    return MaskedBatch(masked_input, target, positions, groups)
