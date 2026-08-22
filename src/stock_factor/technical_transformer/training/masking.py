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
) -> None:
    steps = positions.shape[1]
    if mode == "day":
        count = max(1, int(round(steps * 0.125)))
        days = torch.randperm(steps, generator=generator)[:count]
        positions[batch_index, days[:, None], torch.tensor(_CONTINUOUS_INDICES) ] = True
        groups[batch_index, days[:, None], torch.tensor(_CONTINUOUS_INDICES)] = 0
        return
    if mode == "group":
        count = max(1, int(round(steps * 0.125)))
        days = torch.randperm(steps, generator=generator)[:count]
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


def apply_mask(x: torch.Tensor, *, mode: str = "mixed", seed: int = 0) -> MaskedBatch:
    """Apply the one canonical structured mask used by train and validation.

    ``positions`` is feature-level so reconstruction can be restricted to the
    selected primitives. ``groups`` contains ``-1`` for unmasked features,
    ``0`` for day/span masks, and a stable positive id for group masks.
    """
    if x.ndim != 3 or x.shape[2] != len(FEATURE_NAMES):
        raise ValueError(f"expected [batch, sequence, {len(FEATURE_NAMES)}], got {tuple(x.shape)}")
    target = x.detach().clone()
    masked_input = x.detach().clone()
    positions = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype=torch.bool)
    groups = torch.full((x.shape[0], x.shape[1], x.shape[2]), fill_value=-1, dtype=torch.int64)
    if mode in {"none", "off"}:
        return MaskedBatch(masked_input, target, positions, groups)
    if mode not in {"mixed", "day", "group", "span"}:
        raise ValueError(f"unknown mask mode: {mode}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for batch_index in range(x.shape[0]):
        if mode == "mixed":
            draw = float(torch.rand((), generator=generator))
            selected = "day" if draw < 0.50 else ("group" if draw < 0.80 else "span")
        else:
            selected = mode
        _one_mask(positions, groups, batch_index, mode=selected, generator=generator)
    positions = positions.to(x.device)
    groups = groups.to(x.device)
    # Zero is only the transport value.  The encoder receives positions and
    # adds a learnable mask embedding, distinguishing it from a real zero.
    masked_input = masked_input.masked_fill(positions, 0.0)
    return MaskedBatch(masked_input, target, positions, groups)
