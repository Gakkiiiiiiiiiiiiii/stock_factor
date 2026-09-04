from __future__ import annotations

import torch

from stock_factor.technical_transformer.data.schemas import FEATURE_NAMES
from stock_factor.technical_transformer.training.masking import apply_mask


def test_mixed_mask_is_non_empty() -> None:
    result = apply_mask(torch.randn(2, 128, len(FEATURE_NAMES)), mode="mixed", seed=123)
    assert result.positions.any()
    assert int(result.positions.sum()) > 0


def test_each_structured_mask_mode_is_non_empty() -> None:
    x = torch.randn(2, 128, len(FEATURE_NAMES))
    assert all(apply_mask(x, mode=mode, seed=7).positions.any() for mode in ("day", "group", "span"))
