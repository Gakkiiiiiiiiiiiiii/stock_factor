from __future__ import annotations

import torch

from stock_factor.technical_transformer.data.schemas import FEATURE_NAMES
from stock_factor.technical_transformer.training.masking import apply_mask


def test_mask_never_selects_invalid_days_or_unobserved_turnover() -> None:
    x = torch.randn(2, 128, len(FEATURE_NAMES))
    valid_days = torch.zeros(2, 128, dtype=torch.bool)
    valid_days[:, 64:] = True
    feature_validity = torch.ones_like(x, dtype=torch.bool)
    feature_validity[:, :, FEATURE_NAMES.index("turnover")] = False
    result = apply_mask(x, valid_days=valid_days, feature_validity=feature_validity, mode="day", seed=7)
    assert not result.positions[:, :64].any()
    assert not result.positions[:, :, FEATURE_NAMES.index("turnover")].any()
