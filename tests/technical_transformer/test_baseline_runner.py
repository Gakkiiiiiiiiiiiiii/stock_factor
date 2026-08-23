from __future__ import annotations

import torch

from stock_factor.technical_transformer.data.schemas import FEATURE_NAMES, LABEL_SCHEMA
from stock_factor.technical_transformer.evaluation.baselines import make_baseline


def test_baseline_models_have_common_contract() -> None:
    x = torch.randn(2, 128, len(FEATURE_NAMES))
    for name in ("mlp", "gru", "tcn", "transformer"):
        output = make_baseline(name, len(FEATURE_NAMES), len(LABEL_SCHEMA.names))(x)
        assert output.shape == (2, len(LABEL_SCHEMA.names))
