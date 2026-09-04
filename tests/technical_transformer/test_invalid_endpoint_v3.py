from __future__ import annotations

import numpy as np
import pandas as pd

from stock_factor.technical_transformer.data.dataset import DatasetConfig, _eligible_samples


def test_invalid_endpoint_is_not_sampled_even_with_good_window_ratio() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=128, freq="B"))
    listing_days = pd.Series(np.full(128, 200.0))
    quality = np.ones(128, dtype=float)
    quality[-1] = 0.0
    config = DatasetConfig(step_len=128, stride=1, min_listing_days=160, min_quality_ratio=0.80)
    assert _eligible_samples(dates, listing_days, quality, config) == []
