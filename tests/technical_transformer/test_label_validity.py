from __future__ import annotations

import numpy as np
import pandas as pd

from stock_factor.technical_transformer.data.labels import build_labels


def _bars(rows: int = 240) -> pd.DataFrame:
    close = np.linspace(10, 13, rows)
    return pd.DataFrame({
        "trading_date": pd.date_range("2020-01-01", periods=rows, freq="B"), "symbol": "AAA.SZ",
        "open": close - 0.02, "high": close + 0.05, "low": close - 0.05, "close": close,
        "volume": 100000.0, "amount": close * 100000.0, "turnover": 0.01,
        "turnover_observed": True,
    })


def test_missing_turnover_masks_effort_but_not_trend() -> None:
    bars = _bars()
    bars.loc[160, ["turnover", "turnover_observed"]] = [0.0, False]
    labels = build_labels(bars)
    assert not labels.valid.loc[160, "effort_result_score"]
    assert labels.valid.loc[160, "trend_direction"]


def test_rolling_targets_are_invalid_before_history_exists() -> None:
    labels = build_labels(_bars(130))
    assert not labels.valid.loc[18, "percent_b"]
    assert not labels.valid.loc[118, "bandwidth_percentile"]
