from __future__ import annotations

import numpy as np
import pandas as pd

from stock_factor.technical_transformer.data.labels import build_labels


def test_placeholder_turnover_never_becomes_effort_negative() -> None:
    rows = 240
    close = np.linspace(10, 13, rows)
    frame = pd.DataFrame(
        {
            "trading_date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 100000.0,
            "amount": close * 100000.0,
            "turnover": 0.01,
            "turnover_observed": True,
        }
    )
    frame.loc[160, ["turnover", "turnover_observed"]] = [0.0, False]
    labels = build_labels(frame)
    assert not labels.valid.loc[160, "effort_result_divergence"]
