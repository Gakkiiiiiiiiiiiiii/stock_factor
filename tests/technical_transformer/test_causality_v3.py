from __future__ import annotations

import numpy as np
import pandas as pd

from stock_factor.technical_transformer.data.labels import build_labels
from stock_factor.technical_transformer.evaluation.causality import run_causality_suite


def _frame(rows: int = 150) -> pd.DataFrame:
    close = np.linspace(10.0, 20.0, rows)
    return pd.DataFrame(
        {
            "trading_date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "symbol": "AAA",
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
            "amount": close * 100.0,
            "turnover": 0.01,
        }
    )


def test_causality_detects_intentional_lookahead() -> None:
    def lookahead(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"future_return": frame["close"].shift(-1) / frame["close"] - 1.0})

    result = run_causality_suite(
        _frame(),
        feature_builder=lookahead,
        label_builder=build_labels,
        cases=3,
        seed=4,
    )
    assert result["feature_violations"] > 0
    assert result["passed"] is False
