from __future__ import annotations

import numpy as np
import pandas as pd

from stock_factor.technical_transformer.data.features import build_features
from stock_factor.technical_transformer.data.labels import build_labels
from stock_factor.technical_transformer.evaluation.causality import run_causality_suite


def test_causality_runner_reports_real_prefix_cases() -> None:
    rows = 150
    close = np.linspace(10, 20, rows)
    frame = pd.DataFrame(
        {
            "trading_date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "symbol": "A",
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
            "amount": close * 100,
            "turnover": 0.01,
        }
    )
    result = run_causality_suite(frame, feature_builder=build_features, label_builder=build_labels, cases=3)
    assert result["status"] == "EVALUATED" and result["cases"] == 3 and result["total_violations"] == 0
