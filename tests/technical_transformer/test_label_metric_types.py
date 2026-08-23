from __future__ import annotations

from stock_factor.technical_transformer.data.schemas import LABEL_SCHEMA


def test_days_since_cross_is_regression_not_event() -> None:
    assert LABEL_SCHEMA.spec_for("days_since_bull_cross_5_20").task_type == "regression"
    assert LABEL_SCHEMA.spec_for("bull_cross_5_20").task_type == "binary_event"
