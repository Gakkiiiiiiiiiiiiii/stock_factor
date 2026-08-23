from __future__ import annotations

from stock_factor.technical_transformer.data.labels import build_labels
from stock_factor.technical_transformer.evaluation.synthetic import synthetic_pattern


def test_synthetic_trends_and_events_are_detectable() -> None:
    assert build_labels(synthetic_pattern("linear_uptrend")).values["trend_direction"].iloc[-1] > 0
    assert build_labels(synthetic_pattern("linear_downtrend")).values["trend_direction"].iloc[-1] < 0
    assert build_labels(synthetic_pattern("spring")).values["spring_score"].max() > 0
    assert build_labels(synthetic_pattern("upthrust")).values["upthrust_score"].max() > 0
    assert build_labels(synthetic_pattern("boll_squeeze")).values["squeeze_score"].iloc[-1] > 0.8
    assert build_labels(synthetic_pattern("boll_expansion")).values["boll_expansion_score"].max() > 0.8
