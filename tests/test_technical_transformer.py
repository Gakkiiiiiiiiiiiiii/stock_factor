# ruff: noqa: I001
from __future__ import annotations

import numpy as np
import pytest


pd = pytest.importorskip("pandas")

from stock_factor.technical_transformer.data.dataset import (  # noqa: E402
    DatasetConfig,
    RobustFeatureProcessor,
    SplitConfig,
    _eligible_samples,
)  # noqa: E402
from stock_factor.technical_transformer.data.features import build_features  # noqa: E402
from stock_factor.technical_transformer.data.labels import build_labels  # noqa: E402
from stock_factor.technical_transformer.data.schemas import ALL_LABELS, CONTINUOUS_FEATURES, FEATURE_NAMES, LABEL_SCHEMA  # noqa: E402


def _bars(rows: int = 320) -> pd.DataFrame:
    dates = pd.date_range("2023-01-03", periods=rows, freq="B")
    close = 10 + np.cumsum(np.sin(np.arange(rows) / 9) * 0.05 + 0.02)
    return pd.DataFrame(
        {
            "trading_date": dates,
            "symbol": "AAA.SZ",
            "open": close - 0.03,
            "high": close + 0.08,
            "low": close - 0.08,
            "close": close,
            "volume": 100000 + np.arange(rows) * 10,
            "amount": (100000 + np.arange(rows) * 10) * close,
            "turnover": 0.01 + np.arange(rows) * 0.000001,
            "is_suspended": 0.0,
            "is_st": 0.0,
            "is_star_st": 0.0,
            "is_delisting": 0.0,
            "is_limit_up": 0.0,
            "is_limit_down": 0.0,
            "state_observed": 1.0,
            "listing_days": np.arange(1, rows + 1),
        }
    )


def test_feature_and_label_schemas_are_versioned_and_unique() -> None:
    assert len(CONTINUOUS_FEATURES) == 36
    assert len(FEATURE_NAMES) == 51
    assert len(ALL_LABELS) == 61
    assert "close_sma20" not in FEATURE_NAMES
    assert LABEL_SCHEMA.version == "technical-label.v2"
    assert len(set(ALL_LABELS)) == len(ALL_LABELS)
    assert len(LABEL_SCHEMA.names) == len(set(LABEL_SCHEMA.names))


def test_labels_do_not_read_future_rows() -> None:
    bars = _bars()
    cutoff = 220
    baseline = build_labels(bars).iloc[:cutoff]
    changed = bars.copy()
    changed.loc[cutoff:, ["open", "high", "low", "close", "volume", "amount", "turnover"]] *= 1000
    altered = build_labels(changed).iloc[:cutoff]
    np.testing.assert_allclose(baseline.to_numpy(), altered.to_numpy(), rtol=1e-6, atol=1e-6)


def test_split_window_does_not_cross_segment_start() -> None:
    bars = _bars()
    split = SplitConfig(
        train_start="2023-01-03",
        train_end="2023-12-29",
        valid_start="2024-03-01",
        valid_end="2024-09-30",
        test_start="2024-12-02",
        test_end="2024-12-31",
    )
    config = DatasetConfig(split=split, min_listing_days=1, min_quality_ratio=0.0)
    samples = _eligible_samples(bars["trading_date"], bars["listing_days"], np.ones(len(bars)), config)
    assert samples
    for item in samples:
        if item["split"] == "valid":
            assert bars["trading_date"].iloc[item["start_index"]] >= pd.Timestamp(split.valid_start)
        if item["split"] == "test":
            assert bars["trading_date"].iloc[item["start_index"]] >= pd.Timestamp(split.test_start)


def test_processor_fit_scope_is_explicit() -> None:
    processor = RobustFeatureProcessor().fit(np.ones((20, len(CONTINUOUS_FEATURES)), dtype=np.float32))
    assert processor.as_dict()["fit_scope"] == "train"
    assert processor.transform(np.zeros((2, len(CONTINUOUS_FEATURES)), dtype=np.float32)).shape == (
        2,
        len(CONTINUOUS_FEATURES),
    )


def test_features_keep_suspension_quality_mask() -> None:
    bars = _bars()
    bars.loc[100, ["open", "high", "low", "close"]] = np.nan
    bars.loc[100, "volume"] = 0
    features = build_features(bars)
    assert features.loc[100, "quality_mask"] == 0
