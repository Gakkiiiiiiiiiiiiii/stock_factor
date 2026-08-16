"""DSR / PSR（详细修改方案 P1-2）。"""
from __future__ import annotations

import numpy as np

from stock_factor.engine.sharpe_validation import probabilistic_sharpe_ratio, sharpe_validation


def test_sharpe_validation_output_contract():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.001, 0.02, size=500)
    report = sharpe_validation(returns, number_of_trials=20)
    for field in (
        "observed_sharpe", "probabilistic_sharpe_ratio", "deflated_sharpe_ratio",
        "number_of_trials", "skewness", "kurtosis", "sample_length",
    ):
        assert field in report
    assert report["sample_length"] == 500
    assert report["number_of_trials"] == 20
    assert 0.0 <= report["probabilistic_sharpe_ratio"] <= 1.0
    assert 0.0 <= report["deflated_sharpe_ratio"] <= 1.0


def test_more_trials_deflate_sharpe():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0008, 0.02, size=400)
    single = sharpe_validation(returns, number_of_trials=1)
    hundred = sharpe_validation(returns, number_of_trials=100)
    assert hundred["deflated_sharpe_ratio"] < single["deflated_sharpe_ratio"]


def test_psr_monotonic_in_observed_sharpe():
    low = probabilistic_sharpe_ratio(0.01, 0.0, 250)
    high = probabilistic_sharpe_ratio(0.1, 0.0, 250)
    assert high > low


def test_empty_series_safe():
    report = sharpe_validation(np.array([]))
    assert report["sample_length"] == 0 and report["observed_sharpe"] == 0.0
