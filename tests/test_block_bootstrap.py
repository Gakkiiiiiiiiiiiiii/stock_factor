"""Block Bootstrap（详细修改方案 P1-4）。"""
from __future__ import annotations

import numpy as np

from stock_factor.engine.bootstrap import moving_block_bootstrap, stationary_bootstrap


def _series(seed: int = 3, size: int = 300) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.05, 1.0, size=size)


def test_moving_block_bootstrap_deterministic_and_ordered():
    series = _series()
    first = moving_block_bootstrap(series, np.mean, block_length=15, iterations=200, seed=42)
    second = moving_block_bootstrap(series, np.mean, block_length=15, iterations=200, seed=42)
    assert first == second
    assert first["ci_lower"] <= first["point_estimate"] <= first["ci_upper"]
    assert first["method"] == "moving_block"


def test_stationary_bootstrap_covers_mean():
    series = _series(seed=8)
    report = stationary_bootstrap(series, np.mean, expected_block_length=12.0, iterations=200, seed=1)
    assert report["ci_lower"] is not None and report["ci_lower"] < report["ci_upper"]
    assert report["method"] == "stationary"


def test_block_bootstrap_wider_ci_than_iid_for_autocorrelated_series():
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 1.0, size=400)
    autocorr = np.array([base[0]] + [0.9 * autocorr_value + 0.1 * base_value for autocorr_value, base_value in zip(base[:-1], base[1:])])
    block = moving_block_bootstrap(autocorr, np.mean, block_length=20, iterations=300, seed=0)
    assert block["ci_upper"] - block["ci_lower"] > 0


def test_short_series_degrades_safely():
    report = moving_block_bootstrap(np.array([1.0, 2.0]), np.mean, block_length=10)
    assert report["iterations"] == 0
