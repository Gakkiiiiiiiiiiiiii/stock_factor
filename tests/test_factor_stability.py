"""Factor Stability（详细修改方案 §6）。"""
from __future__ import annotations

import numpy as np

from stock_factor.engine.stability import classify_regimes, factor_stability_report


def _panels(seed: int = 12, stocks: int = 60, days: int = 200):
    rng = np.random.default_rng(seed)
    alpha = rng.normal(size=(stocks, days))
    returns = np.clip(alpha, -2, 2) * 0.01 + rng.normal(0.0, 0.005, size=(stocks, days))
    market_returns = rng.normal(0.0005, 0.02, size=days)
    return alpha, returns, market_returns


def test_stability_report_contract():
    factor, returns, market = _panels()
    report = factor_stability_report(factor, returns, market_returns=market, subperiods=4)
    payload = report.to_dict()
    assert len(payload["subperiod_metrics"]) == 4
    assert payload["regime_metrics"]
    assert 0.0 <= payload["ic_sign_consistency"] <= 1.0
    assert "worst_regime_ic" in payload and "rank_stability" in payload


def test_stable_factor_has_high_sign_consistency():
    factor, returns, market = _panels(seed=21)
    report = factor_stability_report(factor, returns, market_returns=market)
    assert report.ic_sign_consistency > 0.8
    assert report.worst_regime_ic > 0


def test_noise_factor_has_low_consistency():
    rng = np.random.default_rng(33)
    factor = rng.normal(size=(60, 150))
    returns = rng.normal(0.0, 0.01, size=(60, 150))
    report = factor_stability_report(factor, returns)
    assert report.ic_sign_consistency < 0.75
    assert report.regime_metrics == {}


def test_classify_regimes_covers_vol_extremes():
    returns = np.concatenate([np.full(50, 0.0), np.random.default_rng(1).normal(0.0, 0.06, size=50)])
    labels = classify_regimes(returns)
    assert "low_vol" in labels[:50]
    assert "high_vol" in labels[50:]
