"""metric_scope 治理（详细修改方案 P0-1）。

- fitness 产出必须标 metric_scope=RESEARCH_PROXY；
- 研究代理收益字段不得使用权威回测命名；
- Factor API 禁止输出 backtest_return / strategy_return / portfolio_return。
"""
from __future__ import annotations

import numpy as np
import pytest

from stock_factor.engine.fitness import METRIC_SCOPE, evaluate_factor_range

FORBIDDEN_AUTHORITY_FIELDS = ("backtest_return", "strategy_return", "portfolio_return")


def _panel(symbols: int = 40, days: int = 120, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    closes = 10 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=(symbols, days)), axis=1))
    factor = rng.normal(size=(symbols, days))
    return factor, closes


def test_fitness_metrics_are_research_proxy_scoped():
    factor, closes = _panel()
    metrics = evaluate_factor_range(factor, closes, 0, 100, horizon=5)
    if metrics.get("passed") is False and "metric_scope" not in metrics:
        pytest.skip("coverage too short")
    assert metrics["metric_scope"] == METRIC_SCOPE == "RESEARCH_PROXY"
    assert "research_topk_return_proxy" in metrics
    assert "research_benchmark_return_proxy" in metrics
    assert "research_excess_return_proxy" in metrics


def test_fitness_metrics_never_use_authoritative_names():
    factor, closes = _panel()
    for start, end in ((0, 100), (10, 110)):
        metrics = evaluate_factor_range(factor, closes, start, end, horizon=5)
        for forbidden in FORBIDDEN_AUTHORITY_FIELDS + (
            "topk_annual_return", "benchmark_annual_return", "topk_excess_annual_return",
        ):
            assert forbidden not in metrics


def test_alpha_api_does_not_expose_authoritative_backtest_fields(tmp_path):
    """P0-1：Factor API 输出中禁止出现权威回测字段（只允许来自 Quant backtest.v1）。"""
    from fastapi.testclient import TestClient

    from stock_factor.api.dependencies import build_application
    from stock_factor.api.main import create_app
    from tests.test_integration import FixtureContent, FixtureMarket

    application = build_application(f"sqlite:///{tmp_path / 'scope.db'}", FixtureMarket(), FixtureContent())
    client = TestClient(create_app(application))
    symbols = [f"6000{index:02d}" for index in range(20)]
    response = client.post("/api/v1/alpha/score", json={"symbols": symbols, "as_of": None, "factor_set": "production"})
    assert response.status_code == 200
    text = response.text
    for forbidden in FORBIDDEN_AUTHORITY_FIELDS:
        assert forbidden not in text
