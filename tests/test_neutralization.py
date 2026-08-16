"""Neutralization（详细修改方案 §7）。"""
from __future__ import annotations

import numpy as np

from stock_factor.engine.neutralization import neutralize_cross_section, neutralized_ic_report


def test_neutralize_removes_linear_exposure():
    rng = np.random.default_rng(4)
    market_cap = rng.normal(size=200)
    # factor 完全由市值线性生成 -> 中性化后残差接近噪声
    factor = 2.0 * market_cap + rng.normal(0.0, 1e-3, size=200)
    residuals = neutralize_cross_section(factor, market_cap)
    assert abs(np.corrcoef(residuals, market_cap)[0, 1]) < 0.05


def test_neutralized_ic_report_flags_style_only_alpha():
    rng = np.random.default_rng(6)
    stocks, days = 80, 60
    market_cap = rng.normal(size=(stocks, 1))
    market_cap_panel = np.repeat(market_cap, days, axis=1)
    # 因子 = 市值暴露；收益与市值无关 -> raw IC 偶然、中性化后归零附近
    factor = market_cap_panel + rng.normal(0.0, 0.5, size=(stocks, days))
    returns = rng.normal(0.0, 0.02, size=(stocks, days))
    report = neutralized_ic_report(factor, returns, market_cap_panel)
    assert "raw_ic" in report and "neutralized_ic" in report
    assert report["evaluated_days"] > 0
    assert abs(report["neutralized_ic"]) <= abs(report["raw_ic"]) + 0.05


def test_neutralization_preserves_genuine_alpha():
    rng = np.random.default_rng(10)
    stocks, days = 100, 80
    alpha = rng.normal(size=(stocks, days))
    market_cap = rng.normal(size=(stocks, 1))
    market_cap_panel = np.repeat(market_cap, days, axis=1)
    factor = alpha + 0.2 * market_cap_panel
    # 收益由 alpha 驱动
    returns = np.clip(alpha, -2, 2) * 0.01 + rng.normal(0.0, 0.005, size=(stocks, days))
    report = neutralized_ic_report(factor, returns, market_cap_panel)
    assert report["neutralized_ic"] > 0.1
