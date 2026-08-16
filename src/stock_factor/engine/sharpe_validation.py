"""Probabilistic / Deflated Sharpe Ratio（详细修改方案 P1-2）。

Factor 层 Sharpe 基于 research proxy（P0-1），但 DSR/PSR 对自动挖因子的
selection-bias 控制不可或缺。
"""
from __future__ import annotations

from math import erf, sqrt

import numpy as np

from stock_factor.engine.statistical_validation import deflated_sharpe_ratio


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    sample_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR：观测 Sharpe 显著高于 benchmark 的概率（Bailey & Lopez de Prado）。"""
    if sample_length <= 1:
        return 0.0
    variance = max(1e-12, 1.0 - skewness * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2)
    z = (observed_sharpe - benchmark_sharpe) * sqrt(sample_length - 1) / sqrt(variance)
    return round(float(_norm_cdf(z)), 8)


def sharpe_validation(
    returns: np.ndarray,
    number_of_trials: int = 1,
    benchmark_sharpe: float = 0.0,
    periods_per_year: float = 250.0,
) -> dict:
    """P1-2 标准输出：observed/PSR/DSR + 样本统计。"""
    series = np.asarray(returns, dtype=float)
    series = series[~np.isnan(series)]
    sample_length = int(series.size)
    if sample_length <= 1:
        return {
            "observed_sharpe": 0.0,
            "probabilistic_sharpe_ratio": 0.0,
            "deflated_sharpe_ratio": 0.0,
            "number_of_trials": int(number_of_trials),
            "skewness": 0.0,
            "kurtosis": 3.0,
            "sample_length": sample_length,
        }
    mean = float(np.mean(series))
    std = float(np.std(series, ddof=1))
    observed = mean / std if std > 1e-12 else 0.0
    skewness = float(np.mean(((series - mean) / std) ** 3)) if std > 1e-12 else 0.0
    kurtosis = float(np.mean(((series - mean) / std) ** 4)) if std > 1e-12 else 3.0
    annualized = observed * sqrt(periods_per_year)
    benchmark_per_period = benchmark_sharpe / sqrt(periods_per_year)
    psr = probabilistic_sharpe_ratio(observed, benchmark_per_period, sample_length, skewness, kurtosis)
    dsr = deflated_sharpe_ratio(observed, sample_length, trials=max(1, number_of_trials), skewness=skewness, kurtosis=kurtosis)
    return {
        "observed_sharpe": round(observed, 8),
        "annualized_sharpe": round(float(annualized), 8),
        "probabilistic_sharpe_ratio": psr,
        "deflated_sharpe_ratio": dsr,
        "number_of_trials": int(number_of_trials),
        "skewness": round(skewness, 8),
        "kurtosis": round(kurtosis, 8),
        "sample_length": sample_length,
    }


__all__ = ["probabilistic_sharpe_ratio", "sharpe_validation"]
