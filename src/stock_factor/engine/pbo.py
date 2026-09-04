"""Probability of Backtest Overfitting（详细修改方案 P1-3）。

CSCV（Combinatorially Symmetric Cross-Validation）：
从大量候选中挑出的最好因子有多大概率只是过拟合。
"""

from __future__ import annotations

import numpy as np

from stock_factor.engine.statistical_validation import probability_of_backtest_overfitting


def pbo_report(returns_by_trial: np.ndarray, partitions: int = 8, max_pbo: float = 0.35) -> dict:
    """P1-3 标准输出：PBO + 输入规模 + 阈值判定。"""
    values = np.asarray(returns_by_trial, dtype=float)
    pbo = probability_of_backtest_overfitting(values, partitions=partitions)
    trials, periods = values.shape if values.ndim == 2 else (0, 0)
    return {
        "pbo": pbo,
        "trials": int(trials),
        "periods": int(periods),
        "partitions": int(partitions),
        "max_pbo": max_pbo,
        "passed": bool(pbo <= max_pbo),
    }


__all__ = ["pbo_report"]
