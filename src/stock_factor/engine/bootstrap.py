"""Block Bootstrap（详细修改方案 P1-4）。

金融时序不是 IID：IC / Return Proxy 的置信区间必须用分块自助法。
提供 Moving Block Bootstrap 与 Stationary Bootstrap（ Politis & Romano）。
"""
from __future__ import annotations

import numpy as np


def moving_block_bootstrap(
    series: np.ndarray,
    statistic,
    block_length: int = 10,
    iterations: int = 500,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    count = values.size
    if count < max(4, block_length) or block_length < 1:
        return {"point_estimate": float(statistic(values)) if count else float("nan"), "ci_lower": None, "ci_upper": None, "iterations": 0}
    rng = np.random.default_rng(seed)
    starts_max = count - block_length + 1
    estimates = np.empty(iterations)
    for index in range(iterations):
        starts = rng.integers(0, starts_max, size=(count + block_length - 1) // block_length)
        blocks = [values[start : start + block_length] for start in starts]
        sample = np.concatenate(blocks)[:count]
        estimates[index] = statistic(sample)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return {
        "method": "moving_block",
        "point_estimate": float(statistic(values)),
        "ci_lower": round(float(lower), 8),
        "ci_upper": round(float(upper), 8),
        "confidence": confidence,
        "iterations": iterations,
        "block_length": block_length,
    }


def stationary_bootstrap(
    series: np.ndarray,
    statistic,
    expected_block_length: float = 10.0,
    iterations: int = 500,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """几何分布决定块长（均值 = expected_block_length），环形取样。"""
    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    count = values.size
    if count < 4 or expected_block_length < 1.0:
        return {"point_estimate": float(statistic(values)) if count else float("nan"), "ci_lower": None, "ci_upper": None, "iterations": 0}
    rng = np.random.default_rng(seed)
    switch_probability = 1.0 / expected_block_length
    estimates = np.empty(iterations)
    for index in range(iterations):
        sample = np.empty(count)
        position = int(rng.integers(0, count))
        for step in range(count):
            sample[step] = values[position % count]
            if rng.random() < switch_probability:
                position = int(rng.integers(0, count))
            else:
                position += 1
        estimates[index] = statistic(sample)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return {
        "method": "stationary",
        "point_estimate": float(statistic(values)),
        "ci_lower": round(float(lower), 8),
        "ci_upper": round(float(upper), 8),
        "confidence": confidence,
        "iterations": iterations,
        "expected_block_length": expected_block_length,
    }


__all__ = ["moving_block_bootstrap", "stationary_bootstrap"]
