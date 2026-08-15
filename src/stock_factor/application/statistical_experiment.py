"""Cohort-level statistical validation for a factor mining experiment.

PBO and false-discovery controls are properties of a *set* of alternatives.
This module deliberately accepts the full candidate-by-date matrix instead of
allowing a caller to repeat one scalar rank-IC as a synthetic time series.
"""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np

from stock_factor.engine.statistical_validation import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)


def rank_ic_series(factor_panel: np.ndarray, closes: np.ndarray, horizon: int) -> np.ndarray:
    """Return the real cross-sectional rank-IC observed on every usable date."""
    values = np.asarray(factor_panel, dtype=float)
    prices = np.asarray(closes, dtype=float)
    if values.shape != prices.shape:
        raise ValueError("factor_panel and closes must have the same shape")
    if horizon < 1 or values.shape[1] <= horizon:
        return np.asarray([], dtype=float)

    forward = prices[:, horizon:] / prices[:, :-horizon] - 1
    observations: list[float] = []
    for day in range(forward.shape[1]):
        factor, returns = values[:, day], forward[:, day]
        valid = np.isfinite(factor) & np.isfinite(returns)
        if valid.sum() < 10:
            continue
        factor_ranks = np.argsort(np.argsort(factor[valid], kind="mergesort"), kind="mergesort")
        return_ranks = np.argsort(np.argsort(returns[valid], kind="mergesort"), kind="mergesort")
        if np.std(factor_ranks) > 0 and np.std(return_ranks) > 0:
            observations.append(float(np.corrcoef(factor_ranks, return_ranks)[0, 1]))
    return np.asarray(observations, dtype=float)


def validate_statistical_experiment(
    candidate_series: dict[str, np.ndarray],
    *,
    alpha: float = 0.05,
    min_deflated_sharpe: float = 0.5,
    max_pbo: float = 0.35,
) -> dict[str, dict]:
    """Validate every candidate jointly and return auditable per-candidate data.

    A one-candidate cohort is explicitly invalid for PBO.  It remains
    persisted as a candidate, but cannot pass a promotion gate until it has
    been evaluated together with an independent cohort.
    """
    keys = list(candidate_series)
    aligned = [np.asarray(candidate_series[key], dtype=float) for key in keys]
    usable_lengths = [len(series) for series in aligned if len(series)]
    if not usable_lengths:
        return {
            key: {
                "raw_p_value": 1.0,
                "adjusted_p_value": 1.0,
                "q_value": 1.0,
                "pbo": 1.0,
                "effective_trials": 0,
                "passed_multiple_testing": False,
                "passed_pbo": False,
                "passed": False,
                "reject_reasons": ["INSUFFICIENT_TIME_SERIES"],
            }
            for key in keys
        }

    length = min(usable_lengths)
    matrix = np.vstack([series[-length:] if len(series) >= length else np.full(length, np.nan) for series in aligned])
    raw_p_values: list[float] = []
    sharpes: list[float] = []
    for series in matrix:
        finite = series[np.isfinite(series)]
        if len(finite) < 2:
            raw_p_values.append(1.0)
            sharpes.append(0.0)
            continue
        mean = float(np.mean(finite))
        deviation = float(np.std(finite, ddof=1))
        z_score = abs(mean) * sqrt(len(finite)) / max(deviation, 1e-12)
        raw_p_values.append(float(erfc(z_score / sqrt(2))))
        sharpes.append(mean / max(deviation, 1e-12))

    fdr = benjamini_hochberg(raw_p_values, alpha)
    pbo_matrix = matrix[np.isfinite(matrix).sum(axis=1) >= 2]
    pbo = probability_of_backtest_overfitting(pbo_matrix)
    effective_trials = len(pbo_matrix)
    results: dict[str, dict] = {}
    for index, key in enumerate(keys):
        adjusted = float(fdr["adjusted_p_values"][index])
        dsr = deflated_sharpe_ratio(sharpes[index], length, effective_trials)
        reasons: list[str] = []
        if not fdr["rejected"][index]:
            reasons.append("FDR_FAILED")
        if dsr < min_deflated_sharpe:
            reasons.append("DEFLATED_SHARPE_FAILED")
        if effective_trials < 2:
            reasons.append("PBO_REQUIRES_COHORT")
        elif pbo > max_pbo:
            reasons.append("PBO_FAILED")
        results[key] = {
            "raw_p_value": round(raw_p_values[index], 10),
            "adjusted_p_value": round(adjusted, 10),
            "q_value": round(adjusted, 10),
            "deflated_sharpe": dsr,
            "pbo": pbo,
            "effective_trials": effective_trials,
            "observations": length,
            "passed_multiple_testing": bool(fdr["rejected"][index]),
            "passed_pbo": effective_trials >= 2 and pbo <= max_pbo,
            "passed": not reasons,
            "reject_reasons": reasons,
            "method": "cohort_bh_cscv_v1",
        }
    return results
