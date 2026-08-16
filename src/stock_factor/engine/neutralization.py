"""Neutralization（详细修改方案 P1-5 / §7）。

行业 / 市值 / Beta / 波动率 / 流动性中性化：
截面回归剔除暴露后重新计算 IC，避免把风格暴露误判为 Alpha。
每个评估必须同时记录 raw_ic 与 neutralized_ic。
"""
from __future__ import annotations

import numpy as np

SUPPORTED_EXPOSURES = ("industry", "market_cap", "beta", "volatility", "liquidity")


def _demean(matrix: np.ndarray) -> np.ndarray:
    means = np.nanmean(matrix, axis=0)
    return matrix - means


def neutralize_cross_section(factor_day: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """单日截面：factor 对 exposures 做 OLS，返回残差（NaN 安全）。

    exposures: (n_stocks, n_exposures)；industry 需调用方先做 one-hot。
    """
    factor = np.asarray(factor_day, dtype=float)
    matrix = np.asarray(exposures, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    valid = ~np.isnan(factor) & ~np.isnan(matrix).any(axis=1)
    if valid.sum() < max(3, matrix.shape[1] + 2):
        return factor
    x = _demean(matrix[valid])
    y = factor[valid] - np.nanmean(factor[valid])
    gram = x.T @ x
    try:
        coefficients = np.linalg.solve(gram + np.eye(matrix.shape[1]) * 1e-10, x.T @ y)
    except np.linalg.LinAlgError:
        return factor
    residuals = y - x @ coefficients
    output = np.full_like(factor, np.nan)
    output[valid] = residuals + np.nanmean(factor[valid])
    return output


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() < 10:
        return None
    x, y = a[valid], b[valid]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    rank_x = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
    rank_y = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(float)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def neutralized_ic_report(
    factor_panel: np.ndarray,
    forward_returns: np.ndarray,
    exposure_panel: np.ndarray,
) -> dict:
    """逐日中性化后计算 IC；输出 raw_ic / neutralized_ic。"""
    factor = np.asarray(factor_panel, dtype=float)
    returns = np.asarray(forward_returns, dtype=float)
    exposures = np.asarray(exposure_panel, dtype=float)
    days = factor.shape[1]
    raw_ics: list[float] = []
    neutralized_ics: list[float] = []
    for day in range(days):
        raw = _spearman(factor[:, day], returns[:, day])
        if raw is None:
            continue
        neutralized_factor = neutralize_cross_section(factor[:, day], exposures[:, day] if exposures.ndim == 3 else exposures)
        neutralized = _spearman(neutralized_factor, returns[:, day])
        if neutralized is None:
            continue
        raw_ics.append(raw)
        neutralized_ics.append(neutralized)
    raw_ic = float(np.mean(raw_ics)) if raw_ics else 0.0
    neutralized_ic = float(np.mean(neutralized_ics)) if neutralized_ics else 0.0
    return {
        "raw_ic": round(raw_ic, 6),
        "neutralized_ic": round(neutralized_ic, 6),
        "ic_retained_ratio": round(neutralized_ic / raw_ic, 6) if abs(raw_ic) > 1e-9 else None,
        "evaluated_days": len(raw_ics),
    }


__all__ = ["SUPPORTED_EXPOSURES", "neutralize_cross_section", "neutralized_ic_report"]
