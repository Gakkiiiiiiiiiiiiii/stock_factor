"""Factor Stability（详细修改方案 §6）。

Promotion Gate 不能只看 overall IC/ICIR：
必须给出 Subperiod / Regime 稳定性与符号一致性。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stock_factor.engine.fitness import rank


def _rank_ic_series(factor_panel: np.ndarray, forward_returns: np.ndarray) -> list[float]:
    symbols, days = factor_panel.shape
    ics: list[float] = []
    for day in range(days):
        factor, returns = factor_panel[:, day], forward_returns[:, day]
        valid = ~np.isnan(factor) & ~np.isnan(returns)
        if valid.sum() < 10 or np.std(factor[valid]) < 1e-12 or np.std(returns[valid]) < 1e-12:
            continue
        ics.append(float(np.corrcoef(rank(factor[valid]), rank(returns[valid]))[0, 1]))
    return ics


@dataclass(frozen=True)
class FactorStabilityReport:
    regime_metrics: dict[str, dict] = field(default_factory=dict)
    subperiod_metrics: list[dict] = field(default_factory=list)
    worst_regime_ic: float = 0.0
    ic_sign_consistency: float = 0.0
    rank_stability: float = 0.0

    def to_dict(self) -> dict:
        return {
            "regime_metrics": self.regime_metrics,
            "subperiod_metrics": self.subperiod_metrics,
            "worst_regime_ic": round(self.worst_regime_ic, 6),
            "ic_sign_consistency": round(self.ic_sign_consistency, 4),
            "rank_stability": round(self.rank_stability, 4),
        }


def _summarize(ics: list[float]) -> dict:
    if not ics:
        return {"ic_mean": 0.0, "icir": 0.0, "days": 0}
    mean = float(np.mean(ics))
    std = float(np.std(ics))
    return {"ic_mean": round(mean, 6), "icir": round(mean / std if std > 1e-12 else 0.0, 6), "days": len(ics)}


def classify_regimes(market_returns: np.ndarray, vol_window: int = 20) -> np.ndarray:
    """基于市场收益与波动划分 regime 标签（bull/bear/range/high_vol/low_vol）。"""
    returns = np.asarray(market_returns, dtype=float)
    days = returns.size
    labels = np.full(days, "range", dtype=object)
    for index in range(days):
        window = returns[max(0, index - vol_window) : index + 1]
        mean = float(np.mean(window))
        vol = float(np.std(window))
        if vol > 0.03:
            labels[index] = "high_vol"
        elif vol < 0.012:
            labels[index] = "low_vol"
        elif mean > 0.0008:
            labels[index] = "bull"
        elif mean < -0.0008:
            labels[index] = "bear"
    return labels


def factor_stability_report(
    factor_panel: np.ndarray,
    forward_returns: np.ndarray,
    market_returns: np.ndarray | None = None,
    subperiods: int = 4,
) -> FactorStabilityReport:
    ics = _rank_ic_series(factor_panel, forward_returns)
    days = len(ics)
    ic_array = np.asarray(ics)

    # Subperiod stability
    subperiod_metrics: list[dict] = []
    if days:
        chunks = np.array_split(ic_array, min(max(subperiods, 1), days))
        for index, chunk in enumerate(chunks):
            subperiod_metrics.append({"subperiod": index, **_summarize([float(item) for item in chunk])})

    # Regime stability
    regime_metrics: dict[str, dict] = {}
    if market_returns is not None and days:
        labels = classify_regimes(np.asarray(market_returns, dtype=float))
        usable = labels[:days]
        for regime in ("bull", "bear", "range", "high_vol", "low_vol"):
            mask = usable == regime
            if mask.any():
                regime_metrics[regime] = _summarize([float(item) for item in ic_array[mask]])

    worst = float(min(regime_metrics.values(), key=lambda item: item["ic_mean"])["ic_mean"]) if regime_metrics else (
        float(ic_array.min()) if days else 0.0
    )
    consistency = float(np.mean(np.sign(ic_array) == np.sign(np.mean(ic_array)))) if days and abs(float(np.mean(ic_array))) > 1e-9 else 0.0

    # Rank stability：相邻评估日 top 分位成员重合度的代理（IC 自相关）。
    rank_stability = float(np.corrcoef(ic_array[:-1], ic_array[1:])[0, 1]) if days > 2 and np.std(ic_array) > 1e-12 else 0.0

    return FactorStabilityReport(
        regime_metrics=regime_metrics,
        subperiod_metrics=subperiod_metrics,
        worst_regime_ic=worst,
        ic_sign_consistency=consistency,
        rank_stability=rank_stability,
    )


__all__ = ["FactorStabilityReport", "classify_regimes", "factor_stability_report"]
