"""Cost Sensitivity（详细修改方案 §8）。

研究敏感性分析（不是 Quant 权威成交回测）：
turnover proxy + 5/10/20/50bps 成本后的代理收益 + 盈亏平衡成本。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS_PER_YEAR = 250


@dataclass(frozen=True)
class CostSensitivityReport:
    turnover_proxy: float
    return_after_5bps: float
    return_after_10bps: float
    return_after_20bps: float
    return_after_50bps: float
    break_even_cost_bps: float

    def to_dict(self) -> dict:
        return {
            "metric_scope": "RESEARCH_PROXY",
            "turnover_proxy": round(self.turnover_proxy, 6),
            "return_after_5bps": round(self.return_after_5bps, 6),
            "return_after_10bps": round(self.return_after_10bps, 6),
            "return_after_20bps": round(self.return_after_20bps, 6),
            "return_after_50bps": round(self.return_after_50bps, 6),
            "break_even_cost_bps": round(self.break_even_cost_bps, 4),
        }


def cost_sensitivity_report(
    selection_by_day: list[set[int]],
    period_returns: list[float],
    horizon: int = 5,
) -> CostSensitivityReport:
    """selection_by_day：每个调仓日选中的标的下标集合；period_returns：对应周期代理收益。"""
    if not period_returns:
        return CostSensitivityReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    turnovers: list[float] = []
    previous: set[int] | None = None
    for selection in selection_by_day:
        if previous is not None and selection:
            turnovers.append(len(selection - previous) / max(len(selection), 1))
        previous = selection
    turnover_proxy = float(np.mean(turnovers)) if turnovers else 1.0
    base = float(np.mean(period_returns)) * TRADING_DAYS_PER_YEAR / horizon

    def after(cost_bps: float) -> float:
        # 每次调仓按 turnover 比例承担单边成本（研究近似）。
        drag = cost_bps / 10_000.0 * turnover_proxy * (TRADING_DAYS_PER_YEAR / horizon)
        return base - drag

    low, high = 0.0, 500.0
    for _ in range(40):
        mid = (low + high) / 2.0
        if after(mid) > 0:
            low = mid
        else:
            high = mid
    return CostSensitivityReport(
        turnover_proxy=turnover_proxy,
        return_after_5bps=after(5.0),
        return_after_10bps=after(10.0),
        return_after_20bps=after(20.0),
        return_after_50bps=after(50.0),
        break_even_cost_bps=low,
    )


__all__ = ["CostSensitivityReport", "cost_sensitivity_report"]
