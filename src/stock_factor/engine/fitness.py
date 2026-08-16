from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS_PER_YEAR = 250
TURNOVER_COST = 0.001

# 详细修改方案 P0-1：本模块产出均为研究代理指标，不是 Quant 权威回测。
METRIC_SCOPE = "RESEARCH_PROXY"


@dataclass(frozen=True)
class EvaluationThresholds:
    min_coverage: float = 0.6
    min_rank_ic: float = 0.02
    min_icir: float = 0.3
    min_topk_excess_annual_return: float = 0.0


def rank(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    valid = ~np.isnan(values)
    count = int(valid.sum())
    if count:
        order = np.argsort(values[valid], kind="mergesort")
        ranked = np.empty(count)
        ranked[order] = (np.arange(count) + 1) / count
        out[valid] = ranked
    return out


def evaluate_factor(
    factor_panel: np.ndarray,
    closes: np.ndarray,
    horizon: int = 5,
    top_k: int | None = None,
    eval_window: int | None = None,
    thresholds: EvaluationThresholds | None = None,
) -> dict:
    # 全窗口评估：逻辑边界 = 面板长度，horizon 截断由 range 内部处理。
    end = factor_panel.shape[1]
    start = max(0, end - horizon - eval_window) if eval_window else 0
    return evaluate_factor_range(factor_panel, closes, start, end, horizon, top_k, thresholds)


def evaluate_factor_range(
    factor_panel: np.ndarray,
    closes: np.ndarray,
    eval_start: int,
    eval_end: int,
    horizon: int = 5,
    top_k: int | None = None,
    thresholds: EvaluationThresholds | None = None,
) -> dict:
    thresholds = thresholds or EvaluationThresholds()
    symbols, days = factor_panel.shape
    start, end = max(0, eval_start), min(eval_end, days - horizon)
    if end <= start:
        return {"passed": False, "coverage": 0.0, "fitness": float("-inf"), "warning": "eval range too short"}
    forward = np.full((symbols, days), np.nan)
    # 收尾文档 §21：forward return 不得越过评估窗口边界（否则 discovery 评估
    # 会读到 Final OOS 数据，构成未来函数）：仅当 t + horizon < eval_end 时有效。
    limit = max(0, min(days - horizon, eval_end - horizon))
    if limit > 0:
        forward[:, :limit] = closes[:, horizon : horizon + limit] / closes[:, :limit] - 1
    ics, rank_ics, periods = [], [], []
    selected = top_k or max(5, int(symbols * 0.01))
    last_day = start - horizon
    previous: set[int] | None = None
    for day in range(start, end):
        factor, returns = factor_panel[:, day], forward[:, day]
        valid = ~np.isnan(factor) & ~np.isnan(returns)
        if valid.sum() < 10 or np.std(factor[valid]) < 1e-12 or np.std(returns[valid]) < 1e-12:
            continue
        ics.append(float(np.corrcoef(factor[valid], returns[valid])[0, 1]))
        rank_ics.append(float(np.corrcoef(rank(factor[valid]), rank(returns[valid]))[0, 1]))
        if day - last_day < horizon:
            continue
        last_day = day
        indexes = np.where(valid)[0]
        chosen = set(indexes[np.argsort(factor[valid])[-min(selected, int(valid.sum())) :]].tolist())
        turnover = 1.0 if previous is None else len(chosen - previous) / max(len(chosen), 1)
        periods.append(
            (float(np.mean(forward[list(chosen), day])) - TURNOVER_COST * turnover, float(np.mean(returns[valid])))
        )
        previous = chosen
    coverage = len(ics) / (end - start)
    if coverage < thresholds.min_coverage or not rank_ics:
        return {
            "rank_ic": 0.0,
            "ic_mean": 0.0,
            "icir": 0.0,
            "coverage": round(coverage, 4),
            "fitness": float("-inf"),
            "passed": False,
            "evaluated_days": end - start,
            "valid_ic_days": len(ics),
        }
    rank_ic, ic_mean = float(np.mean(rank_ics)), float(np.mean(ics))
    deviation = float(np.std(rank_ics))
    icir = rank_ic / (deviation if deviation > 1e-12 else 0.01)
    annual = float(np.mean([item[0] for item in periods]) * TRADING_DAYS_PER_YEAR / horizon) if periods else 0.0
    benchmark = float(np.mean([item[1] for item in periods]) * TRADING_DAYS_PER_YEAR / horizon) if periods else 0.0
    excess = annual - benchmark
    passed = (
        rank_ic >= thresholds.min_rank_ic
        and icir >= thresholds.min_icir
        and excess > thresholds.min_topk_excess_annual_return
    )
    return {
        "metric_scope": METRIC_SCOPE,
        "rank_ic": round(rank_ic, 4),
        "ic_mean": round(ic_mean, 4),
        "icir": round(icir, 4),
        "coverage": round(coverage, 4),
        # P0-1：研究代理收益改名，避免与 Quant backtest.v1 权威指标混淆。
        "research_topk_return_proxy": round(annual, 4),
        "research_benchmark_return_proxy": round(benchmark, 4),
        "research_excess_return_proxy": round(excess, 4),
        "fitness": round(5 * rank_ic + 0.5 * icir + excess, 4),
        "evaluated_days": end - start,
        "valid_ic_days": len(ics),
        "top_k": selected,
        "passed": bool(passed),
    }
