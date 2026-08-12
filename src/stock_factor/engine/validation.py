from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stock_factor.engine.fitness import evaluate_factor_range


@dataclass(frozen=True)
class DataSplitConfig:
    discovery_days: int = 120
    final_oos_days: int = 40
    max_warmup_days: int = 120


@dataclass(frozen=True)
class FactorResearchSplit:
    warmup_start: int
    discovery_start: int
    discovery_end: int
    final_oos_start: int
    final_oos_end: int

    def diagnostics(self, horizon: int, n_days: int) -> dict:
        return {
            "history_range": (self.warmup_start, self.discovery_start),
            "discovery_range": (self.discovery_start, self.discovery_end),
            "final_oos_range": (self.final_oos_start, self.final_oos_end),
            "future_return_observation_range": (self.final_oos_end, min(n_days, self.final_oos_end + horizon)),
            "latest_evaluable_day": max(0, n_days - horizon),
        }


def build_research_split(n_days: int, config: DataSplitConfig, horizon: int) -> FactorResearchSplit | None:
    final_end = n_days - horizon
    final_start = final_end - config.final_oos_days
    discovery_end = final_start
    discovery_start = discovery_end - config.discovery_days
    warmup_start = max(0, discovery_start - config.max_warmup_days)
    if horizon <= 0 or discovery_start <= warmup_start or final_start < discovery_end or final_end + horizon > n_days:
        return None
    return FactorResearchSplit(warmup_start, discovery_start, discovery_end, final_start, final_end)


def build_eval_windows(eval_start: int, eval_end: int, n_windows: int, embargo: int) -> list[tuple[int, int]]:
    count, gap = max(n_windows, 1), max(embargo, 0)
    usable = eval_end - eval_start - gap * (count - 1)
    if usable < count:
        return []
    base, remainder, cursor, result = usable // count, usable % count, eval_start, []
    for index in range(count):
        end = cursor + base + (1 if index < remainder else 0)
        result.append((cursor, end))
        cursor = end + gap
    return result


def run_purged_walkforward(
    factor_panel: np.ndarray,
    closes: np.ndarray,
    eval_start: int = 0,
    eval_end: int | None = None,
    horizon: int = 5,
    n_windows: int = 3,
    embargo: int = 5,
) -> dict:
    end = min(factor_panel.shape[1] - horizon, eval_end or factor_panel.shape[1] - horizon)
    windows = []
    for index, (start, stop) in enumerate(build_eval_windows(eval_start, end, n_windows, embargo)):
        metrics = evaluate_factor_range(factor_panel, closes, start, stop, horizon)
        windows.append({"window_index": index, "history_range": (0, max(eval_start, start - embargo)), "embargo_range": (max(eval_start, start - embargo), start), "test_range": (start, stop), "metrics": metrics, "passed": bool(metrics.get("passed"))})
    ranks = [float(item["metrics"].get("rank_ic", 0)) for item in windows]
    excess = [float(item["metrics"].get("topk_excess_annual_return", 0)) for item in windows]
    pass_ratio = sum(item["passed"] for item in windows) / len(windows) if windows else 0.0
    positive_ratio = sum(value > 0 for value in ranks) / len(ranks) if ranks else 0.0
    return {"method": "purged_walkforward", "windows": windows, "mean_rank_ic": round(float(np.mean(ranks)), 6) if ranks else 0.0, "min_rank_ic": round(float(np.min(ranks)), 6) if ranks else 0.0, "window_pass_ratio": round(pass_ratio, 6), "positive_window_ratio": round(positive_ratio, 6), "oos_excess_return": round(float(np.mean(excess)), 6) if excess else 0.0, "passed": bool(windows and pass_ratio >= 2 / 3 and positive_ratio >= 2 / 3 and np.mean(excess) > 0 and min(ranks) > -0.05)}
