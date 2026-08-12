import numpy as np

from stock_factor.engine.fitness import evaluate_factor
from stock_factor.engine.validation import DataSplitConfig, build_eval_windows, build_research_split


def test_future_return_factor_has_positive_rank_ic():
    rng = np.random.default_rng(7)
    closes = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, (20, 80)), axis=1)
    forward = np.full_like(closes, np.nan)
    forward[:, :-5] = closes[:, 5:] / closes[:, :-5] - 1
    metrics = evaluate_factor(forward, closes, horizon=5)
    assert metrics["rank_ic"] > 0.9
    assert metrics["passed"] is True


def test_research_split_and_embargo_windows():
    split = build_research_split(100, DataSplitConfig(30, 10, 20), 5)
    assert split.final_oos_start == 85
    assert split.discovery_start == 55
    windows = build_eval_windows(10, 70, 3, 4)
    assert windows[1][0] - windows[0][1] == 4
    assert max(end - start for start, end in windows) - min(end - start for start, end in windows) <= 1
