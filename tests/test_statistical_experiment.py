import numpy as np

from stock_factor.application.statistical_experiment import rank_ic_series, validate_statistical_experiment


def test_single_candidate_is_not_a_valid_pbo_experiment():
    result = validate_statistical_experiment({"only": np.array([0.1, 0.2, 0.15, 0.18])})
    assert result["only"]["passed_pbo"] is False
    assert "PBO_REQUIRES_COHORT" in result["only"]["reject_reasons"]


def test_rank_ic_series_uses_observed_dates_not_a_repeated_scalar():
    values = np.arange(10, dtype=float)[:, None] * np.ones((1, 4))
    base = np.arange(10, dtype=float)
    day_0 = np.full(10, 100.0)
    day_1 = day_0 + base
    day_2 = day_1 * (1 + (9 - base) / 100)
    day_3 = day_2 * (1 + base / 100)
    closes = np.column_stack([day_0, day_1, day_2, day_3])
    series = rank_ic_series(values, closes, horizon=1)
    assert len(series) == 3
    assert not np.all(series == series[0])
