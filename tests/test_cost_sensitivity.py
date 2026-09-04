"""Cost Sensitivity（详细修改方案 §8）。"""

from __future__ import annotations

from stock_factor.engine.cost_sensitivity import cost_sensitivity_report


def test_cost_report_fields_and_monotonic_drag():
    selections = [{0, 1, 2, 3}, {2, 3, 4, 5}, {4, 5, 6, 7}]
    returns = [0.004, 0.003, 0.005]
    report = cost_sensitivity_report(selections, returns, horizon=5)
    payload = report.to_dict()
    assert payload["metric_scope"] == "RESEARCH_PROXY"
    assert (
        payload["return_after_5bps"]
        > payload["return_after_10bps"]
        > payload["return_after_20bps"]
        > payload["return_after_50bps"]
    )
    assert payload["turnover_proxy"] > 0


def test_break_even_cost_positive_for_profitable_strategy():
    selections = [{0, 1}, {0, 1}, {0, 1}]
    report = cost_sensitivity_report(selections, [0.01, 0.01, 0.01], horizon=5)
    assert report.break_even_cost_bps > 50


def test_empty_inputs_safe():
    report = cost_sensitivity_report([], [])
    assert report.to_dict()["turnover_proxy"] == 0.0
