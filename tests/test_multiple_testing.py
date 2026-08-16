"""Multiple Testing（详细修改方案 P1-1）。"""
from __future__ import annotations

import pytest

from stock_factor.engine.multiple_testing import (
    TrialRegistry,
    benjamini_hochberg_fdr,
    bonferroni,
    correct_multiple_testing,
    holm,
)


def test_bonferroni_penalizes_family_size():
    result = bonferroni([0.01, 0.04, 0.5], alpha=0.05)
    assert result["adjusted_p_values"][0] == 0.03
    assert result["rejected"] == [True, False, False]


def test_holm_no_weaker_than_bonferroni():
    p_values = [0.001, 0.02, 0.03, 0.6]
    bon = bonferroni(p_values)["adjusted_p_values"]
    hlm = holm(p_values)["adjusted_p_values"]
    assert all(h <= b + 1e-12 for h, b in zip(hlm, bon))
    assert holm(p_values)["rejected"][0] is True


def test_bh_fdr_controls_false_discovery():
    result = benjamini_hochberg_fdr([0.001, 0.008, 0.03, 0.2, 0.9], alpha=0.05)
    assert result["family_pass"] is True
    assert result["rejected"][:2] == [True, True]


def test_registry_requires_full_p_value_family():
    with pytest.raises(ValueError):
        correct_multiple_testing(TrialRegistry(hypothesis_count=10))


def test_p_value_boundaries_are_validated_and_clamped():
    # 越界 p-value 直接拒绝。
    with pytest.raises(ValueError):
        bonferroni([-0.1, 0.5])
    with pytest.raises(ValueError):
        holm([0.5, 1.2])
    # 边界 0 / 1：校正结果仍落在 [0, 1]。
    for result in (bonferroni([0.0, 1.0, 0.5]), holm([0.0, 1.0, 0.5]), benjamini_hochberg_fdr([0.0, 1.0, 0.5])):
        assert all(0.0 <= item <= 1.0 for item in result["adjusted_p_values"])
    assert bonferroni([0.0])["rejected"] == [True]
    assert bonferroni([1.0])["rejected"] == [False]


def test_corrected_results_are_stable_and_deterministic():
    # 相同输入 → 相同输出（deterministic），且与单方法结果一致（corrected result 稳定）。
    p_values = [0.001, 0.02, 0.4, 0.8]
    registry_a = TrialRegistry(hypothesis_count=40, candidate_count=30, tested_factor_count=30, p_values=list(p_values))
    registry_b = TrialRegistry(hypothesis_count=40, candidate_count=30, tested_factor_count=30, p_values=list(p_values))
    first = correct_multiple_testing(registry_a)
    second = correct_multiple_testing(registry_b)
    assert first == second
    assert first["methods"]["bonferroni"] == bonferroni(p_values)
    assert first["methods"]["holm"] == holm(p_values)
    assert first["methods"]["benjamini_hochberg"] == benjamini_hochberg_fdr(p_values)


def test_correct_multiple_testing_records_trial_counts():
    registry = TrialRegistry(hypothesis_count=50, candidate_count=30, tested_factor_count=30, p_values=[0.001, 0.4, 0.8])
    result = correct_multiple_testing(registry)
    assert result["trials"]["hypothesis_count"] == 50
    assert result["trials"]["effective_trials"] == 50
    assert result["p_value_count"] == 3
    assert result["passed_fdr"] is True
