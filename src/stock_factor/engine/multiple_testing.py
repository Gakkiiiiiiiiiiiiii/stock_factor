"""Multiple Testing Correction（详细修改方案 P1-1）。

每次 Mining Experiment 必须记录 hypothesis_count / candidate_count / tested_factor_count，
禁止只保存获胜因子的 p-value。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stock_factor.engine.statistical_validation import benjamini_hochberg


def bonferroni(p_values: list[float], alpha: float = 0.05) -> dict:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be values in [0, 1]")
    count = len(values)
    adjusted = np.minimum(values * max(count, 1), 1.0)
    rejected = adjusted <= alpha
    return {
        "method": "bonferroni",
        "adjusted_p_values": [round(float(item), 10) for item in adjusted],
        "rejected": [bool(item) for item in rejected],
        "family_pass": bool(np.any(rejected)),
    }


def holm(p_values: list[float], alpha: float = 0.05) -> dict:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be values in [0, 1]")
    count = len(values)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    adjusted = np.maximum.accumulate(np.minimum(ordered * (count - np.arange(count)), 1.0))
    expanded = np.empty(count, dtype=float)
    expanded[order] = adjusted
    rejected = expanded <= alpha
    return {
        "method": "holm",
        "adjusted_p_values": [round(float(item), 10) for item in expanded],
        "rejected": [bool(item) for item in rejected],
        "family_pass": bool(np.any(rejected)),
    }


def benjamini_hochberg_fdr(p_values: list[float], alpha: float = 0.05) -> dict:
    result = benjamini_hochberg(p_values, alpha)
    return {
        "method": "benjamini_hochberg",
        "adjusted_p_values": result["adjusted_p_values"],
        "rejected": result["rejected"],
        "family_pass": result["fdr_pass"],
    }


METHODS = {
    "bonferroni": bonferroni,
    "holm": holm,
    "benjamini_hochberg": benjamini_hochberg_fdr,
}


@dataclass
class TrialRegistry:
    """P1-1：Mining 试验计数，随实验结果一起落库。"""

    hypothesis_count: int = 0
    candidate_count: int = 0
    tested_factor_count: int = 0
    p_values: list[float] = field(default_factory=list)

    def effective_trials(self) -> int:
        return max(self.hypothesis_count, self.candidate_count, self.tested_factor_count, len(self.p_values), 1)

    def to_dict(self) -> dict:
        return {
            "hypothesis_count": self.hypothesis_count,
            "candidate_count": self.candidate_count,
            "tested_factor_count": self.tested_factor_count,
            "effective_trials": self.effective_trials(),
        }


def correct_multiple_testing(
    registry: TrialRegistry,
    alpha: float = 0.05,
    methods: tuple[str, ...] = ("bonferroni", "holm", "benjamini_hochberg"),
) -> dict:
    """对试验族内全部 p-value 做校正；只保存获胜因子 p-value 的做法被结构性禁止。"""
    if not registry.p_values:
        raise ValueError("multiple testing correction requires the full p-value family (禁止只保存获胜因子 p-value)")
    results = {method: METHODS[method](registry.p_values, alpha) for method in methods}
    return {
        "alpha": alpha,
        "trials": registry.to_dict(),
        "p_value_count": len(registry.p_values),
        "methods": results,
        "passed_fdr": results["benjamini_hochberg"]["family_pass"] if "benjamini_hochberg" in results else None,
    }


__all__ = [
    "bonferroni",
    "holm",
    "benjamini_hochberg_fdr",
    "METHODS",
    "TrialRegistry",
    "correct_multiple_testing",
]
