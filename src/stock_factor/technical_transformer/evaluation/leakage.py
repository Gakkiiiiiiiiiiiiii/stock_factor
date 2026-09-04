from __future__ import annotations

from typing import Sequence

import numpy as np

from .metrics import pearson, spearman


def _affine_r2(feature: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(feature) & np.isfinite(target)
    x, y = feature[mask], target[mask]
    if len(x) < 3 or np.var(x) < 1e-15:
        return 0.0, 0.0, float(np.mean(y)) if len(y) else 0.0
    design = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = slope * x + intercept
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / denominator if denominator > 1e-15 else 0.0
    return float(r2), float(slope), float(intercept)


def audit_shortcut_leakage(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    *,
    pearson_threshold: float = 0.9999,
    r2_threshold: float = 0.9999,
) -> dict:
    """Audit every feature/target pair for direct or affine shortcuts."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != len(feature_names)
        or y.shape[1] != len(target_names)
        or x.shape[0] != y.shape[0]
    ):
        raise ValueError("features/targets and names have incompatible shapes")
    pairs = []
    violations = []
    for feature_index, feature_name in enumerate(feature_names):
        for target_index, target_name in enumerate(target_names):
            feature = x[:, feature_index]
            target = y[:, target_index]
            mask = np.isfinite(feature) & np.isfinite(target)
            equality = float(np.mean(feature[mask] == target[mask])) if mask.any() else 0.0
            r2, slope, intercept = _affine_r2(feature, target)
            item = {
                "feature": str(feature_name),
                "target": str(target_name),
                "pearson": pearson(feature, target),
                "spearman": spearman(feature, target),
                "exact_equality_rate": equality,
                "affine_fit_r2": r2,
                "affine_slope": slope,
                "affine_intercept": intercept,
            }
            pairs.append(item)
            if abs(item["pearson"]) > pearson_threshold and item["affine_fit_r2"] > r2_threshold:
                violations.append(item)
    return {
        "passed": not violations,
        "thresholds": {"abs_pearson_max": pearson_threshold, "affine_fit_r2_max": r2_threshold},
        "violations": violations,
        "pair_metrics": pairs,
    }


def assert_no_shortcut_leakage(*args, **kwargs) -> dict:
    result = audit_shortcut_leakage(*args, **kwargs)
    if not result["passed"]:
        conflicts = ", ".join(f"{item['feature']}->{item['target']}" for item in result["violations"])
        raise AssertionError(f"shortcut leakage detected: {conflicts}")
    return result
