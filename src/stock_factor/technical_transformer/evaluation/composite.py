from __future__ import annotations

from typing import Any

import numpy as np

from ..data.schemas import LABEL_SCHEMA
from .metrics import event_metrics, regression_metrics, soft_phase_metrics


def _mean(items: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in items if key in item and np.isfinite(item[key])]
    return float(np.mean(values)) if values else 0.0


def _metric(group: dict[str, Any], name: str, field: str) -> float:
    value = group.get(name, {}).get(field)
    return float(value) if value is not None and np.isfinite(value) else 0.0


def evaluate_prediction_arrays(
    targets: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    label_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the one metric contract used by validation and frozen tests."""
    target_array = np.asarray(targets, dtype=float)
    if target_array.ndim != 2 or target_array.shape[1] != len(LABEL_SCHEMA.names):
        raise ValueError("targets must have shape [samples, label_dim]")
    valid_array = np.ones_like(target_array, dtype=bool) if label_valid is None else np.asarray(label_valid, dtype=bool)
    if valid_array.shape != target_array.shape:
        raise ValueError("label_valid must have the same shape as targets")
    metrics: dict[str, Any] = {}
    for group in ("ma", "bollinger", "wyckoff_primitives", "phase", "events"):
        prediction = np.asarray(predictions[group], dtype=float)
        label_slice = LABEL_SCHEMA.slices[group]
        target = target_array[:, label_slice]
        valid = valid_array[:, label_slice]
        names = getattr(LABEL_SCHEMA, group)
        if group == "phase":
            rows = valid.all(axis=1)
            metrics[group] = {
                "aggregate": soft_phase_metrics(target[rows], prediction[rows])
                if rows.any()
                else {
                    "soft_ce": 0.0,
                    "kl_divergence": 0.0,
                    "js_divergence": 0.0,
                    "brier_score": 0.0,
                    "ece": 0.0,
                    "macro_f1": 0.0,
                    "confusion_matrix": [],
                }
            }
            continue
        per_target: dict[str, Any] = {}
        for index, name in enumerate(names):
            mask = valid[:, index]
            if not mask.any():
                per_target[name] = {"valid_count": 0}
                continue
            spec = LABEL_SCHEMA.spec_for(name)
            if spec.task_type == "binary_event":
                per_target[name] = event_metrics(target[mask, index], prediction[mask, index])
            else:
                per_target[name] = regression_metrics(prediction[mask, index], target[mask, index])
            per_target[name]["valid_count"] = int(mask.sum())
        if group == "events":
            aggregate_keys = (
                "pr_auc",
                "relative_pr",
                "pr_auc_multiple_of_prevalence",
                "f1",
                "precision",
                "recall",
                "ece",
                "precision_at_top1pct",
                "precision_at_top5pct",
            )
        else:
            aggregate_keys = ("mae", "rmse", "pearson", "spearman", "sign_accuracy")
        valid_items = [value for value in per_target.values() if value.get("valid_count", 0) > 0]
        per_target["aggregate"] = {key: _mean(valid_items, key) for key in aggregate_keys}
        metrics[group] = per_target
    metrics["summary"] = summarize_metrics(metrics)
    return metrics


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    ma = metrics.get("ma", {})
    boll = metrics.get("bollinger", {})
    primitive = metrics.get("wyckoff_primitives", {})
    phase = metrics.get("phase", {}).get("aggregate", {})
    events = metrics.get("events", {})
    slope_names = LABEL_SCHEMA.ma[:6]
    primitive_names = (
        "trend_direction",
        "wyckoff_trend_strength",
        "trading_range_score",
        "support_distance",
        "resistance_distance",
        "breakout_strength",
        "breakdown_strength",
        "false_breakout_score",
        "effort_result_divergence",
    )
    event_names = LABEL_SCHEMA.events
    return {
        "ma_slope_mean_pearson": _mean([ma.get(name, {}) for name in slope_names], "pearson"),
        "ma_slope_mean_sign_accuracy": _mean([ma.get(name, {}) for name in slope_names], "sign_accuracy"),
        "ma_alignment_spearman": _mean(
            [ma.get(name, {}) for name in ("bull_alignment_score", "bear_alignment_score")], "spearman"
        ),
        "ma_compression_pearson": _metric(ma, "compression_score", "pearson"),
        "bollinger_percent_b_pearson": _metric(boll, "percent_b", "pearson"),
        "bollinger_boll_zscore_pearson": _metric(boll, "boll_zscore", "pearson"),
        "bollinger_bandwidth_pearson": _metric(boll, "bandwidth", "pearson"),
        "bollinger_squeeze_spearman": _metric(boll, "squeeze_score", "spearman"),
        "bollinger_expansion_spearman": _metric(boll, "boll_expansion_score", "spearman"),
        "wyckoff_primitive_mean_spearman": _mean([primitive.get(name, {}) for name in primitive_names], "spearman"),
        "phase_macro_f1": float(phase.get("macro_f1", 0.0)),
        "phase_js_divergence": float(phase.get("js_divergence", 0.0)),
        "phase_ece": float(phase.get("ece", 0.0)),
        "event_mean_relative_pr": _mean([events.get(name, {}) for name in event_names], "relative_pr"),
        "event_mean_f1": _mean([events.get(name, {}) for name in event_names], "f1"),
        "event_precision_at_top1pct": _mean([events.get(name, {}) for name in event_names], "precision_at_top1pct"),
        "event_precision_at_top5pct": _mean([events.get(name, {}) for name in event_names], "precision_at_top5pct"),
    }


def technical_components(metrics: dict[str, Any]) -> dict[str, float]:
    summary = metrics.get("summary", metrics)
    ma = (
        0.40 * summary.get("ma_slope_mean_pearson", 0.0)
        + 0.30 * summary.get("ma_slope_mean_sign_accuracy", 0.0)
        + 0.15 * summary.get("ma_alignment_spearman", 0.0)
        + 0.15 * summary.get("ma_compression_pearson", 0.0)
    )
    boll = (
        0.30 * summary.get("bollinger_percent_b_pearson", 0.0)
        + 0.25 * summary.get("bollinger_boll_zscore_pearson", 0.0)
        + 0.20 * summary.get("bollinger_bandwidth_pearson", 0.0)
        + 0.15 * summary.get("bollinger_squeeze_spearman", 0.0)
        + 0.10 * summary.get("bollinger_expansion_spearman", 0.0)
    )
    primitive = summary.get("wyckoff_primitive_mean_spearman", 0.0)
    phase = (
        0.60 * summary.get("phase_macro_f1", 0.0)
        + 0.20 * (1.0 - min(max(summary.get("phase_js_divergence", 0.0) / np.log(2.0), 0.0), 1.0))
        + 0.20 * (1.0 - min(max(summary.get("phase_ece", 0.0), 0.0), 1.0))
    )
    event = (
        0.60 * summary.get("event_mean_relative_pr", 0.0)
        + 0.20 * summary.get("event_mean_f1", 0.0)
        + 0.10 * summary.get("event_precision_at_top1pct", 0.0)
        + 0.10 * summary.get("event_precision_at_top5pct", 0.0)
    )
    return {
        "ma": float(ma),
        "boll": float(boll),
        "primitive": float(primitive),
        "phase": float(phase),
        "event": float(event),
    }


def technical_composite(metrics: dict[str, Any]) -> float:
    components = technical_components(metrics)
    return float(
        0.20 * components["ma"]
        + 0.20 * components["boll"]
        + 0.25 * components["primitive"]
        + 0.15 * components["phase"]
        + 0.20 * components["event"]
    )
