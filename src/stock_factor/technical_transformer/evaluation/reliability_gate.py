from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_GATES: dict[str, dict[str, float]] = {
    "data": {"future_leakage_violations": 0, "shortcut_violations": 0, "split_overlap": 0},
    "ma": {"slope_mean_sign_accuracy_min": 0.90, "slope_mean_pearson_min": 0.88, "alignment_spearman_min": 0.92},
    "bollinger": {"percent_b_pearson_min": 0.95, "bandwidth_pearson_min": 0.92, "squeeze_spearman_min": 0.90, "expansion_spearman_min": 0.82},
    "wyckoff": {"primitive_mean_spearman_min": 0.75, "gold_phase_macro_f1_min": 0.70, "gold_event_pr_auc_multiple_of_prevalence_min": 2.0},
    "oos": {"time_degradation_max": 0.15, "instrument_degradation_max": 0.20, "double_oos_degradation_max": 0.25},
    "embedding": {"nearest_neighbor_semantic_hit_min": 0.70, "noise_cosine_min": 0.95, "price_scale_cosine_min": 0.98},
}


def _merge(base: dict, override: dict | None) -> dict:
    result = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    return result


def _value(report: dict, path: tuple[str, ...], default: float | None = None) -> float | None:
    current: Any = report
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def _split_report(report: dict, split: str = "time_test") -> dict:
    return (report.get("splits", {}).get(split) or report.get(split) or {})


def evaluate_reliability_gate(report: dict[str, Any], gates: dict[str, Any] | None = None) -> dict[str, Any]:
    thresholds = _merge(DEFAULT_GATES, gates)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, actual: float | None, threshold: float, operator: str = ">=") -> None:
        passed = actual is not None and ((actual >= threshold) if operator == ">=" else (actual <= threshold))
        checks[name] = {"passed": passed, "actual": actual, "threshold": threshold, "operator": operator}

    leakage = report.get("leakage_audit", report.get("data", {}).get("leakage_audit", {}))
    shortcut_actual = None if not leakage or "passed" not in leakage else len(leakage.get("violations", []))
    check("data.shortcut_violations", shortcut_actual, 0, "<=")
    check("data.future_leakage_violations", _value(report, ("data", "future_leakage_violations")), 0, "<=")
    check("data.split_overlap", _value(report, ("data", "split_overlap"), _value(report, ("dataset", "split_overlap"))), 0, "<=")

    test = _split_report(report)
    summary = test.get("summary", {})
    check("ma.slope_mean_sign_accuracy", summary.get("ma_slope_mean_sign_accuracy"), thresholds["ma"]["slope_mean_sign_accuracy_min"])
    check("ma.slope_mean_pearson", summary.get("ma_slope_mean_pearson"), thresholds["ma"]["slope_mean_pearson_min"])
    alignment_values = [test.get("ma", {}).get(name, {}).get("spearman") for name in ("bull_alignment_score", "bear_alignment_score")]
    alignment = min((float(value) for value in alignment_values if value is not None), default=None)
    check("ma.alignment_spearman", alignment, thresholds["ma"]["alignment_spearman_min"])
    check("bollinger.percent_b_pearson", test.get("bollinger", {}).get("percent_b", {}).get("pearson"), thresholds["bollinger"]["percent_b_pearson_min"])
    check("bollinger.bandwidth_pearson", test.get("bollinger", {}).get("bandwidth", {}).get("pearson"), thresholds["bollinger"]["bandwidth_pearson_min"])
    check("bollinger.squeeze_spearman", test.get("bollinger", {}).get("squeeze_score", {}).get("spearman"), thresholds["bollinger"]["squeeze_spearman_min"])
    check("bollinger.expansion_spearman", test.get("bollinger", {}).get("boll_expansion_score", {}).get("spearman"), thresholds["bollinger"]["expansion_spearman_min"])
    check("wyckoff.primitive_mean_spearman", summary.get("wyckoff_primitive_mean_spearman"), thresholds["wyckoff"]["primitive_mean_spearman_min"])
    gold = report.get("gold_set", {})
    check("wyckoff.gold_phase_macro_f1", _value(gold, ("phase", "macro_f1")), thresholds["wyckoff"]["gold_phase_macro_f1_min"])
    check("wyckoff.gold_event_pr_auc_multiple", _value(gold, ("event", "pr_auc_multiple_of_prevalence")), thresholds["wyckoff"]["gold_event_pr_auc_multiple_of_prevalence_min"])

    oos = report.get("oos", {})
    check("oos.time_degradation", _value(oos, ("time_degradation",)), thresholds["oos"]["time_degradation_max"], "<=")
    check("oos.instrument_degradation", _value(oos, ("instrument_degradation",)), thresholds["oos"]["instrument_degradation_max"], "<=")
    check("oos.double_oos_degradation", _value(oos, ("double_oos_degradation",)), thresholds["oos"]["double_oos_degradation_max"], "<=")
    embedding = report.get("embedding", {})
    check("embedding.nearest_neighbor_semantic_hit", _value(embedding, ("nearest_neighbor_semantic_hit",)), thresholds["embedding"]["nearest_neighbor_semantic_hit_min"])
    check("embedding.noise_cosine", _value(embedding, ("invariance", "noise_cosine")), thresholds["embedding"]["noise_cosine_min"])
    check("embedding.price_scale_cosine", _value(embedding, ("invariance", "price_scale_cosine")), thresholds["embedding"]["price_scale_cosine_min"])

    baseline = report.get("baseline", {})
    transformer_gain = _value(baseline, ("transformer_wyckoff_gold_relative_gain",))
    checks["baseline.transformer_gain"] = {"passed": transformer_gain is not None and transformer_gain >= 0.05, "actual": transformer_gain, "threshold": 0.05, "operator": ">="}
    direct_failures = report.get("direct_failures", [])
    failed = [name for name, item in checks.items() if not item["passed"]]
    failed.extend(str(item) for item in direct_failures)
    return {
        "status": "PASS" if not failed else "FAIL", "thresholds": thresholds,
        "checks": checks, "failed_checks": failed, "direct_failures": direct_failures,
    }


_ALLOWED_TRANSITIONS = {
    "TRAINING": {"CANDIDATE"}, "CANDIDATE": {"VALIDATED", "REJECTED"},
    "VALIDATED": {"TESTED", "REJECTED"}, "TESTED": {"RELIABILITY_PASSED", "REJECTED"},
    "RELIABILITY_PASSED": {"ACTIVE", "REJECTED"}, "ACTIVE": set(), "REJECTED": set(),
}


def transition_checkpoint_status(current: str, target: str, *, gate_status: str | None = None) -> str:
    current, target = str(current), str(target)
    if target == "ACTIVE" and gate_status != "PASS":
        raise ValueError("only a PASS reliability gate may activate a checkpoint")
    if target not in _ALLOWED_TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid checkpoint transition: {current} -> {target}")
    return target


def promote_checkpoint(checkpoint_dir: str | Path, target: str, report: dict[str, Any]) -> dict[str, Any]:
    path = Path(checkpoint_dir) / "checkpoint_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    gate = report.get("reliability_gate") or evaluate_reliability_gate(report)
    current = manifest.get("checkpoint_status", "CANDIDATE")
    if target == "ACTIVE" and current == "CANDIDATE":
        for intermediate in ("VALIDATED", "TESTED", "RELIABILITY_PASSED", "ACTIVE"):
            current = transition_checkpoint_status(current, intermediate, gate_status=gate.get("status"))
        manifest["checkpoint_status"] = current
    else:
        manifest["checkpoint_status"] = transition_checkpoint_status(current, target, gate_status=gate.get("status"))
    manifest["reliability_gate"] = gate
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
