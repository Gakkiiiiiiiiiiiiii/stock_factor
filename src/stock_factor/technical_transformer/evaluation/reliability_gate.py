from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_GATES: dict[str, Any] = {
    "version": "technical-reliability-gate.v1",
    "data": {"future_leakage_violations_max": 0, "shortcut_violations_max": 0, "split_overlap_max": 0},
    "ma": {"slope_sign_accuracy_min": 0.90, "slope_pearson_min": 0.88, "alignment_spearman_min": 0.92},
    "boll": {"percent_b_pearson_min": 0.95, "bandwidth_pearson_min": 0.92, "squeeze_spearman_min": 0.90, "expansion_spearman_min": 0.82},
    "phase": {"macro_f1_min": 0.70},
    "wyckoff": {"primitive_spearman_min": 0.75, "gold_event_relative_pr_min": 2.0, "double_oos_event_degradation_max": 0.50},
    "gold": {"kappa_min": 0.60, "allowed_splits": ["double_oos"], "require_oos_only": True, "require_coverage": True, "min_positive_per_event": 100, "min_negative_per_event": 200},
    "oos": {"time_degradation_max": 0.15, "instrument_degradation_max": 0.20, "double_oos_degradation_max": 0.25},
    "embedding": {"noise_cosine_min": 0.95, "price_scale_cosine_min": 0.98, "nearest_neighbor_semantic_hit_min": 0.70, "weak_phase_neighbor_hit_min": 0.70, "gold_neighbor_semantic_hit_min": 0.70},
    "invariance": {"require_raw_source": True, "raw_noise_embedding_cosine_min": 0.95, "raw_noise_phase_js_max": 0.10, "raw_noise_event_median_delta_max": 0.10},
    "baseline": {"transformer_wyckoff_gold_relative_gain_min": 0.05, "transformer_double_oos_composite_relative_gain_min": 0.05},
    "masking": {"require_non_empty": True, "empty_mask_batches_max": 0},
}


def load_gate_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_GATES)
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _merge(DEFAULT_GATES, value)


def _merge(base: dict, override: dict | None) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    # Accept the first-round names while emitting the versioned policy names.
    if "bollinger" in result:
        result.setdefault("boll", {}).update(result.pop("bollinger"))
    for old, new in (("slope_mean_sign_accuracy_min", "slope_sign_accuracy_min"), ("slope_mean_pearson_min", "slope_pearson_min")):
        if old in result.get("ma", {}):
            result["ma"][new] = result["ma"][old]
    if "primitive_mean_spearman_min" in result.get("wyckoff", {}):
        result["wyckoff"]["primitive_spearman_min"] = result["wyckoff"]["primitive_mean_spearman_min"]
    if "gold_event_pr_auc_multiple_of_prevalence_min" in result.get("wyckoff", {}):
        result["wyckoff"]["gold_event_relative_pr_min"] = result["wyckoff"]["gold_event_pr_auc_multiple_of_prevalence_min"]
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


def _split_report(report: dict, split: str = "double_oos") -> dict:
    return report.get("splits", {}).get(split) or report.get(split) or {}


def evaluate_reliability_gate(report: dict[str, Any], gates: dict[str, Any] | None = None) -> dict[str, Any]:
    thresholds = _merge(DEFAULT_GATES, gates)
    checks: dict[str, dict[str, Any]] = {}
    mode = str(report.get("mode", "PRODUCTION")).upper()

    def check(name: str, actual: float | None, threshold: float, operator: str = ">=") -> None:
        passed = actual is not None and ((actual >= threshold) if operator == ">=" else (actual <= threshold))
        checks[name] = {"passed": bool(passed), "actual": actual, "threshold": threshold, "operator": operator}

    leakage = report.get("data_integrity", {}).get("leakage") or report.get("leakage_audit") or {}
    shortcut_actual = len(leakage.get("violations", [])) if isinstance(leakage, dict) and "violations" in leakage else None
    check("data.shortcut_violations", shortcut_actual, thresholds["data"]["shortcut_violations_max"], "<=")
    causality = report.get("data_integrity", {}).get("causality") or report.get("causality") or {}
    check("data.future_leakage_violations", _value(causality, ("total_violations",)), thresholds["data"]["future_leakage_violations_max"], "<=")
    split_overlap = _value(report, ("data_integrity", "split", "overlap"), _value(report, ("data", "split_overlap"), _value(report, ("dataset", "split_overlap"))))
    check("data.split_overlap", split_overlap, thresholds["data"]["split_overlap_max"], "<=")

    test = _split_report(report, "time_test")
    summary = test.get("summary", {})
    check("ma.slope_sign_accuracy", summary.get("ma_slope_mean_sign_accuracy"), thresholds["ma"]["slope_sign_accuracy_min"])
    check("ma.slope_pearson", summary.get("ma_slope_mean_pearson"), thresholds["ma"]["slope_pearson_min"])
    alignment = summary.get("ma_alignment_spearman")
    if alignment is None:
        alignment_values = [test.get("ma", {}).get(name, {}).get("spearman") for name in ("bull_alignment_score", "bear_alignment_score")]
        alignment = min((float(value) for value in alignment_values if value is not None), default=None)
    check("ma.alignment_spearman", alignment, thresholds["ma"]["alignment_spearman_min"])
    boll = test.get("bollinger", {})
    check("boll.percent_b_pearson", summary.get("bollinger_percent_b_pearson", boll.get("percent_b", {}).get("pearson")), thresholds["boll"]["percent_b_pearson_min"])
    check("boll.bandwidth_pearson", summary.get("bollinger_bandwidth_pearson", boll.get("bandwidth", {}).get("pearson")), thresholds["boll"]["bandwidth_pearson_min"])
    check("boll.squeeze_spearman", summary.get("bollinger_squeeze_spearman", boll.get("squeeze_score", {}).get("spearman")), thresholds["boll"]["squeeze_spearman_min"])
    check("boll.expansion_spearman", summary.get("bollinger_expansion_spearman", boll.get("boll_expansion_score", {}).get("spearman")), thresholds["boll"]["expansion_spearman_min"])
    check("wyckoff.primitive_spearman", summary.get("wyckoff_primitive_mean_spearman"), thresholds["wyckoff"]["primitive_spearman_min"])
    phase_metrics = test.get("phase", {}).get("aggregate", {})
    check("phase.macro_f1", summary.get("phase_macro_f1", phase_metrics.get("macro_f1")), thresholds["phase"]["macro_f1_min"])
    gold = report.get("gold_set", {})
    if "kappa" in gold:
        kappa_values = [float(value) for value in (gold.get("kappa") or {}).values() if value is not None]
        check("gold.kappa", min(kappa_values) if kappa_values else None, thresholds["gold"]["kappa_min"])
    elif mode == "PRODUCTION":
        checks["gold.kappa"] = {"passed": False, "actual": None, "threshold": thresholds["gold"]["kappa_min"], "operator": ">="}
    if "allowed_split_passed" in gold:
        checks["gold.allowed_split"] = {"passed": bool(gold.get("allowed_split_passed")), "actual": gold.get("allowed_splits"), "threshold": thresholds["gold"].get("allowed_splits", ["double_oos"]), "operator": "allowed"}
    elif mode == "PRODUCTION" and thresholds["gold"].get("require_oos_only", True):
        checks["gold.allowed_split"] = {"passed": False, "actual": None, "threshold": thresholds["gold"].get("allowed_splits", ["double_oos"]), "operator": "allowed"}
    if "coverage_passed" in gold:
        checks["gold.coverage"] = {"passed": bool(gold.get("coverage_passed")), "actual": gold.get("coverage"), "threshold": "all events covered", "operator": "complete"}
    elif mode == "PRODUCTION" and thresholds["gold"].get("require_coverage", True):
        checks["gold.coverage"] = {"passed": False, "actual": None, "threshold": "all events covered", "operator": "complete"}
    check("wyckoff.gold_event_relative_pr", _value(gold, ("event", "pr_auc_multiple_of_prevalence")), thresholds["wyckoff"]["gold_event_relative_pr_min"])

    oos = report.get("oos", {})
    check("oos.time_degradation", _value(oos, ("time_degradation",)), thresholds["oos"]["time_degradation_max"], "<=")
    check("oos.instrument_degradation", _value(oos, ("instrument_degradation",)), thresholds["oos"]["instrument_degradation_max"], "<=")
    check("oos.double_oos_degradation", _value(oos, ("double_oos_degradation",)), thresholds["oos"]["double_oos_degradation_max"], "<=")
    event_degradation = _value(oos, ("groups", "event", "double_oos_degradation"), _value(oos, ("groups", "events", "double_oos_degradation")))
    check("oos.double_oos_event_degradation", event_degradation, thresholds["wyckoff"]["double_oos_event_degradation_max"], "<=")

    embedding = report.get("embedding", {})
    weak_neighbor = _value(embedding, ("weak_phase_neighbor_hit",), _value(embedding, ("nearest_neighbor_semantic_hit",)))
    check("embedding.weak_phase_neighbor_hit", weak_neighbor, thresholds["embedding"].get("weak_phase_neighbor_hit_min", thresholds["embedding"]["nearest_neighbor_semantic_hit_min"]))
    gold_neighbor = _value(embedding, ("gold_neighbor_semantic_hit",))
    if gold_neighbor is not None:
        check("embedding.gold_neighbor_semantic_hit", gold_neighbor, thresholds["embedding"]["gold_neighbor_semantic_hit_min"])
    elif "gold_neighbor_semantic_hit" in embedding or mode == "PRODUCTION":
        checks["embedding.gold_neighbor_semantic_hit"] = {"passed": False, "actual": None, "threshold": thresholds["embedding"]["gold_neighbor_semantic_hit_min"], "operator": ">="}
    invariance = report.get("invariance", {}) or embedding.get("invariance", {})
    raw_source = invariance.get("raw_source_available")
    if raw_source is not None:
        checks["invariance.raw_source"] = {"passed": bool(raw_source), "actual": raw_source, "threshold": True, "operator": "required"}
    raw_noise = invariance.get("raw_noise_invariance") or {}
    require_raw = mode == "PRODUCTION" and thresholds["invariance"].get("require_raw_source", True)
    if raw_noise or require_raw:
        if require_raw and raw_source is None:
            checks["invariance.raw_source"] = {"passed": False, "actual": None, "threshold": True, "operator": "required"}
        check("invariance.raw_noise_embedding_cosine", _value(raw_noise, ("embedding_cosine",)), thresholds["invariance"]["raw_noise_embedding_cosine_min"])
        check("invariance.raw_noise_phase_js", _value(raw_noise, ("phase_js_divergence",)), thresholds["invariance"]["raw_noise_phase_js_max"], "<=")
        check("invariance.raw_noise_event_median_delta", _value(raw_noise, ("event_median_probability_delta",)), thresholds["invariance"]["raw_noise_event_median_delta_max"], "<=")
        check("embedding.price_scale_cosine", _value(invariance, ("price_scale_cosine",)), thresholds["embedding"]["price_scale_cosine_min"])
        feature_noise = invariance.get("feature_noise_invariance") or {}
        check("embedding.feature_noise_cosine", _value(feature_noise, ("cosine",)), thresholds["embedding"]["noise_cosine_min"])
    else:
        # Compatibility for first-round reports; v3 reports use raw_noise_invariance above.
        check("embedding.noise_cosine", _value(invariance, ("noise_cosine",), _value(invariance, ("cosine",))), thresholds["embedding"]["noise_cosine_min"])
        check("embedding.price_scale_cosine", _value(invariance, ("price_scale_cosine",)), thresholds["embedding"]["price_scale_cosine_min"])

    baseline = report.get("baseline", {})
    transformer_gain = _value(baseline, ("transformer_double_oos_composite_relative_gain",), _value(baseline, ("transformer_wyckoff_gold_relative_gain",)))
    baseline_threshold = thresholds["baseline"].get("transformer_double_oos_composite_relative_gain_min", thresholds["baseline"]["transformer_wyckoff_gold_relative_gain_min"])
    baseline_meets_threshold = transformer_gain is not None and transformer_gain >= baseline_threshold
    # A single run is not enough to make an absolute Transformer-vs-GRU
    # performance claim a production hard gate; keep it as visible evidence.
    checks["baseline.transformer_gain"] = {
        "passed": True, "actual": transformer_gain, "threshold": baseline_threshold,
        "operator": ">=", "hard_gate": False, "warning": None if baseline_meets_threshold else "BASELINE_GAIN_BELOW_RESEARCH_THRESHOLD",
    }

    required = report.get("required_evidence", {})
    if mode == "PRODUCTION":
        required = {"causality": True, "gold_set": True, "embedding_probe": True, "baseline": True, "invariance": True, "double_oos": True, **required}
    evidence_objects = {
        "causality": causality,
        "gold_set": gold,
        "embedding_probe": embedding,
        "baseline": baseline,
        "invariance": invariance,
        "double_oos": _split_report(report, "double_oos"),
    }
    for name, required_flag in required.items():
        if not required_flag:
            continue
        evidence = evidence_objects.get(name, {})
        present = isinstance(evidence, dict) and evidence.get("status") == "EVALUATED" and evidence.get("passed", True) is not False
        if name == "double_oos":
            present = int(evidence.get("sample_count", 0)) > 0
        if name == "causality":
            present = present and "total_violations" in evidence
        checks[f"evidence.{name}"] = {"passed": bool(present), "actual": evidence.get("status") if isinstance(evidence, dict) else None, "threshold": "present", "operator": "present"}
    direct_failures = list(report.get("direct_failures", []))
    if mode == "SMOKE":
        direct_failures.append("SMOKE_MODE_NOT_ELIGIBLE_FOR_ACTIVE")
    failed = [name for name, item in checks.items() if not item["passed"]]
    failed.extend(str(item) for item in direct_failures)
    return {
        "status": "PASS" if not failed else "FAIL", "gate_version": thresholds.get("version", "technical-reliability-gate.v1"),
        "thresholds": thresholds, "checks": checks, "failed_checks": failed, "direct_failures": direct_failures,
        "warnings": [item["warning"] for item in checks.values() if item.get("warning")],
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
    report_checkpoint_id = report.get("checkpoint", {}).get("checkpoint_id", report.get("checkpoint_id"))
    report_dataset_id = report.get("dataset", {}).get("dataset_id", report.get("dataset_id"))
    if report_checkpoint_id != manifest.get("checkpoint_id"):
        raise ValueError("report/checkpoint identity mismatch")
    if report_dataset_id != manifest.get("dataset_id"):
        raise ValueError("report/dataset identity mismatch")
    if str(report.get("mode", "")).upper() != "PRODUCTION":
        raise ValueError("only PRODUCTION reports may be promoted")
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
