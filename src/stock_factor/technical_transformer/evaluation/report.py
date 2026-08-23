from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .composite import technical_components
from .reliability_gate import evaluate_reliability_gate


def _score(split: dict[str, Any]) -> float | None:
    value = split.get("technical_composite")
    if value is not None:
        return float(value)
    try:
        return float(sum((0.20 * technical_components(split)["ma"], 0.20 * technical_components(split)["boll"], 0.25 * technical_components(split)["primitive"], 0.15 * technical_components(split)["phase"], 0.20 * technical_components(split)["event"])))
    except (TypeError, ValueError):
        return None


def _degradation(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    valid = splits.get("valid") or {}
    valid_score = _score(valid)
    result: dict[str, Any] = {"valid_score": valid_score}
    if valid_score in (None, 0):
        result.update({name: None for name in ("time_degradation", "instrument_degradation", "double_oos_degradation")})
    else:
        for split, output in (("time_test", "time_degradation"), ("instrument_test", "instrument_degradation"), ("double_oos", "double_oos_degradation")):
            value = _score(splits.get(split) or {})
            result[output] = None if value is None else max(0.0, 1.0 - value / valid_score)
    valid_components = technical_components(valid) if valid else {}
    group_result: dict[str, Any] = {}
    for group in ("ma", "boll", "primitive", "phase", "event"):
        item: dict[str, Any] = {"valid_score": valid_components.get(group)}
        for split, output in (("time_test", "time_degradation"), ("instrument_test", "instrument_degradation"), ("double_oos", "double_oos_degradation")):
            test_components = technical_components(splits.get(split) or {})
            base = valid_components.get(group)
            value = test_components.get(group)
            item[output] = None if base in (None, 0) or value is None else max(0.0, 1.0 - value / base)
        group_result[group] = item
    group_result["bollinger"] = group_result["boll"]
    group_result["events"] = group_result["event"]
    result["groups"] = group_result
    return result


def validate_reliability_report(report: dict[str, Any], *, production: bool | None = None) -> None:
    required = {"report_version", "mode", "dataset", "checkpoint", "data_integrity", "validation_selection", "splits", "gold_set", "embedding", "invariance", "baseline", "oos", "reliability_gate"}
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"reliability report missing required fields: {missing}")
    if report["report_version"] != "technical-reliability-report.v2":
        raise ValueError("unsupported reliability report version")
    is_production = bool(production if production is not None else str(report.get("mode")).upper() == "PRODUCTION")
    if is_production:
        for name in ("causality", "leakage", "split"):
            if name not in report["data_integrity"]:
                raise ValueError(f"production report missing data_integrity.{name}")


def build_reliability_report(
    *,
    checkpoint_identity: dict[str, Any],
    dataset_manifest: dict[str, Any],
    splits: dict[str, dict[str, Any]],
    mode: str = "PRODUCTION",
    leakage_audit: dict[str, Any] | None = None,
    causality: dict[str, Any] | None = None,
    validation_selection: dict[str, Any] | None = None,
    gold_set: dict[str, Any] | None = None,
    embedding: dict[str, Any] | None = None,
    invariance: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
    ablation: dict[str, Any] | None = None,
    gates: dict[str, Any] | None = None,
    direct_failures: list[str] | None = None,
) -> dict[str, Any]:
    normalized_mode = str(mode).upper()
    report: dict[str, Any] = {
        "report_version": "technical-reliability-report.v2", "mode": normalized_mode,
        "required_evidence": {
            "causality": normalized_mode == "PRODUCTION", "gold_set": normalized_mode == "PRODUCTION",
            "embedding_probe": normalized_mode == "PRODUCTION", "baseline": normalized_mode == "PRODUCTION",
            "invariance": normalized_mode == "PRODUCTION", "double_oos": normalized_mode == "PRODUCTION",
        },
        "dataset": {
            "dataset_id": dataset_manifest.get("dataset_id"), "snapshot_id": dataset_manifest.get("source_market_snapshot_id"),
            "feature_schema_version": dataset_manifest.get("feature_schema_version"), "label_schema_version": dataset_manifest.get("label_schema_version"),
            "split_overlap": dataset_manifest.get("split_overlap"),
        },
        "checkpoint": checkpoint_identity,
        "data_integrity": {
            "leakage": leakage_audit or dataset_manifest.get("leakage_audit", {}),
            "causality": causality or {"status": "NOT_EVALUATED", "total_violations": None},
            "split": {"overlap": dataset_manifest.get("split_overlap")},
        },
        "validation_selection": validation_selection or checkpoint_identity.get("selection", {}),
        "metric_semantics": {
            "RULE_REPRODUCTION": {"targets": ["ma", "bollinger"]},
            "SEMANTIC_GENERALIZATION": {"evidence": ["gold_set", "double_oos", "embedding", "baseline"]},
        },
        "splits": splits,
        "gold_set": gold_set or {"status": "NOT_PROVIDED"},
        "embedding": embedding or {"status": "NOT_EVALUATED"},
        "invariance": invariance or {"status": "NOT_EVALUATED"},
        "baseline": baseline or {"status": "NOT_EVALUATED"},
        "ablation": ablation or {"status": "NOT_EVALUATED"},
        "oos": _degradation(splits),
        "direct_failures": list(direct_failures or []),
    }
    # Keep the first-round aliases readable for downstream consumers.
    report["leakage_audit"] = report["data_integrity"]["leakage"]
    report["data"] = {"future_leakage_violations": report["data_integrity"]["causality"].get("total_violations"), "split_overlap": dataset_manifest.get("split_overlap")}
    report["reliability_gate"] = evaluate_reliability_gate(report, gates)
    validate_reliability_report(report)
    return report


def render_reliability_markdown(report: dict[str, Any]) -> str:
    gate = report.get("reliability_gate", {})
    lines = [
        "# Technical Transformer V1 Reliability Report", "",
        f"- Mode: **{report.get('mode', 'UNKNOWN')}**", f"- Gate: **{gate.get('status', 'NOT_EVALUATED')}**",
        f"- Dataset: `{report.get('dataset', {}).get('dataset_id', 'unknown')}`",
        f"- Checkpoint: `{report.get('checkpoint', {}).get('checkpoint_id', 'unknown')}`", "",
        "## Checks", "", "| Check | Status | Actual | Threshold |", "|---|---:|---:|---:|",
    ]
    for name, item in gate.get("checks", {}).items():
        lines.append(f"| `{name}` | {'PASS' if item.get('passed') else 'FAIL'} | {item.get('actual')} | {item.get('threshold')} |")
    lines.extend(["", "## Split Summary", ""])
    for name, value in report.get("splits", {}).items():
        lines.append(f"- `{name}`: {value.get('sample_count', 0)} samples")
    return "\n".join(lines) + "\n"


def write_reliability_report(report: dict[str, Any], output: str | Path) -> tuple[Path, Path]:
    validate_reliability_report(report)
    path = Path(output)
    if path.suffix.lower() == ".json":
        json_path = path
        markdown_path = path.with_suffix(".md")
    else:
        path.mkdir(parents=True, exist_ok=True)
        json_path = path / "reliability_report.json"
        markdown_path = path / "reliability_report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(render_reliability_markdown(report), encoding="utf-8")
    return json_path, markdown_path
