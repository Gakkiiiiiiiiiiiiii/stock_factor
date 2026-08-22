from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .reliability_gate import evaluate_reliability_gate


def _degradation(splits: dict[str, dict], metric: str = "ma_slope_mean_pearson") -> dict[str, float | None]:
    baseline = (splits.get("valid") or {}).get("summary", {}).get(metric)
    if baseline in (None, 0):
        return {name: None for name in ("time_degradation", "instrument_degradation", "double_oos_degradation")}
    result = {}
    for split, output in (("time_test", "time_degradation"), ("instrument_test", "instrument_degradation"), ("double_oos", "double_oos_degradation")):
        value = (splits.get(split) or {}).get("summary", {}).get(metric)
        result[output] = None if value is None else max(0.0, 1.0 - float(value) / float(baseline))
    return result


def build_reliability_report(
    *,
    checkpoint_identity: dict[str, Any],
    dataset_manifest: dict[str, Any],
    splits: dict[str, dict[str, Any]],
    leakage_audit: dict[str, Any] | None = None,
    gold_set: dict[str, Any] | None = None,
    embedding: dict[str, Any] | None = None,
    invariance: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
    gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "report_version": "technical-reliability-report.v1",
        "dataset": {"dataset_id": dataset_manifest.get("dataset_id"), "snapshot_id": dataset_manifest.get("source_market_snapshot_id"), "feature_schema_version": dataset_manifest.get("feature_schema_version"), "label_schema_version": dataset_manifest.get("label_schema_version"), "split_overlap": dataset_manifest.get("split_overlap")},
        "checkpoint": checkpoint_identity,
        "leakage_audit": leakage_audit or dataset_manifest.get("leakage_audit", {}),
        "splits": splits,
        "gold_set": gold_set or {"status": "NOT_PROVIDED"},
        "embedding": embedding or {"status": "NOT_EVALUATED", "invariance": invariance or {}},
        "baseline": baseline or {"status": "NOT_EVALUATED"},
        "oos": _degradation(splits),
        "data": {"future_leakage_violations": None, "split_overlap": dataset_manifest.get("split_overlap")},
    }
    report["invariance"] = invariance or {"status": "NOT_EVALUATED"}
    report["reliability_gate"] = evaluate_reliability_gate(report, gates)
    return report


def render_reliability_markdown(report: dict[str, Any]) -> str:
    gate = report.get("reliability_gate", {})
    lines = [
        "# Technical Transformer V1 Reliability Report", "",
        f"- Gate: **{gate.get('status', 'NOT_EVALUATED')}**",
        f"- Dataset: `{report.get('dataset', {}).get('dataset_id', 'unknown')}`",
        f"- Checkpoint: `{report.get('checkpoint', {}).get('checkpoint_id', 'unknown')}`", "",
        "## Checks", "",
        "| Check | Status | Actual | Threshold |", "|---|---:|---:|---:|",
    ]
    for name, item in gate.get("checks", {}).items():
        lines.append(f"| `{name}` | {'PASS' if item.get('passed') else 'FAIL'} | {item.get('actual')} | {item.get('threshold')} |")
    lines.extend(["", "## Split Summary", ""])
    for name, value in report.get("splits", {}).items():
        lines.append(f"- `{name}`: {value.get('sample_count', 0)} samples")
    return "\n".join(lines) + "\n"


def write_reliability_report(report: dict[str, Any], output: str | Path) -> tuple[Path, Path]:
    path = Path(output)
    if path.suffix.lower() == ".json":
        json_path = path; markdown_path = path.with_suffix(".md")
    else:
        path.mkdir(parents=True, exist_ok=True); json_path = path / "reliability_report.json"; markdown_path = path / "reliability_report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(render_reliability_markdown(report), encoding="utf-8")
    return json_path, markdown_path
