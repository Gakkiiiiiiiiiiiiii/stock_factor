from __future__ import annotations

import json
from pathlib import Path

import scripts.run_benchmarks as benchmarks
from scripts.run_benchmarks import run_benchmarks


def test_benchmark_smoke_is_reproducible_and_preserves_research_universe():
    first = run_benchmarks()
    second = run_benchmarks()

    def stable(result):
        return {
            key: value if key != "tasks" else [{k: item[k] for k in ("name", "artifact_hash")} for item in value]
            for key, value in result.items()
        }

    assert stable(first) == stable(second)
    assert [task["name"] for task in first["tasks"]] == [
        "dataset_build",
        "candidate_evaluation",
        "oos",
        "transformer_inference",
    ]
    assert first["candidate_order"] == first["multiple_testing_universe"]
    assert all(len(task["artifact_hash"]) == 64 for task in first["tasks"])
    baseline = json.loads((Path(__file__).parents[1] / "benchmarks" / "baseline_v1.json").read_text())
    assert first["statistical_scope"] == baseline["statistical_scope"]
    assert {task["name"]: task["artifact_hash"] for task in first["tasks"]} == baseline["expected_artifact_hashes"]
    assert all(task["within_budget"] and task["elapsed_seconds"] >= 0 for task in first["tasks"])


def test_benchmark_calls_production_paths(monkeypatch):
    calls = {name: 0 for name in ("features", "labels", "factor", "oos", "forward")}
    original_features, original_labels = benchmarks.build_features, benchmarks.build_labels
    original_factor, original_oos = benchmarks.evaluate_factor, benchmarks.evaluate_oos_splits
    original_forward = benchmarks.TechnicalTransformerSystem.forward

    def features(*args, **kwargs):
        calls["features"] += 1
        return original_features(*args, **kwargs)

    def labels(*args, **kwargs):
        calls["labels"] += 1
        return original_labels(*args, **kwargs)

    def factor(*args, **kwargs):
        calls["factor"] += 1
        return original_factor(*args, **kwargs)

    def oos(*args, **kwargs):
        calls["oos"] += 1
        return original_oos(*args, **kwargs)

    def forward(self, *args, **kwargs):
        calls["forward"] += 1
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(benchmarks, "build_features", features)
    monkeypatch.setattr(benchmarks, "build_labels", labels)
    monkeypatch.setattr(benchmarks, "evaluate_factor", factor)
    monkeypatch.setattr(benchmarks, "evaluate_oos_splits", oos)
    monkeypatch.setattr(benchmarks.TechnicalTransformerSystem, "forward", forward)
    run_benchmarks()
    assert all(calls.values())


def test_budget_drift_is_rejected():
    baseline = json.loads((Path(__file__).parents[1] / "benchmarks" / "baseline_v1.json").read_text())
    result = run_benchmarks()
    result["tasks"][0]["within_budget"] = False
    assert "dataset_build exceeded budget" in benchmarks.check_result(result, baseline)
