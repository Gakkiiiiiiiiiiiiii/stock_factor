"""Run deterministic fixture benchmarks through production research/ML paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_factor.engine.fitness import evaluate_factor
from stock_factor.engine.oos import evaluate_oos_splits
from stock_factor.engine.oos_seal import CandidateFreeze
from stock_factor.engine.vm import StackVM
from stock_factor.technical_transformer.data.features import build_features
from stock_factor.technical_transformer.data.labels import build_labels
from stock_factor.technical_transformer.data.schemas import FEATURE_NAMES
from stock_factor.technical_transformer.training.inference import predict
from stock_factor.technical_transformer.training.train import TechnicalTransformerSystem, seed_everything

ROOT = Path(__file__).parents[1]
BASELINE = ROOT / "benchmarks" / "baseline_v1.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _fixture_frame(seed: int, *, symbols: int = 12, days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    rows: list[dict[str, Any]] = []
    for index in range(symbols):
        close = 100 + index + np.cumsum(rng.normal(0.05, 0.8, days))
        close = np.maximum(close, 1.0)
        volume = 1_000_000 + rng.integers(0, 100_000, days)
        for day_index, (day, value) in enumerate(zip(dates, close)):
            rows.append(
                {
                    "symbol": f"S{index:03d}",
                    "trading_date": day,
                    "open": float(value * 0.998),
                    "high": float(value * 1.01),
                    "low": float(value * 0.99),
                    "close": float(value),
                    "volume": float(volume[day_index]),
                    "amount": float(volume[day_index] * value),
                    "turnover": 0.02,
                    "listing_days": (day - dates[0]).days + 200,
                }
            )
    return pd.DataFrame(rows)


def _fixture_panel(
    seed: int, *, symbols: int = 32, days: int = 120
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.02, (symbols, days))
    closes = 100 * np.cumprod(1 + returns, axis=1)
    features = {"close": closes, "ret": returns, "volume": np.abs(rng.normal(1.0, 0.1, (symbols, days)))}
    return closes, returns, features


def _dataset_build(seed: int) -> dict[str, Any]:
    frame = _fixture_frame(seed, symbols=4, days=180)
    group = frame[frame["symbol"] == "S000"].copy()
    features = build_features(group)
    labels = build_labels(group)
    with tempfile.TemporaryDirectory(prefix="stock-factor-benchmark-") as output:
        # Exercise the same immutable on-disk preparation primitive used by
        # training without retaining generated datasets in the repository.
        feature_path = Path(output) / "features.npy"
        np.save(feature_path, features.fillna(0.0).to_numpy(dtype=np.float32))
        labels_digest = _hash(labels.values.fillna(0.0).to_numpy(dtype=np.float32).round(8).tolist())
    return {
        "rows": len(group),
        "feature_columns": list(features.columns),
        "feature_digest": _hash(features.fillna(0.0).to_numpy(dtype=np.float32).round(8).tolist()),
        "label_digest": labels_digest,
    }


def _candidate_evaluation(seed: int, candidates: list[str]) -> dict[str, Any]:
    closes, _returns, features = _fixture_panel(seed)
    formulas = {
        "candidate-001": ["close"],
        "candidate-002": ["ret", "neg"],
        "candidate-003": ["volume", "cs_rank"],
    }
    vm = StackVM()
    results = []
    for candidate in candidates:
        factor = vm.execute(formulas[candidate], features)
        metrics = evaluate_factor(factor, closes, horizon=5, top_k=3) if factor is not None else {"passed": False}
        results.append({"candidate": candidate, "formula": formulas[candidate], "metrics": metrics})
    return {"candidate_order": candidates, "results": results}


def _oos_evaluation(seed: int, universe: list[str]) -> dict[str, Any]:
    closes, _returns, features = _fixture_panel(seed)
    factor = StackVM().execute(["close"], features)
    report = evaluate_oos_splits(factor, closes, horizon=5)
    freeze = CandidateFreeze(
        candidate_hash=_hash(universe),
        formula=["close"],
        dsl_version="factor-dsl.v1",
        feature_set_version="benchmark-features-v1",
        discovery_snapshot_id="fixture-discovery-v1",
        final_oos_snapshot_id="fixture-final-oos-v1",
        selection_rank=1,
        candidate_count=len(universe),
        candidate_frozen_at="2026-01-01T00:00:00+00:00",
    )
    return {"report": report, "freeze": freeze.to_dict(), "universe": universe}


def _transformer_inference(seed: int) -> dict[str, Any]:
    seed_everything(seed)
    model = TechnicalTransformerSystem(
        {"hidden_size": 16, "layers": 1, "heads": 2, "ffn_size": 32, "embedding_dim": 8, "sequence_length": 128}
    ).eval()
    window = np.linspace(-1.0, 1.0, 128 * len(FEATURE_NAMES), dtype=np.float32).reshape(128, -1)
    result = predict(model, window, device="cpu")
    return {"formal_ineligible": result["formal_ineligible"], "result_digest": _hash(result)}


def _run_task(name: str, budget: float, function) -> dict[str, Any]:
    started = time.perf_counter()
    payload = function()
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "artifact_hash": _hash(payload),
        "elapsed_seconds": round(elapsed, 6),
        "budget_seconds": budget,
        "within_budget": elapsed <= budget,
    }


def run_benchmarks() -> dict[str, Any]:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    seed = int(baseline["seed"])
    candidates = list(baseline["candidate_order"])
    universe = list(baseline["multiple_testing_universe"])
    budget_keys = {
        "dataset_build": "dataset_build_seconds",
        "candidate_evaluation": "candidate_evaluation_seconds",
        "oos": "oos_seconds",
        "transformer_inference": "transformer_inference_seconds",
    }
    tasks = [
        _run_task("dataset_build", baseline["budgets"][budget_keys["dataset_build"]], lambda: _dataset_build(seed)),
        _run_task(
            "candidate_evaluation",
            baseline["budgets"][budget_keys["candidate_evaluation"]],
            lambda: _candidate_evaluation(seed, candidates),
        ),
        _run_task("oos", baseline["budgets"][budget_keys["oos"]], lambda: _oos_evaluation(seed, universe)),
        _run_task(
            "transformer_inference",
            baseline["budgets"][budget_keys["transformer_inference"]],
            lambda: _transformer_inference(seed),
        ),
    ]
    return {
        "version": baseline["version"],
        "seed": seed,
        "tasks": tasks,
        "candidate_order": candidates,
        "multiple_testing_universe": universe,
        "statistical_scope": baseline["statistical_scope"],
    }


def check_result(result: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    hashes = {task["name"]: task["artifact_hash"] for task in result["tasks"]}
    if hashes != baseline.get("expected_artifact_hashes", {}):
        errors.append("artifact hash drift")
    if result.get("candidate_order") != baseline.get("candidate_order"):
        errors.append("candidate order drift")
    if result.get("multiple_testing_universe") != baseline.get("multiple_testing_universe"):
        errors.append("multiple-testing universe drift")
    if result.get("statistical_scope") != baseline.get("statistical_scope"):
        errors.append("statistical scope drift")
    errors.extend(f"{task['name']} exceeded budget" for task in result["tasks"] if not task["within_budget"])
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="fail on baseline hash/scope/order/budget drift")
    args = parser.parse_args()
    result = run_benchmarks()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if args.check:
        errors = check_result(result, baseline)
        if errors:
            raise SystemExit("benchmark check failed: " + "; ".join(errors))
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8", newline="\n")
    print(encoded)


if __name__ == "__main__":
    main()
