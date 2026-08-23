from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.dataset import RobustFeatureProcessor, TechnicalWindowDataset
from ..data.features import build_features
from ..data.labels import build_labels
from ..data.schemas import CONTINUOUS_FEATURES, FEATURE_NAMES, LABEL_SCHEMA
from ..training.inference import load_checkpoint
from .ablation import occlude_feature_group
from .baseline_runner import run_baseline_runner
from .causality import run_causality_suite
from .embedding_probe import nearest_neighbor_audit, run_embedding_probe
from .evaluator import evaluate_all_splits
from .gold_evaluator import evaluate_gold_set
from .invariance import model_embedding_invariance, model_event_probability_delta, transform_price_scale
from .reliability_gate import load_gate_config
from .report import build_reliability_report, write_reliability_report


def _collect_embeddings(model: torch.nn.Module, dataset: TechnicalWindowDataset, device: torch.device, *, limit: int = 2000) -> dict[str, np.ndarray]:
    embeddings: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    raw: list[np.ndarray] = []
    for index in range(min(len(dataset), limit)):
        window, target, label_valid, _metadata = dataset[index]
        with torch.no_grad():
            output = model(torch.from_numpy(window).unsqueeze(0).to(device))
        embeddings.append(output["technical_embedding"][0].detach().cpu().numpy())
        targets.append(target)
        valid.append(label_valid)
        raw.append(window[-1])
    if not embeddings:
        return {"embedding": np.empty((0, 0)), "targets": np.empty((0, len(LABEL_SCHEMA.names))), "valid": np.empty((0, len(LABEL_SCHEMA.names))), "raw": np.empty((0, len(FEATURE_NAMES)))}
    return {"embedding": np.asarray(embeddings), "targets": np.asarray(targets), "valid": np.asarray(valid), "raw": np.asarray(raw)}


def _embedding_evidence(model: torch.nn.Module, dataset_dir: Path, device: torch.device) -> dict[str, Any]:
    train = _collect_embeddings(model, TechnicalWindowDataset(dataset_dir, "train"), device)
    if len(train["embedding"]) == 0:
        return {"status": "NOT_EVALUATED", "reason": "EMPTY_TRAIN"}
    task_columns = {
        "ma_alignment": "bull_alignment_score", "trend_direction": "trend_direction",
        "boll_squeeze": "squeeze_score", "trading_range": "trading_range_score",
        "gold_spring": "spring_score", "gold_upthrust": "upthrust_score",
    }
    train_targets = {task: train["targets"][:, LABEL_SCHEMA.names.index(name)] for task, name in task_columns.items()}
    train_targets["phase"] = train["targets"][:, LABEL_SCHEMA.slices["phase"]]
    train_valid = {task: train["valid"][:, LABEL_SCHEMA.names.index(name)].astype(bool) for task, name in task_columns.items()}
    train_valid["phase"] = train["valid"][:, LABEL_SCHEMA.slices["phase"]].all(axis=1).astype(bool)
    split_results: dict[str, Any] = {}
    for split in ("time_test", "instrument_test", "double_oos"):
        test = _collect_embeddings(model, TechnicalWindowDataset(dataset_dir, split), device)
        if len(test["embedding"]) == 0:
            split_results[split] = {"status": "NOT_EVALUATED", "reason": "EMPTY_SPLIT"}
            continue
        test_targets = {task: test["targets"][:, LABEL_SCHEMA.names.index(name)] for task, name in task_columns.items()}
        test_targets["phase"] = test["targets"][:, LABEL_SCHEMA.slices["phase"]]
        test_valid = {task: test["valid"][:, LABEL_SCHEMA.names.index(name)].astype(bool) for task, name in task_columns.items()}
        test_valid["phase"] = test["valid"][:, LABEL_SCHEMA.slices["phase"]].all(axis=1).astype(bool)
        probe = run_embedding_probe(
            train["embedding"], train_targets, train_embedding=train["embedding"], test_embedding=test["embedding"],
            train_targets=train_targets, test_targets=test_targets,
            train_valid=train_valid, test_valid=test_valid,
            train_raw_features=train["raw"], test_raw_features=test["raw"],
        )
        nearest = nearest_neighbor_audit(test["embedding"], labels=np.argmax(test_targets["phase"], axis=1), k=min(20, max(1, len(test["embedding"]) - 1)))
        split_results[split] = {"status": "EVALUATED", **probe, **nearest}
    primary = split_results.get("double_oos", {"status": "NOT_EVALUATED"})
    return {"status": "EVALUATED", "splits": split_results, **primary}


def _invariance_evidence(model: torch.nn.Module, dataset_dir: Path, device: torch.device, *, limit: int = 64) -> dict[str, Any]:
    dataset = TechnicalWindowDataset(dataset_dir, "double_oos")
    if not len(dataset):
        return {"status": "NOT_EVALUATED", "reason": "EMPTY_DOUBLE_OOS"}
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    source_path = manifest.get("source_market_snapshot_path")
    original_windows: list[np.ndarray] = []
    scaled_windows: list[np.ndarray] = []
    if source_path and Path(source_path).exists():
        source = pd.read_parquet(source_path)
        processor_data = manifest.get("processor") or json.loads((dataset_dir / "processor.json").read_text(encoding="utf-8"))
        processor = RobustFeatureProcessor()
        processor.median = np.asarray(processor_data["median"], dtype=float)
        processor.scale = np.asarray(processor_data["scale"], dtype=float)
        processor.clip = float(processor_data.get("clip", 8.0))

        def make_window(frame: pd.DataFrame) -> np.ndarray:
            features = build_features(frame.sort_values("trading_date"))
            transformed = features.copy()
            transformed[CONTINUOUS_FEATURES] = processor.transform(features[CONTINUOUS_FEATURES].to_numpy(dtype=np.float32))
            transformed[FEATURE_NAMES] = transformed[FEATURE_NAMES].fillna(0.0)
            return transformed[FEATURE_NAMES].tail(128).to_numpy(dtype=np.float32)

        groups = {str(symbol): group.sort_values("trading_date").reset_index(drop=True) for symbol, group in source.groupby("symbol", sort=False)}
        for index in range(min(len(dataset), limit)):
            item = dataset.records[index]
            group = groups.get(str(item["symbol"]))
            if group is None:
                continue
            end = int(item["end_index"]) + 1
            prefix = group.iloc[:end]
            original_windows.append(make_window(prefix))
            scaled_windows.append(make_window(transform_price_scale(prefix)))
    else:
        original_windows = [dataset[index][0] for index in range(min(len(dataset), limit))]
        scaled_windows = list(original_windows)
    if not original_windows:
        return {"status": "NOT_EVALUATED", "reason": "NO_INVARIANCE_WINDOWS"}
    original = torch.from_numpy(np.asarray(original_windows, dtype=np.float32)).to(device)
    scaled = torch.from_numpy(np.asarray(scaled_windows, dtype=np.float32)).to(device)
    noisy = original + torch.randn_like(original) * 0.0005
    price_scale_cosine = model_embedding_invariance(model, original, scaled)
    noise_cosine = model_embedding_invariance(model, original, noisy)
    event_delta = model_event_probability_delta(model, original, noisy)
    return {
        "status": "EVALUATED", "source": "raw_snapshot" if source_path and Path(source_path).exists() else "dataset_feature_space", "price_scale_cosine": price_scale_cosine,
        "noise_cosine": noise_cosine, "event_probability_delta": event_delta,
    }


def _ablation_evidence(model: torch.nn.Module, dataset_dir: Path, device: torch.device, *, limit: int = 64) -> dict[str, Any]:
    dataset = TechnicalWindowDataset(dataset_dir, "double_oos")
    if not len(dataset):
        return {"status": "NOT_EVALUATED", "reason": "EMPTY_DOUBLE_OOS"}
    windows = torch.from_numpy(np.asarray([dataset[index][0] for index in range(min(len(dataset), limit))], dtype=np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        full = torch.sigmoid(model(windows)["events"]).mean().item()
        no_volume = torch.sigmoid(model(occlude_feature_group(windows, "VOLUME"))["events"]).mean().item()
    delta = abs(full - no_volume)
    return {
        "status": "EVALUATED", "full_event_score": full, "no_volume_event_score": no_volume,
        "event_score_delta": delta, "warnings": ["WYCKOFF_VOLUME_NOT_USED"] if delta < 0.01 else [],
    }


def _causality_evidence(model: torch.nn.Module, dataset_manifest: dict[str, Any], device: torch.device, cases: int) -> dict[str, Any]:
    source_path = dataset_manifest.get("source_market_snapshot_path")
    if not source_path or not Path(source_path).exists():
        return run_causality_suite(None, feature_builder=build_features, label_builder=build_labels, model=model, cases=cases)
    frame = pd.read_parquet(source_path)
    processor_data = dataset_manifest.get("processor") or json.loads((Path(dataset_manifest["series_path"]).parent / "processor.json").read_text(encoding="utf-8"))
    processor = RobustFeatureProcessor()
    processor.median = np.asarray(processor_data["median"], dtype=float)
    processor.scale = np.asarray(processor_data["scale"], dtype=float)
    processor.clip = float(processor_data.get("clip", 8.0))

    def window_builder(value: pd.DataFrame) -> torch.Tensor:
        features = build_features(value.sort_values("trading_date"))
        transformed = features.copy()
        transformed[CONTINUOUS_FEATURES] = processor.transform(features[CONTINUOUS_FEATURES].to_numpy(dtype=np.float32))
        transformed["quality_mask"] = transformed["quality_mask"].fillna(0.0)
        transformed[FEATURE_NAMES] = transformed[FEATURE_NAMES].fillna(0.0)
        return torch.from_numpy(transformed[FEATURE_NAMES].tail(128).to_numpy(dtype=np.float32))

    return run_causality_suite(
        frame, feature_builder=build_features, label_builder=build_labels,
        window_builder=window_builder, model=model, cases=cases,
    )


def run_reliability_evaluation(
    *,
    checkpoint: str | Path,
    dataset: str | Path,
    mode: str = "PRODUCTION",
    gold_set: str | Path | None = None,
    baseline_root: str | Path | None = None,
    gate_config: str | Path | None = None,
    report: str | Path = "artifacts/reports/technical",
    device: str = "auto",
    batch_size: int = 32,
    causality_cases: int = 500,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint)
    dataset_dir = Path(dataset)
    identity = json.loads((checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    direct_failures: list[str] = []
    if identity.get("dataset_id") != manifest.get("dataset_id"):
        direct_failures.append("CHECKPOINT_DATASET_ID_MISMATCH")
    if identity.get("market_snapshot_id") != manifest.get("source_market_snapshot_id"):
        direct_failures.append("CHECKPOINT_SNAPSHOT_ID_MISMATCH")
    if identity.get("feature_schema_hash") != manifest.get("feature_schema_hash"):
        direct_failures.append("CHECKPOINT_FEATURE_SCHEMA_MISMATCH")
    if identity.get("label_schema_hash") != manifest.get("label_schema_hash"):
        direct_failures.append("CHECKPOINT_LABEL_SCHEMA_MISMATCH")
    model = load_checkpoint(checkpoint_dir, device=device)
    target_device = next(model.parameters()).device
    gates = load_gate_config(gate_config)
    max_batches = 2 if str(mode).upper() == "SMOKE" else None
    splits = evaluate_all_splits(checkpoint_dir, dataset_dir, device=str(target_device), batch_size=batch_size, max_batches=max_batches)
    causality = _causality_evidence(model, manifest, target_device, causality_cases) if str(mode).upper() != "SMOKE" else {"status": "NOT_EVALUATED", "total_violations": None, "cases": 0}
    gold = evaluate_gold_set(model, str(dataset_dir), str(gold_set), device=target_device, min_kappa=float(gates["gold"]["kappa_min"])) if gold_set and str(mode).upper() != "SMOKE" else {"status": "NOT_PROVIDED" if not gold_set else "NOT_EVALUATED"}
    embedding = _embedding_evidence(model, dataset_dir, target_device) if str(mode).upper() != "SMOKE" else {"status": "NOT_EVALUATED"}
    invariance = _invariance_evidence(model, dataset_dir, target_device) if str(mode).upper() != "SMOKE" else {"status": "NOT_EVALUATED"}
    ablation = _ablation_evidence(model, dataset_dir, target_device) if str(mode).upper() != "SMOKE" else {"status": "NOT_EVALUATED"}
    baseline = {"status": "NOT_EVALUATED"}
    if str(mode).upper() != "SMOKE":
        baseline = run_baseline_runner(dataset_dir, names=("gru", "last_day_mlp"), device=target_device, output_dir=baseline_root)
        transformer_score = (splits.get("double_oos") or {}).get("technical_composite")
        gru_score = baseline.get("scores", {}).get("gru")
        if transformer_score is not None and gru_score is not None:
            from .baselines import relative_gain
            baseline["scores"]["transformer"] = transformer_score
            baseline["transformer_wyckoff_gold_relative_gain"] = relative_gain(transformer_score, gru_score)
        last_day_score = baseline.get("scores", {}).get("last_day_mlp")
        if transformer_score is not None and last_day_score is not None and transformer_score <= last_day_score:
            baseline.setdefault("warnings", []).append("SEQUENCE_MODEL_NOT_NEEDED")
    report_value = build_reliability_report(
        checkpoint_identity=identity, dataset_manifest=manifest, splits=splits, mode=mode,
        leakage_audit=manifest.get("leakage_audit"), causality=causality,
        gold_set=gold, embedding=embedding, invariance=invariance, baseline=baseline,
        ablation=ablation, gates=gates, direct_failures=direct_failures,
    )
    write_reliability_report(report_value, report)
    return report_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Technical Transformer reliability orchestrator")
    parser.add_argument("--mode", default="PRODUCTION", choices=["SMOKE", "RESEARCH", "PRODUCTION", "smoke", "research", "production"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gold-set")
    parser.add_argument("--baseline-root")
    parser.add_argument("--gate-config")
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--causality-cases", type=int, default=500)
    args = parser.parse_args()
    result = run_reliability_evaluation(
        checkpoint=args.checkpoint, dataset=args.dataset, mode=args.mode,
        gold_set=args.gold_set, baseline_root=args.baseline_root, gate_config=args.gate_config,
        report=args.report, device=args.device, batch_size=args.batch_size, causality_cases=args.causality_cases,
    )
    print(json.dumps({"json": str(Path(args.report) / "reliability_report.json") if Path(args.report).suffix == "" else str(args.report), "gate": result["reliability_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
