from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.dataset import TechnicalWindowDataset
from ..data.schemas import EVENT_LABELS, PHASE_LABELS
from .gold_set import load_gold_set, validate_gold_set
from .metrics import event_metrics, soft_phase_metrics

_EVENT_ALIASES = {name.lower(): name for name in EVENT_LABELS}
_EVENT_ALIASES.update({name.removesuffix("_score").lower(): name for name in EVENT_LABELS})
_EVENT_ALIASES.update({"spring": "spring_score", "ut": "upthrust_score", "sc": "sc_score", "bc": "bc_score", "sos": "sos_score", "sow": "sow_score"})


def _event_key(name: str) -> str | None:
    return _EVENT_ALIASES.get(str(name).strip().lower())


def _mean(items: list[dict[str, float]], key: str) -> float:
    values = [float(item[key]) for item in items if key in item and np.isfinite(item[key])]
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def evaluate_gold_set(
    model: torch.nn.Module,
    dataset_dir: str,
    gold_set: str,
    *,
    device: str | torch.device = "cpu",
    min_kappa: float = 0.60,
    allowed_splits: tuple[str, ...] = ("double_oos",),
    min_positive_per_event: int = 100,
    min_negative_per_event: int = 200,
) -> dict[str, Any]:
    records = load_gold_set(gold_set)
    dataset = TechnicalWindowDataset(dataset_dir, "train")
    manifest = dataset.dataset_dir.joinpath("dataset_manifest.json")
    dataset_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    snapshot_id = str(dataset_manifest.get("source_market_snapshot_id", ""))
    gold_manifest_path = Path(gold_set) / "gold_manifest.json"
    gold_manifest = json.loads(gold_manifest_path.read_text(encoding="utf-8")) if gold_manifest_path.exists() else {}
    validation = validate_gold_set(records, min_kappa=min_kappa)
    base: dict[str, Any] = {
        "status": "GOLD_DATASET_INVALID" if not validation["passed"] else "EVALUATING",
        "sample_count": len(records), "validation": validation, "kappa": validation.get("kappa", {}),
        "allowed_splits": list(allowed_splits),
        "allowed_split_passed": bool(allowed_splits) and not bool(set(allowed_splits) & {"train", "valid"}),
        "gold_manifest": gold_manifest,
    }
    manifest_splits = tuple(gold_manifest.get("allowed_splits", ()))
    if manifest_splits and set(manifest_splits) != set(allowed_splits):
        base.update({"status": "GOLD_MANIFEST_POLICY_MISMATCH", "passed": False})
        return base
    if gold_manifest.get("source_snapshot_id") and str(gold_manifest["source_snapshot_id"]) != snapshot_id:
        base.update({"status": "GOLD_SNAPSHOT_MISMATCH", "passed": False})
        return base
    if not base["allowed_split_passed"]:
        base.update({"status": "GOLD_SPLIT_POLICY_INVALID", "passed": False})
        return base
    if not validation["passed"]:
        base["passed"] = False
        return base
    split_datasets = {
        split: TechnicalWindowDataset(dataset_dir, split)
        for split in ("train", "valid", "time_test", "instrument_test", "double_oos")
    }
    all_index: dict[tuple[str, str], tuple[str, int]] = {}
    for split, split_dataset in split_datasets.items():
        for item_index, item in enumerate(split_dataset.records):
            all_index.setdefault((str(item["symbol"]), str(item["as_of"])), (split, item_index))
    index = {key: value for key, value in all_index.items() if value[0] in set(allowed_splits)}
    event_targets: dict[str, list[float]] = {name: [] for name in EVENT_LABELS}
    event_predictions: dict[str, list[float]] = {name: [] for name in EVENT_LABELS}
    phase_targets: list[list[float]] = []
    phase_predictions: list[list[float]] = []
    missing: list[str] = []
    in_sample: list[str] = []
    split_mismatches: list[str] = []
    matched_records: list[Any] = []
    embeddings: list[np.ndarray] = []
    matched_event_labels: list[dict[str, int]] = []
    target_device = torch.device(device)
    model = model.to(target_device).eval()
    for record in records:
        if str(record.snapshot_id) != snapshot_id:
            raise ValueError(f"gold snapshot mismatch for {record.sample_id}: {record.snapshot_id} != {snapshot_id}")
        actual = all_index.get((str(record.symbol), str(record.as_of)))
        if record.expected_split and record.expected_split not in allowed_splits:
            if actual is not None and actual[0] in {"train", "valid"}:
                in_sample.append(f"{record.symbol}:{record.as_of}:{actual[0]}")
            else:
                split_mismatches.append(f"{record.sample_id}:expected={record.expected_split}")
            continue
        lookup = index.get((str(record.symbol), str(record.as_of)))
        if lookup is None:
            if actual is not None and actual[0] in {"train", "valid"}:
                in_sample.append(f"{record.symbol}:{record.as_of}:{actual[0]}")
            elif actual is not None and record.expected_split and actual[0] != record.expected_split:
                split_mismatches.append(f"{record.sample_id}:expected={record.expected_split}:actual={actual[0]}")
            else:
                missing.append(f"{record.symbol}:{record.as_of}")
            continue
        split, item_index = lookup
        if record.expected_split and split != record.expected_split:
            split_mismatches.append(f"{record.sample_id}:expected={record.expected_split}:actual={split}")
            continue
        matched_records.append(record)
        sample = split_datasets[split][item_index]
        window, _labels, _valid, _metadata = sample
        output = model(torch.from_numpy(window).unsqueeze(0).to(target_device))
        event_prediction = torch.sigmoid(output["events"])[0].detach().cpu().numpy()
        if "technical_embedding" in output:
            embeddings.append(output["technical_embedding"][0].detach().cpu().numpy())
        matched_event_labels.append({
            canonical: int(float(value) > 0.5)
            for key, value in record.labels.items()
            if (canonical := _event_key(key)) is not None
        })
        for key, value in record.labels.items():
            canonical = _event_key(key)
            if canonical is not None:
                event_targets[canonical].append(float(value))
                event_predictions[canonical].append(float(event_prediction[EVENT_LABELS.index(canonical)]))
        if all(name in record.labels for name in PHASE_LABELS):
            phase_targets.append([float(record.labels[name]) for name in PHASE_LABELS])
            # soft_phase_metrics expects logits and performs the one permitted softmax.
            phase_predictions.append(output["phase"][0].detach().cpu().numpy().tolist())
    if in_sample:
        base.update({"status": "GOLD_IN_SAMPLE_LEAKAGE", "in_sample_records": in_sample, "passed": False})
        return base
    if split_mismatches:
        base.update({"status": "GOLD_SPLIT_MISMATCH", "split_mismatches": split_mismatches, "passed": False})
        return base
    if missing:
        base.update({"status": "GOLD_RECORD_NOT_FOUND", "missing_records": missing, "passed": False})
        return base
    event_metrics_by_name: dict[str, Any] = {}
    for name in EVENT_LABELS:
        if not event_targets[name]:
            continue
        event_metrics_by_name[name] = event_metrics(np.asarray(event_targets[name]), np.asarray(event_predictions[name]))
    display_names = {
        "spring_score": "Spring", "upthrust_score": "UT", "sc_score": "SC",
        "bc_score": "BC", "sos_score": "SOS", "sow_score": "SOW",
    }
    coverage: dict[str, Any] = {}
    for name in EVENT_LABELS:
        values = event_targets[name]
        positive = int(sum(value > 0.5 for value in values))
        negative = int(sum(value <= 0.5 for value in values))
        coverage[display_names.get(name, name)] = {
            "positive": positive, "negative": negative,
            "min_positive": int(min_positive_per_event), "min_negative": int(min_negative_per_event),
            "passed": positive >= min_positive_per_event and negative >= min_negative_per_event,
        }
    coverage_passed = all(item["passed"] for item in coverage.values())
    aggregate = {key: _mean(list(event_metrics_by_name.values()), key) for key in ("pr_auc", "relative_pr", "pr_auc_multiple_of_prevalence", "f1", "precision", "recall", "precision_at_top1pct", "precision_at_top5pct", "ece")}
    aggregate["status"] = "EVALUATED" if coverage_passed else "INCOMPLETE"
    display_metrics = {display_names[name]: value for name, value in event_metrics_by_name.items() if name in display_names}
    gold_neighbor = _gold_neighbor_semantic_hit(np.asarray(embeddings), matched_event_labels)
    base.update({
        "status": "EVALUATED" if coverage_passed else "GOLD_EVENT_COVERAGE_INCOMPLETE",
        "passed": bool(coverage_passed),
        "coverage": coverage, "coverage_passed": bool(coverage_passed),
        "event": {**event_metrics_by_name, **display_metrics, **aggregate},
        "matched_count": len(matched_records), "exact_match": True,
        "gold_neighbor_semantic_hit": gold_neighbor,
    })
    if phase_targets:
        base["phase"] = soft_phase_metrics(np.asarray(phase_targets), np.asarray(phase_predictions))
    return base


def _gold_neighbor_semantic_hit(embeddings: np.ndarray, labels: list[dict[str, int]]) -> float | None:
    """Evaluate nearest-neighbor agreement using frozen Gold event semantics."""
    if embeddings.ndim != 2 or len(embeddings) != len(labels) or len(embeddings) < 2:
        return None
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    hits: list[float] = []
    event_names = list(EVENT_LABELS)
    for index, item in enumerate(labels):
        available = [name for name in event_names if name in item]
        if not available:
            continue
        neighbor = int(np.argmax(similarity[index]))
        positive_events = [name for name in available if item[name] == 1]
        if positive_events:
            hits.append(float(any(labels[neighbor].get(name, 0) == 1 for name in positive_events)))
        else:
            hits.append(float(all(item[name] == labels[neighbor].get(name, -1) for name in available)))
    return float(np.mean(hits)) if hits else None
