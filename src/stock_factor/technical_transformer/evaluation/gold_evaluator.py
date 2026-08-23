from __future__ import annotations

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
) -> dict[str, Any]:
    records = load_gold_set(gold_set)
    dataset = TechnicalWindowDataset(dataset_dir, "train")
    manifest = dataset.dataset_dir.joinpath("dataset_manifest.json")
    import json
    dataset_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    snapshot_id = str(dataset_manifest.get("source_market_snapshot_id", ""))
    validation = validate_gold_set(records, min_kappa=min_kappa)
    base: dict[str, Any] = {
        "status": "GOLD_DATASET_INVALID" if not validation["passed"] else "EVALUATING",
        "sample_count": len(records), "validation": validation, "kappa": validation.get("kappa", {}),
    }
    if not validation["passed"]:
        base["passed"] = False
        return base
    index: dict[tuple[str, str], int] = {}
    for split in ("train", "valid", "time_test", "instrument_test", "double_oos"):
        split_dataset = TechnicalWindowDataset(dataset_dir, split)
        for item_index, item in enumerate(split_dataset.records):
            index.setdefault((str(item["symbol"]), str(item["as_of"])), (split, item_index))
    event_targets: dict[str, list[float]] = {name: [] for name in EVENT_LABELS}
    event_predictions: dict[str, list[float]] = {name: [] for name in EVENT_LABELS}
    phase_targets: list[list[float]] = []
    phase_predictions: list[list[float]] = []
    missing: list[str] = []
    target_device = torch.device(device)
    model = model.to(target_device).eval()
    for record in records:
        if str(record.snapshot_id) != snapshot_id:
            raise ValueError(f"gold snapshot mismatch for {record.sample_id}: {record.snapshot_id} != {snapshot_id}")
        lookup = index.get((str(record.symbol), str(record.as_of)))
        if lookup is None:
            missing.append(f"{record.symbol}:{record.as_of}")
            continue
        split, item_index = lookup
        sample = TechnicalWindowDataset(dataset_dir, split)[item_index]
        window, _labels, _valid, _metadata = sample
        output = model(torch.from_numpy(window).unsqueeze(0).to(target_device))
        event_prediction = torch.sigmoid(output["events"])[0].detach().cpu().numpy()
        for key, value in record.labels.items():
            canonical = _event_key(key)
            if canonical is not None:
                event_targets[canonical].append(float(value))
                event_predictions[canonical].append(float(event_prediction[EVENT_LABELS.index(canonical)]))
        if all(name in record.labels for name in PHASE_LABELS):
            phase_targets.append([float(record.labels[name]) for name in PHASE_LABELS])
            phase_predictions.append(torch.softmax(output["phase"], dim=-1)[0].detach().cpu().numpy().tolist())
    if missing:
        base.update({"status": "GOLD_RECORD_NOT_FOUND", "missing_records": missing, "passed": False})
        return base
    event_metrics_by_name: dict[str, Any] = {}
    for name in EVENT_LABELS:
        if not event_targets[name]:
            continue
        event_metrics_by_name[name] = event_metrics(np.asarray(event_targets[name]), np.asarray(event_predictions[name]))
    aggregate = {key: _mean(list(event_metrics_by_name.values()), key) for key in ("pr_auc", "relative_pr", "pr_auc_multiple_of_prevalence", "f1", "precision", "recall", "precision_at_top1pct", "precision_at_top5pct", "ece")}
    display_names = {
        "spring_score": "Spring", "upthrust_score": "UT", "sc_score": "SC",
        "bc_score": "BC", "sos_score": "SOS", "sow_score": "SOW",
    }
    display_metrics = {display_names[name]: value for name, value in event_metrics_by_name.items() if name in display_names}
    base.update({
        "status": "EVALUATED", "passed": True, "event": {**event_metrics_by_name, **display_metrics, **aggregate},
        "matched_count": len(records), "exact_match": True,
    })
    if phase_targets:
        base["phase"] = soft_phase_metrics(np.asarray(phase_targets), np.asarray(phase_predictions))
    return base
