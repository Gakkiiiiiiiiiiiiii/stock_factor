from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import TechnicalWindowDataset, canonical_split_name
from ..data.schemas import LABEL_SCHEMA
from ..training.inference import load_checkpoint
from ..training.train import TechnicalTransformerSystem
from .metrics import event_metrics, regression_metrics, soft_phase_metrics


@dataclass(frozen=True)
class EvaluationResult:
    split: str
    metrics: dict[str, Any]
    sample_count: int

    def as_dict(self) -> dict[str, Any]:
        return {"split": self.split, "sample_count": self.sample_count, **self.metrics}


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _collate(batch: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    x, y, metadata = zip(*batch)
    return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y)), list(metadata)


def _mean_metric(items: list[dict[str, float]], name: str) -> float:
    values = [float(item[name]) for item in items if name in item and np.isfinite(item[name])]
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def evaluate_split(
    model: TechnicalTransformerSystem,
    dataset: TechnicalWindowDataset,
    *,
    split: str | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    max_batches: int | None = None,
) -> EvaluationResult:
    target_split = canonical_split_name(split or dataset.split)
    target_dataset = dataset if dataset.split == target_split else TechnicalWindowDataset(dataset.dataset_dir, target_split)
    device_obj = torch.device(device)
    model = freeze_model(model).to(device_obj)
    loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    predictions: dict[str, list[np.ndarray]] = {"ma": [], "bollinger": [], "wyckoff_primitives": [], "phase": [], "events": []}
    targets: list[np.ndarray] = []
    seen = 0
    for batch_index, (x, y, _metadata) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        output = model(x.to(device_obj))
        for name in predictions:
            value = output[name]
            if name == "events":
                value = torch.sigmoid(value)
            predictions[name].append(value.detach().cpu().numpy())
        targets.append(y.numpy())
        seen += len(y)
    if not targets:
        return EvaluationResult(target_split, {"data": {"empty": True}}, 0)
    target_array = np.concatenate(targets)
    metrics: dict[str, Any] = {"data": {"test_is_frozen": True}}
    for group, prediction_parts in predictions.items():
        pred = np.concatenate(prediction_parts)
        target = target_array[:, LABEL_SCHEMA.slices[group]]
        if group == "phase":
            group_metrics = {name: soft_phase_metrics(target, pred) for name in ["aggregate"]}
            metrics[group] = group_metrics
        else:
            names = getattr(LABEL_SCHEMA, group)
            if group == "events":
                per_target = {name: event_metrics(target[:, index], pred[:, index]) for index, name in enumerate(names)}
            elif group == "ma":
                per_target = {
                    name: event_metrics(target[:, index], 1.0 / (1.0 + np.exp(-pred[:, index]))) if "cross" in name else regression_metrics(pred[:, index], target[:, index])
                    for index, name in enumerate(names)
                }
            else:
                per_target = {name: regression_metrics(pred[:, index], target[:, index]) for index, name in enumerate(names)}
            if group == "events":
                per_target["aggregate"] = {key: _mean_metric(list(per_target.values()), key) for key in ("pr_auc", "pr_auc_multiple_of_prevalence", "f1", "precision", "recall", "ece")}
            else:
                regression_items = [value for name, value in per_target.items() if "cross" not in name]
                per_target["aggregate"] = {key: _mean_metric(regression_items, key) for key in ("mae", "rmse", "pearson", "spearman", "sign_accuracy")}
            metrics[group] = per_target
    metrics["sample_count"] = seen
    metrics["summary"] = {
        "ma_slope_mean_pearson": _mean_metric([metrics["ma"][name] for name in LABEL_SCHEMA.ma[:6]], "pearson"),
        "ma_slope_mean_sign_accuracy": _mean_metric([metrics["ma"][name] for name in LABEL_SCHEMA.ma[:6]], "sign_accuracy"),
        "bollinger_mean_pearson": _mean_metric([metrics["bollinger"][name] for name in LABEL_SCHEMA.bollinger], "pearson"),
        "wyckoff_primitive_mean_spearman": _mean_metric([metrics["wyckoff_primitives"][name] for name in LABEL_SCHEMA.wyckoff_primitives], "spearman"),
        "event_mean_pr_auc_multiple_of_prevalence": _mean_metric([metrics["events"][name] for name in LABEL_SCHEMA.events], "pr_auc_multiple_of_prevalence"),
        "phase_macro_f1": float(metrics["phase"]["aggregate"]["macro_f1"]),
    }
    return EvaluationResult(target_split, metrics, seen)


def evaluate_checkpoint(
    checkpoint: str | Path,
    dataset_dir: str | Path,
    *,
    split: str = "time_test",
    device: str = "auto",
    batch_size: int = 32,
    max_batches: int | None = None,
) -> EvaluationResult:
    model = load_checkpoint(checkpoint, device=device)
    dataset = TechnicalWindowDataset(dataset_dir, split)
    return evaluate_split(model, dataset, split=split, device=next(model.parameters()).device, batch_size=batch_size, max_batches=max_batches)


def evaluate_all_splits(checkpoint: str | Path, dataset_dir: str | Path, *, device: str = "auto", batch_size: int = 32) -> dict[str, dict[str, Any]]:
    model = freeze_model(load_checkpoint(checkpoint, device=device))
    result = {}
    for split in ("valid", "time_test", "instrument_test", "double_oos"):
        evaluated = evaluate_split(model, TechnicalWindowDataset(dataset_dir, split), split=split, device=next(model.parameters()).device, batch_size=batch_size)
        result[split] = evaluated.as_dict()
    return result
