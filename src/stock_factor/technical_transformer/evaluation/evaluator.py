from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import TechnicalWindowDataset, canonical_split_name
from ..training.inference import load_checkpoint
from ..training.train import TechnicalTransformerSystem
from .composite import evaluate_prediction_arrays, technical_composite


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


def _collate(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    x, y, valid, metadata = zip(*batch)
    return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y)), torch.from_numpy(np.stack(valid)), list(metadata)


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
    valid_targets: list[np.ndarray] = []
    seen = 0
    for batch_index, (x, y, label_valid, _metadata) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        output = model(x.to(device_obj))
        for name in predictions:
            value = output[name]
            if name == "events":
                value = torch.sigmoid(value)
            predictions[name].append(value.detach().cpu().numpy())
        targets.append(y.numpy())
        valid_targets.append(label_valid.numpy())
        seen += len(y)
    if not targets:
        return EvaluationResult(target_split, {"data": {"empty": True}}, 0)
    target_array = np.concatenate(targets)
    valid_array = np.concatenate(valid_targets)
    prediction_array = {name: np.concatenate(values) for name, values in predictions.items()}
    metrics: dict[str, Any] = {"data": {"test_is_frozen": True}, **evaluate_prediction_arrays(target_array, prediction_array, label_valid=valid_array)}
    metrics["sample_count"] = seen
    metrics["technical_composite"] = technical_composite(metrics)
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


def evaluate_all_splits(checkpoint: str | Path, dataset_dir: str | Path, *, device: str = "auto", batch_size: int = 32, max_batches: int | None = None) -> dict[str, dict[str, Any]]:
    model = freeze_model(load_checkpoint(checkpoint, device=device))
    result = {}
    for split in ("valid", "time_test", "instrument_test", "double_oos"):
        evaluated = evaluate_split(model, TechnicalWindowDataset(dataset_dir, split), split=split, device=next(model.parameters()).device, batch_size=batch_size, max_batches=max_batches)
        result[split] = evaluated.as_dict()
    return result
