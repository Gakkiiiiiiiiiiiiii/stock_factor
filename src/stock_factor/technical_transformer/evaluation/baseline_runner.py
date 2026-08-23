from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data.dataset import TechnicalWindowDataset
from ..data.schemas import LABEL_SCHEMA
from .baselines import make_baseline, relative_gain
from .composite import evaluate_prediction_arrays, technical_composite


def _collate(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]):
    x, y, valid, metadata = zip(*batch)
    return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y)), torch.from_numpy(np.stack(valid)), metadata


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    values = F.huber_loss(prediction, target, reduction="none")
    values = values[valid]
    return values.mean() if values.numel() else prediction.sum() * 0.0


def _baseline_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses = []
    for group in ("ma", "bollinger", "wyckoff_primitives"):
        sl = LABEL_SCHEMA.slices[group]
        losses.append(_masked_mse(prediction[:, sl], target[:, sl], valid[:, sl]))
    phase = LABEL_SCHEMA.slices["phase"]
    phase_rows = valid[:, phase].all(dim=-1)
    if phase_rows.any():
        losses.append((-target[phase][phase_rows] * F.log_softmax(prediction[phase][phase_rows], dim=-1)).sum(dim=-1).mean())
    event = LABEL_SCHEMA.slices["events"]
    event_values = F.binary_cross_entropy_with_logits(prediction[:, event], target[:, event], reduction="none")
    event_values = event_values[valid[:, event]]
    if event_values.numel():
        losses.append(event_values.mean())
    return sum(losses, prediction.sum() * 0.0) / max(len(losses), 1)


@torch.no_grad()
def evaluate_baseline_model(model: torch.nn.Module, dataset: TechnicalWindowDataset, *, device: torch.device, batch_size: int = 64) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    for x, y, label_valid, _ in loader:
        predictions.append(model(x.to(device)).cpu().numpy())
        targets.append(y.numpy())
        valid.append(label_valid.numpy())
    if not targets:
        return {"sample_count": 0, "technical_composite": 0.0}
    target_array = np.concatenate(targets)
    raw = np.concatenate(predictions)
    prediction_arrays = {
        "ma": raw[:, LABEL_SCHEMA.slices["ma"]],
        "bollinger": raw[:, LABEL_SCHEMA.slices["bollinger"]],
        "wyckoff_primitives": raw[:, LABEL_SCHEMA.slices["wyckoff_primitives"]],
        "phase": raw[:, LABEL_SCHEMA.slices["phase"]],
        "events": 1.0 / (1.0 + np.exp(-raw[:, LABEL_SCHEMA.slices["events"]])),
    }
    metrics = evaluate_prediction_arrays(target_array, prediction_arrays, label_valid=np.concatenate(valid))
    metrics["sample_count"] = len(target_array)
    metrics["technical_composite"] = technical_composite(metrics)
    return metrics


def run_baseline_runner(
    dataset_dir: str | Path,
    *,
    names: tuple[str, ...] = ("gru",),
    device: str | torch.device = "cpu",
    epochs: int = 3,
    patience: int = 2,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train baselines on exactly the Transformer train/valid contract."""
    torch.manual_seed(seed)
    target_device = torch.device(device)
    train_dataset = TechnicalWindowDataset(dataset_dir, "train")
    valid_dataset = TechnicalWindowDataset(dataset_dir, "valid")
    test_datasets = {name: TechnicalWindowDataset(dataset_dir, name) for name in ("time_test", "instrument_test", "double_oos")}
    result: dict[str, Any] = {"status": "EVALUATED", "models": {}}
    for name in names:
        model = make_baseline(name, len(__import__("stock_factor.technical_transformer.data.schemas", fromlist=["FEATURE_NAMES"]).FEATURE_NAMES), len(LABEL_SCHEMA.names)).to(target_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
        best_state = None
        best_score = float("-inf")
        stale = 0
        for _epoch in range(epochs):
            model.train()
            for x, y, label_valid, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = _baseline_loss(model(x.to(target_device)), y.to(target_device), label_valid.to(target_device, dtype=torch.bool))
                loss.backward()
                optimizer.step()
            validation = evaluate_baseline_model(model, valid_dataset, device=target_device, batch_size=batch_size)
            score = float(validation.get("technical_composite", 0.0))
            if score > best_score:
                best_score, stale = score, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        item: dict[str, Any] = {"valid": evaluate_baseline_model(model, valid_dataset, device=target_device, batch_size=batch_size)}
        item["splits"] = {split: evaluate_baseline_model(model, split_dataset, device=target_device, batch_size=batch_size) for split, split_dataset in test_datasets.items()}
        result["models"][name] = item
        if output_dir:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path / f"{name}.pt")
    result["scores"] = {name: value["splits"]["double_oos"].get("technical_composite", 0.0) for name, value in result["models"].items()}
    if "transformer" in result["scores"] and "gru" in result["scores"]:
        result["transformer_wyckoff_gold_relative_gain"] = relative_gain(result["scores"]["transformer"], result["scores"]["gru"])
    return result


def evaluate_baselines(*args, **kwargs) -> dict[str, Any]:
    return run_baseline_runner(*args, **kwargs)
