from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import DatasetConfig, TechnicalWindowDataset, build_dataset
from ..data.schemas import FEATURE_NAMES, FEATURE_SCHEMA, LABEL_SCHEMA, MASK_RECONSTRUCTION_FEATURES
from ..data.snapshot import QuantSnapshot
from ..evaluation.composite import evaluate_prediction_arrays, technical_composite
from ..model.encoder import TechnicalTransformer
from ..model.heads import TechnicalHeads
from ..model.losses import compute_loss, compute_mask_loss, event_pos_weight
from .masking import apply_mask
from .optimizer import TrainingStage, build_optimizer, optimizer_group_lrs
from .selection import StageSelectionResult, select_stage_score

MASK_TARGET_FEATURES = MASK_RECONSTRUCTION_FEATURES


class TechnicalTransformerSystem(nn.Module):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = TechnicalTransformer(
            input_dim=int(model_config.get("input_dim", len(FEATURE_NAMES))),
            hidden_size=int(model_config.get("hidden_size", 384)), layers=int(model_config.get("layers", 6)),
            heads=int(model_config.get("heads", 8)), ffn_size=int(model_config.get("ffn_size", 1536)),
            dropout=float(model_config.get("dropout", 0.10)), embedding_dim=int(model_config.get("embedding_dim", 256)),
            sequence_length=int(model_config.get("sequence_length", 128)),
        )
        self.heads = TechnicalHeads(hidden_size=self.encoder.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        mask_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(x, padding_mask=padding_mask, mask_positions=mask_positions)
        encoded.update(self.heads(encoded["cls_hidden"]))
        return encoded


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _collate(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    x, y, valid, metadata = zip(*batch)
    return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y)), torch.from_numpy(np.stack(valid)), list(metadata)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _load_dataset(config: dict[str, Any], snapshot: QuantSnapshot) -> dict[str, Any]:
    dataset_cfg = DatasetConfig.from_mapping(config.get("dataset", {}))
    dataset_dir = Path(config["dataset"]["path"])
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        manifest = build_dataset(snapshot, dataset_dir, dataset_cfg)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_market_snapshot_id") != snapshot.snapshot_id:
        raise RuntimeError("dataset snapshot identity does not match training snapshot_id")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA["schema_version"] or manifest.get("label_schema_version") != LABEL_SCHEMA.version:
        raise RuntimeError("dataset schema is not Technical Transformer V2; rebuild the dataset before training")
    return manifest


def _stage_config(config: dict[str, Any], name: str) -> TrainingStage:
    for item in config.get("stages", []):
        if item.get("name") == name:
            return TrainingStage.from_mapping(item, default_lr=float(config.get("training", {}).get("lr", 1e-4)))
    raise ValueError(f"stage not found: {name}")


def _validation_composite(metrics: dict[str, Any]) -> float:
    """Compatibility wrapper for callers that used the old helper."""
    return float(select_stage_score(str(metrics.get("stage", "wyckoff_phase_events")), metrics).score)


def _save_checkpoint(
    model: TechnicalTransformerSystem,
    output_dir: Path,
    config: dict[str, Any],
    snapshot: QuantSnapshot,
    dataset_manifest: dict[str, Any],
    metrics: dict[str, Any],
    stage: TrainingStage,
    epoch: int,
    *,
    optimizer_steps: int,
    best_valid: bool,
    optimizer: torch.optim.Optimizer,
    selection: StageSelectionResult,
) -> Path:
    checkpoint_id = f"tech-v1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{stage.name}-e{epoch:03d}"
    directory = output_dir / checkpoint_id
    directory.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    try:
        from safetensors.torch import save_file
        save_file(state, str(directory / "model.safetensors"))
    except ImportError:
        torch.save(state, directory / "model.pt")
    training_config = dict(config)
    training_config.setdefault("training", {})["effective_batch_size"] = int(config.get("training", {}).get("batch_size", 32)) * int(config.get("training", {}).get("gradient_accumulation", 1))
    training_config["training"]["optimizer_steps"] = int(optimizer_steps)
    training_config["training"]["optimizer_group_lrs"] = optimizer_group_lrs(optimizer)
    (directory / "model_config.json").write_text(json.dumps(config.get("model", {}), indent=2), encoding="utf-8")
    (directory / "training_config.json").write_text(json.dumps(training_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (directory / "feature_schema.json").write_text(json.dumps(FEATURE_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "label_schema.json").write_text(json.dumps(LABEL_SCHEMA.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (directory / "quant_snapshot_manifest.json").write_text(json.dumps(snapshot.manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (directory / "validation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    identity = {
        "manifest_version": "technical-checkpoint.v2", "checkpoint_id": checkpoint_id, "model_version": "technical-transformer.v1-reliability-v2",
        "checkpoint_status": "CANDIDATE", "stage": stage.name, "epoch": epoch, "best_valid": best_valid,
        "git_commit": _git_commit(), "dataset_id": dataset_manifest.get("dataset_id"),
        "market_snapshot_id": snapshot.snapshot_id, "seed": config.get("training", {}).get("seed", 42),
        "feature_schema_hash": dataset_manifest.get("feature_schema_hash"), "label_schema_hash": dataset_manifest.get("label_schema_hash"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device": next(model.parameters()).device.type, "created_at": datetime.now(timezone.utc).isoformat(),
        "training": {
            "batch_size": int(config.get("training", {}).get("batch_size", 32)),
            "gradient_accumulation": int(config.get("training", {}).get("gradient_accumulation", 1)),
            "effective_batch_size": int(config.get("training", {}).get("batch_size", 32)) * int(config.get("training", {}).get("gradient_accumulation", 1)),
            "optimizer_steps": int(optimizer_steps), "optimizer_group_lrs": optimizer_group_lrs(optimizer),
        },
        "validation": {"report_id": f"{checkpoint_id}:valid", "best_valid": best_valid},
        "selection": {
            "policy_version": "technical-selection.v1", "stage": selection.stage,
            "score": selection.score, "components": selection.components, "valid": selection.valid,
        },
        "test": {"executed": False}, "reliability_gate": {"status": "NOT_EVALUATED"},
    }
    (directory / "checkpoint_manifest.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory


@torch.no_grad()
def _evaluate(
    model: TechnicalTransformerSystem,
    loader: DataLoader,
    device: torch.device,
    stage: TrainingStage,
    *,
    seed: int,
    max_batches: int | None = None,
    event_weight: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    predictions: dict[str, list[np.ndarray]] = {"ma": [], "bollinger": [], "wyckoff_primitives": [], "phase": [], "events": []}
    targets: list[np.ndarray] = []
    valid_targets: list[np.ndarray] = []
    for batch_index, (x, y, label_valid, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        label_valid = label_valid.to(device=device, dtype=torch.bool)
        if stage.name == "masked_pretraining":
            quality_index = FEATURE_NAMES.index("quality_mask")
            turnover_index = FEATURE_NAMES.index("turnover_observed")
            valid_days = x[..., quality_index] > 0
            feature_validity = torch.ones_like(x, dtype=torch.bool)
            feature_validity[..., FEATURE_NAMES.index("turnover")] = x[..., turnover_index] > 0
            masked = apply_mask(
                x, valid_days=valid_days, feature_validity=feature_validity,
                mode=stage.mask_mode or "mixed", seed=seed + batch_index,
            )
            output = model(masked.input, mask_positions=masked.positions)
            target_indices = tuple(FEATURE_NAMES.index(name) for name in MASK_RECONSTRUCTION_FEATURES)
            loss = compute_mask_loss(output["mask_prediction"], masked.target, masked.positions, target_indices=target_indices)
            items = {"mask": float(loss.cpu())}
        else:
            output = model(x)
            loss, items = compute_loss(
                output, y, stage=stage.name, active_heads=stage.active_heads or None,
                label_valid=label_valid, event_weight=event_weight,
            )
            predictions["ma"].append(output["ma"].cpu().numpy())
            predictions["bollinger"].append(output["bollinger"].cpu().numpy())
            predictions["wyckoff_primitives"].append(output["wyckoff_primitives"].cpu().numpy())
            predictions["phase"].append(output["phase"].cpu().numpy())
            predictions["events"].append(torch.sigmoid(output["events"]).cpu().numpy())
            targets.append(y.cpu().numpy())
            valid_targets.append(label_valid.cpu().numpy())
        sums["loss"] = sums.get("loss", 0.0) + float(loss.cpu())
        for name, value in items.items():
            sums[name] = sums.get(name, 0.0) + value
        count += 1
    result: dict[str, Any] = {name: value / max(count, 1) for name, value in sums.items()}
    if targets:
        result.update(evaluate_prediction_arrays(
            np.concatenate(targets), {key: np.concatenate(value) for key, value in predictions.items()},
            label_valid=np.concatenate(valid_targets),
        ))
        result["technical_composite"] = technical_composite(result)
    selection = select_stage_score(stage.name, result)
    result["selection_score"] = selection.score
    result["selection_components"] = selection.components
    result["selection_valid"] = selection.valid
    result["validation_composite"] = selection.score
    return result


def _event_weights(dataset: TechnicalWindowDataset, device: torch.device) -> torch.Tensor | None:
    if len(dataset) == 0:
        return None
    values = np.stack([dataset[index][1][LABEL_SCHEMA.slices["events"]] for index in range(len(dataset))])
    valid = np.stack([dataset[index][2][LABEL_SCHEMA.slices["events"]] for index in range(len(dataset))])
    return event_pos_weight(torch.from_numpy(values).to(device), torch.from_numpy(valid).to(device))


def train(config: dict[str, Any]) -> Path:
    seed = int(config.get("training", {}).get("seed", 42))
    seed_everything(seed)
    snapshot_cfg = config.get("source", {})
    snapshot = QuantSnapshot.load(Path(snapshot_cfg["snapshot_root"]), str(snapshot_cfg["market_snapshot_id"]), require_qfq=True)
    snapshot.verify()
    dataset_manifest = _load_dataset(config, snapshot)
    dataset_dir = Path(config["dataset"]["path"])
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("hardware", {}).get("device", "auto") != "cpu" else "cpu")
    model = TechnicalTransformerSystem(config.get("model", {})).to(device)
    train_dataset = TechnicalWindowDataset(dataset_dir, "train")
    valid_dataset = TechnicalWindowDataset(dataset_dir, "valid")
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    accumulation = max(1, int(config.get("training", {}).get("gradient_accumulation", 1)))
    num_workers = int(config.get("training", {}).get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=_collate)
    output_root = Path(config.get("checkpoint", {}).get("root", "artifacts/models/technical"))
    output_root.mkdir(parents=True, exist_ok=True)
    stages = config.get("training", {}).get("run_stages") or [item["name"] for item in config.get("stages", [])]
    last_checkpoint: Path | None = None
    event_weight = _event_weights(train_dataset, device)
    for stage_name in stages:
        stage = _stage_config(config, stage_name)
        optimizer = build_optimizer(model, stage, float(config.get("training", {}).get("weight_decay", 0.01)))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best_score = float("-inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_checkpoint: Path | None = None
        stale_epochs = 0
        optimizer_steps = 0
        for epoch in range(1, stage.epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            optimizer.zero_grad(set_to_none=True)
            configured_max_steps = next((item.get("max_steps_per_epoch") for item in config.get("stages", []) if item.get("name") == stage.name), None)
            limit = min(len(train_loader), int(configured_max_steps)) if configured_max_steps is not None else len(train_loader)
            for step, (x, y, label_valid, _) in enumerate(train_loader):
                if step >= limit:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                label_valid = label_valid.to(device=device, dtype=torch.bool, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    if stage.name == "masked_pretraining":
                        quality_index = FEATURE_NAMES.index("quality_mask")
                        turnover_index = FEATURE_NAMES.index("turnover_observed")
                        valid_days = x[..., quality_index] > 0
                        feature_validity = torch.ones_like(x, dtype=torch.bool)
                        feature_validity[..., FEATURE_NAMES.index("turnover")] = x[..., turnover_index] > 0
                        masked = apply_mask(
                            x, valid_days=valid_days, feature_validity=feature_validity,
                            mode=stage.mask_mode or "mixed", seed=seed + epoch * 100000 + step,
                        )
                        output = model(masked.input, mask_positions=masked.positions)
                        target_indices = tuple(FEATURE_NAMES.index(name) for name in MASK_RECONSTRUCTION_FEATURES)
                        raw_loss = compute_mask_loss(output["mask_prediction"], masked.target, masked.positions, target_indices=target_indices)
                        items = {"mask": float(raw_loss.detach().cpu())}
                    else:
                        output = model(x)
                        raw_loss, items = compute_loss(
                            output, y, stage=stage.name, active_heads=stage.active_heads or None,
                            label_valid=label_valid, event_weight=event_weight,
                        )
                    loss = raw_loss / accumulation
                scaler.scale(loss).backward()
                running += float(raw_loss.detach().cpu())
                seen += 1
                is_last = step + 1 == limit
                if (step + 1) % accumulation == 0 or is_last:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("training", {}).get("grad_clip", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
            validation = _evaluate(
                model, valid_loader, device, stage, seed=seed + epoch * 1000,
                max_batches=next((item.get("eval_max_batches") for item in config.get("stages", []) if item.get("name") == stage.name), None),
                event_weight=event_weight,
            )
            metrics = {"stage": stage.name, "epoch": epoch, "train_loss": running / max(seen, 1), "valid": validation}
            selection = select_stage_score(stage.name, validation)
            score = float(selection.score)
            improved = score > best_score + 1e-8
            if improved:
                best_score = score
                stale_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale_epochs += 1
            checkpoint = _save_checkpoint(
                model, output_root, config, snapshot, dataset_manifest, metrics, stage, epoch,
                optimizer_steps=optimizer_steps, best_valid=improved, optimizer=optimizer, selection=selection,
            )
            if improved:
                best_checkpoint = checkpoint
            print(json.dumps({"checkpoint": str(checkpoint), **metrics, "selection_score": score, "best_so_far": improved, "optimizer_steps": optimizer_steps}, ensure_ascii=False), flush=True)
            if stale_epochs >= stage.patience:
                break
        if best_state is None or best_checkpoint is None:
            raise RuntimeError(f"stage {stage.name} produced no validation checkpoint")
        model.load_state_dict(best_state)
        last_checkpoint = best_checkpoint
    if last_checkpoint is None:
        raise RuntimeError("no training stage executed")
    return last_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Technical Transformer V1 reliability candidate")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    print(train(config))


if __name__ == "__main__":
    main()
