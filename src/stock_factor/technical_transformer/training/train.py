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
from ..data.schemas import FEATURE_NAMES, FEATURE_SCHEMA, LABEL_SCHEMA
from ..data.snapshot import QuantSnapshot
from ..model.encoder import TechnicalTransformer
from ..model.heads import TechnicalHeads
from ..model.losses import compute_loss, compute_mask_loss

MASK_TARGET_FEATURES = ["ret_1", "intraday_range_prev_close", "volume_ratio_20", "turnover", "atr14_close", "realized_vol_20"]


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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(x)
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


def _collate(batch: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    x, y, metadata = zip(*batch)
    return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y)), list(metadata)


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
    return manifest


def _stage_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    for item in config.get("stages", []):
        if item.get("name") == name:
            return item
    raise ValueError(f"stage not found: {name}")


def _save_checkpoint(
    model: TechnicalTransformerSystem,
    output_dir: Path,
    config: dict[str, Any],
    snapshot: QuantSnapshot,
    dataset_manifest: dict[str, Any],
    metrics: dict[str, Any],
    stage: str,
    epoch: int,
) -> Path:
    checkpoint_id = f"tech-v1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{stage}-e{epoch:03d}"
    directory = output_dir / checkpoint_id
    directory.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    try:
        from safetensors.torch import save_file
        save_file(state, str(directory / "model.safetensors"))
    except ImportError:
        torch.save(state, directory / "model.pt")
    (directory / "model_config.json").write_text(json.dumps(config.get("model", {}), indent=2), encoding="utf-8")
    (directory / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (directory / "feature_schema.json").write_text(json.dumps(FEATURE_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "label_schema.json").write_text(json.dumps(LABEL_SCHEMA.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (directory / "quant_snapshot_manifest.json").write_text(json.dumps(snapshot.manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (directory / "validation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    identity = {
        "checkpoint_id": checkpoint_id, "model_version": "technical-transformer.v1", "stage": stage, "epoch": epoch,
        "git_commit": _git_commit(), "dataset_id": dataset_manifest.get("dataset_id"),
        "market_snapshot_id": snapshot.snapshot_id, "seed": config.get("training", {}).get("seed", 42),
        "feature_schema_hash": dataset_manifest.get("feature_schema_hash"), "label_schema_hash": dataset_manifest.get("label_schema_hash"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device": "cuda" if torch.cuda.is_available() else "cpu", "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (directory / "checkpoint_manifest.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory


@torch.no_grad()
def _evaluate(model: TechnicalTransformerSystem, loader: DataLoader, device: torch.device, stage: str, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    direction_hits = 0
    direction_total = 0
    for batch_index, (x, y, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        output = model(x)
        if stage == "masked_pretraining":
            mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=device)
            mask[:, ::7] = True
            target = x[:, :, [FEATURE_NAMES.index(name) for name in MASK_TARGET_FEATURES]]
            loss = compute_mask_loss(output["mask_prediction"], target, mask)
            items = {"mask": float(loss.detach().cpu())}
        else:
            loss, items = compute_loss(output, y, stage=stage)
        sums["loss"] = sums.get("loss", 0.0) + float(loss.cpu())
        for name, value in items.items():
            sums[name] = sums.get(name, 0.0) + value
        if stage != "masked_pretraining":
            predicted = output["ma"][:, :6].sign()
            actual = y[:, :6].sign()
            direction_hits += int((predicted == actual).sum().cpu())
            direction_total += int(actual.numel())
        count += 1
    result = {name: value / max(count, 1) for name, value in sums.items()}
    if direction_total:
        result["ma_direction_accuracy"] = direction_hits / direction_total
    return result


def train(config: dict[str, Any]) -> Path:
    seed = int(config.get("training", {}).get("seed", 42))
    seed_everything(seed)
    snapshot_cfg = config.get("source", {})
    snapshot_root = Path(snapshot_cfg["snapshot_root"])
    snapshot_id = str(snapshot_cfg["market_snapshot_id"])
    snapshot = QuantSnapshot.load(snapshot_root, snapshot_id, require_qfq=True)
    snapshot.verify()
    dataset_manifest = _load_dataset(config, snapshot)
    dataset_dir = Path(config["dataset"]["path"])
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("hardware", {}).get("device", "auto") != "cpu" else "cpu")
    model = TechnicalTransformerSystem(config.get("model", {})).to(device)
    train_dataset = TechnicalWindowDataset(dataset_dir, "train")
    valid_dataset = TechnicalWindowDataset(dataset_dir, "valid")
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    num_workers = int(config.get("training", {}).get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=_collate)
    output_root = Path(config.get("checkpoint", {}).get("root", "artifacts/models/technical"))
    output_root.mkdir(parents=True, exist_ok=True)
    stages = config.get("training", {}).get("run_stages") or [item["name"] for item in config.get("stages", [])]
    last_checkpoint: Path | None = None
    for stage in stages:
        stage_cfg = _stage_config(config, stage)
        epochs = int(stage_cfg.get("epochs", 1))
        lr = float(stage_cfg.get("lr", config.get("training", {}).get("lr", 1e-4)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config.get("training", {}).get("weight_decay", 0.01)))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            max_steps = stage_cfg.get("max_steps_per_epoch")
            for step, (x, y, _) in enumerate(train_loader):
                if max_steps is not None and step >= int(max_steps):
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if stage == "masked_pretraining":
                    generator = torch.Generator(device=device).manual_seed(seed + epoch * 100000 + step)
                    mask = torch.rand((x.shape[0], x.shape[1]), generator=generator, device=device) < 0.15
                    mask_targets = x[:, :, [FEATURE_NAMES.index(name) for name in MASK_TARGET_FEATURES]].detach().clone()
                    mask_features = x.clone()
                    mask_features[:, :, [FEATURE_NAMES.index(name) for name in MASK_TARGET_FEATURES]] = torch.where(mask.unsqueeze(-1), torch.zeros_like(mask_targets), mask_targets)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        output = model(mask_features)
                        loss = compute_mask_loss(output["mask_prediction"], mask_targets, mask)
                        items = {"mask": float(loss.detach().cpu())}
                else:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        output = model(x)
                        loss, items = compute_loss(output, y, stage=stage)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("training", {}).get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.detach().cpu())
                seen += 1
            metrics = {"stage": stage, "epoch": epoch, "train_loss": running / max(seen, 1), "valid": _evaluate(model, valid_loader, device, stage, max_batches=stage_cfg.get("eval_max_batches"))}
            last_checkpoint = _save_checkpoint(model, output_root, config, snapshot, dataset_manifest, metrics, stage, epoch)
            print(json.dumps({"checkpoint": str(last_checkpoint), **metrics}, ensure_ascii=False), flush=True)
    if last_checkpoint is None:
        raise RuntimeError("no training stage executed")
    return last_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Technical Transformer V1")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    print(train(config))


if __name__ == "__main__":
    main()
