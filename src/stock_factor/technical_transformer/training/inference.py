from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.dataset import TechnicalWindowDataset
from ..data.schemas import LABEL_SCHEMA
from .train import TechnicalTransformerSystem


def load_checkpoint(checkpoint_dir: str | Path, device: str = "auto") -> TechnicalTransformerSystem:
    directory = Path(checkpoint_dir)
    config = json.loads((directory / "model_config.json").read_text(encoding="utf-8"))
    model = TechnicalTransformerSystem(config)
    model_path = directory / "model.safetensors"
    if model_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(model_path), device="cpu")
    else:
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    target = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    model.to(target).eval()
    return model


def load_registered_checkpoint(registry, model_id: str, device: str = "auto") -> TechnicalTransformerSystem:
    """Load only a promoted registry model for formal inference."""
    artifact = registry.require_promoted(model_id)
    checkpoint = registry.checkpoint_path(model_id)
    model = load_checkpoint(checkpoint.parent, device=device)
    setattr(model, "model_artifact_id", artifact.artifact_id)
    setattr(model, "model_record_id", artifact.record_id)
    return model


@torch.no_grad()
def predict(
    model: TechnicalTransformerSystem,
    window: np.ndarray,
    device: str | None = None,
    *,
    model_artifact_id: str | None = None,
) -> dict[str, Any]:
    registered_id = getattr(model, "model_artifact_id", None)
    registered_record_id = getattr(model, "model_record_id", None)
    if model_artifact_id is not None and registered_id != model_artifact_id:
        raise ValueError("formal model_artifact_id must come from a registered promoted model")
    effective_artifact_id = registered_id
    target_device = device or next(model.parameters()).device.type
    tensor = torch.from_numpy(np.asarray(window, dtype=np.float32)).unsqueeze(0).to(target_device)
    output = model(tensor)
    ma = output["ma"][0].cpu().numpy().tolist()
    boll = output["bollinger"][0].cpu().numpy().tolist()
    primitives = output["wyckoff_primitives"][0].cpu().numpy().tolist()
    phase_logits = output["phase"][0]
    phase = torch.softmax(phase_logits, dim=-1).cpu().numpy().tolist()
    events = torch.sigmoid(output["events"][0]).cpu().numpy().tolist()
    result = {
        "model_version": "technical-transformer.v1-reliability-v2",
        "formal_ineligible": effective_artifact_id is None or registered_record_id is None,
        "technical_embedding": output["technical_embedding"][0].cpu().numpy().round(8).tolist(),
        "moving_average": dict(zip(LABEL_SCHEMA.ma, ma)),
        "bollinger": dict(zip(LABEL_SCHEMA.bollinger, boll)),
        "wyckoff": {
            "phase_probs": dict(zip(LABEL_SCHEMA.phase, phase)),
            "primitives": dict(zip(LABEL_SCHEMA.wyckoff_primitives, primitives)),
            "events": dict(zip(LABEL_SCHEMA.events, events)),
        },
    }
    if effective_artifact_id is not None:
        result["model_artifact_id"] = effective_artifact_id
    if registered_record_id is not None:
        result["record_id"] = registered_record_id
    return result


def predict_registered(registry, model_id: str, window: np.ndarray, device: str | None = None) -> dict[str, Any]:
    artifact = registry.require_promoted(model_id)
    model = load_registered_checkpoint(registry, model_id, device=device or "auto")
    return predict(model, window, device=device, model_artifact_id=artifact.artifact_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Technical Transformer V1 inference on one dataset window")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--split", default="time_test", choices=["train", "valid", "test", "time_test", "instrument_test", "double_oos"]
    )
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    dataset = TechnicalWindowDataset(args.dataset, args.split)
    window, _labels, _label_valid, metadata = dataset[args.index]
    result = predict(load_checkpoint(args.checkpoint), window)
    result.update({"checkpoint": str(Path(args.checkpoint).resolve()), "sample": metadata})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
