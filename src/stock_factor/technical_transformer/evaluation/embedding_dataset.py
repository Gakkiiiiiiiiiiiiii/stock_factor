from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.dataset import TechnicalWindowDataset


@dataclass(frozen=True)
class EmbeddingSplit:
    embeddings: np.ndarray
    targets: np.ndarray
    label_valid: np.ndarray
    raw_features: np.ndarray
    metadata: tuple[dict[str, Any], ...]


def build_embedding_split(
    model: Any,
    dataset: TechnicalWindowDataset,
    *,
    device: Any = "cpu",
    limit: int | None = None,
) -> EmbeddingSplit:
    """Materialize a fixed dataset split for probes and nearest-neighbor audit."""
    import torch

    embeddings: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    raw: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    model.eval()
    for index in range(min(len(dataset), limit if limit is not None else len(dataset))):
        window, target, label_valid, item = dataset[index]
        with torch.no_grad():
            output = model(torch.from_numpy(window).unsqueeze(0).to(device))
        embeddings.append(output["technical_embedding"][0].detach().cpu().numpy())
        targets.append(target)
        valid.append(label_valid)
        raw.append(window[-1])
        metadata.append(item)
    dimension = int(getattr(model, "encoder", model).embedding_dim) if embeddings == [] else embeddings[0].shape[0]
    return EmbeddingSplit(
        embeddings=np.asarray(embeddings, dtype=np.float32).reshape((-1, dimension)),
        targets=np.asarray(targets, dtype=np.float32).reshape((-1, len(targets[0])))
        if targets
        else np.empty((0, 0), dtype=np.float32),
        label_valid=np.asarray(valid, dtype=np.uint8).reshape((-1, len(valid[0])))
        if valid
        else np.empty((0, 0), dtype=np.uint8),
        raw_features=np.asarray(raw, dtype=np.float32).reshape((-1, raw[0].shape[-1]))
        if raw
        else np.empty((0, 0), dtype=np.float32),
        metadata=tuple(metadata),
    )
