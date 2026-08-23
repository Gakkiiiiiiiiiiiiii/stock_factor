from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GoldSetRecord:
    sample_id: str
    symbol: str
    as_of: str
    snapshot_id: str
    labels: dict[str, int | float]
    confidence: float
    annotator_a: str | None = None
    annotator_b: str | None = None
    annotator_a_labels: dict[str, int | float] | None = None
    annotator_b_labels: dict[str, int | float] | None = None
    notes: str = ""
    expected_split: str | None = None


def load_gold_set(path: str | Path) -> list[GoldSetRecord]:
    root = Path(path)
    files = [root / "records.jsonl", root / "gold_set.jsonl", root / "records.json"]
    source = next((item for item in files if item.exists()), None)
    if source is None:
        raise FileNotFoundError(f"gold set records not found under {root}")
    if source.suffix == ".jsonl":
        values = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    else:
        values = json.loads(source.read_text(encoding="utf-8"))
    records = [GoldSetRecord(**value) for value in values]
    if len({item.sample_id for item in records}) != len(records):
        raise ValueError("gold set contains duplicate sample_id")
    return records


def cohens_kappa(annotator_a: list[int], annotator_b: list[int]) -> float:
    a = np.asarray(annotator_a); b = np.asarray(annotator_b)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("annotator arrays must have the same non-empty shape")
    observed = float(np.mean(a == b))
    labels = np.unique(np.concatenate([a, b]))
    expected = sum(float(np.mean(a == label)) * float(np.mean(b == label)) for label in labels)
    return float((observed - expected) / max(1.0 - expected, 1e-12))


def validate_gold_set(records: list[GoldSetRecord], *, min_kappa: float = 0.60) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_count": len(records), "frozen": True, "kappa": {}}
    keys = sorted({key for item in records for key in item.labels})
    for key in keys:
        paired = [item for item in records if item.annotator_a_labels and item.annotator_b_labels and key in item.annotator_a_labels and key in item.annotator_b_labels]
        if paired:
            result["kappa"][key] = cohens_kappa([int(item.annotator_a_labels[key]) for item in paired], [int(item.annotator_b_labels[key]) for item in paired])
    result["min_kappa"] = min_kappa
    result["passed"] = bool(records) and bool(result["kappa"]) and all(value is not None and value >= min_kappa for value in result["kappa"].values())
    return result
