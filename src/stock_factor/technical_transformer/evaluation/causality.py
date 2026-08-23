from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.labels import TechnicalLabels


def _as_frame_list(frames: pd.DataFrame | dict[str, pd.DataFrame] | Iterable[pd.DataFrame]) -> list[pd.DataFrame]:
    if isinstance(frames, pd.DataFrame):
        return [item for _, item in frames.groupby("symbol", sort=True)] if "symbol" in frames else [frames]
    if isinstance(frames, dict):
        return list(frames.values())
    return list(frames)


def _array(value: Any) -> np.ndarray:
    if isinstance(value, TechnicalLabels):
        value = value.values
    if isinstance(value, pd.DataFrame):
        value = value.to_numpy()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _changed_future(frame: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    changed = frame.copy()
    columns = [column for column in ("open", "high", "low", "close", "volume", "amount", "turnover") if column in changed]
    if cutoff + 1 < len(changed) and columns:
        changed.loc[cutoff + 1:, columns] = changed.loc[cutoff + 1:, columns].astype(float) * 1000.0
    return changed


def run_causality_suite(
    frames: pd.DataFrame | dict[str, pd.DataFrame] | Iterable[pd.DataFrame] | None,
    *,
    feature_builder: Callable[[pd.DataFrame], Any],
    label_builder: Callable[[pd.DataFrame], Any],
    window_builder: Callable[[pd.DataFrame], Any] | None = None,
    model: torch.nn.Module | None = None,
    cases: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Run prefix invariance against real raw bars.

    A missing input is deliberately represented as one failed evidence item,
    so a production gate cannot silently pass because the audit was skipped.
    """
    if frames is None:
        return {
            "status": "NOT_EVALUATED", "cases": 0, "feature_violations": 0, "label_violations": 0,
            "window_violations": 0, "model_output_violations": 0,
            "total_violations": 1, "passed": False, "reason": "NO_REAL_INPUT",
        }
    candidates: list[tuple[pd.DataFrame, int]] = []
    rng = np.random.default_rng(seed)
    for frame in _as_frame_list(frames):
        ordered = frame.sort_values("trading_date").reset_index(drop=True)
        minimum = 1 if window_builder is None else 128
        if len(ordered) > minimum + 1:
            positions = np.arange(minimum - 1, len(ordered) - 1, dtype=int)
            for position in positions:
                candidates.append((ordered, int(position)))
    if len(candidates) > cases:
        selected = rng.choice(len(candidates), size=cases, replace=False)
        candidates = [candidates[int(index)] for index in selected]
    counts = {"feature_violations": 0, "label_violations": 0, "window_violations": 0, "model_output_violations": 0}
    for frame, cutoff in candidates:
        changed = _changed_future(frame, cutoff)
        original_prefix = frame.iloc[:cutoff + 1]
        changed_prefix = changed.iloc[:cutoff + 1]
        feature_same = np.allclose(_array(feature_builder(original_prefix)), _array(feature_builder(changed_prefix)), equal_nan=True)
        label_same = np.allclose(_array(label_builder(original_prefix)), _array(label_builder(changed_prefix)), equal_nan=True)
        counts["feature_violations"] += int(not feature_same)
        counts["label_violations"] += int(not label_same)
        if window_builder is not None:
            first_window = window_builder(original_prefix)
            second_window = window_builder(changed_prefix)
            window_same = np.allclose(_array(first_window), _array(second_window), equal_nan=True)
            counts["window_violations"] += int(not window_same)
            if model is not None:
                model.eval()
                with torch.no_grad():
                    first_tensor = first_window if isinstance(first_window, torch.Tensor) else torch.as_tensor(first_window)
                    second_tensor = second_window if isinstance(second_window, torch.Tensor) else torch.as_tensor(second_window)
                    if first_tensor.ndim == 2:
                        first_tensor = first_tensor.unsqueeze(0)
                    if second_tensor.ndim == 2:
                        second_tensor = second_tensor.unsqueeze(0)
                    first_output = model(first_tensor)
                    second_output = model(second_tensor)
                output_same = _outputs_equal(first_output, second_output)
                counts["model_output_violations"] += int(not output_same)
    total = int(sum(counts.values()))
    return {"status": "EVALUATED", "cases": len(candidates), **counts, "total_violations": total, "passed": bool(candidates) and total == 0}


def _outputs_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return bool(torch.allclose(first, second, rtol=1e-6, atol=1e-6, equal_nan=True))
    if isinstance(first, dict) and isinstance(second, dict) and first.keys() == second.keys():
        return all(_outputs_equal(first[key], second[key]) for key in first)
    return first == second
