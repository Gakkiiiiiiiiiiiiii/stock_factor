from __future__ import annotations

import inspect
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


def _historical(value: Any, end: int) -> Any:
    if isinstance(value, TechnicalLabels):
        return TechnicalLabels(value.values[:end], value.valid[:end])
    if isinstance(value, pd.DataFrame):
        return value.iloc[:end]
    if isinstance(value, dict):
        return {key: _historical(item, end) for key, item in value.items()}
    try:
        return np.asarray(value)[:end]
    except (TypeError, IndexError):
        return value


def _compare_arrays(first: Any, second: Any) -> bool:
    left, right = _array(first), _array(second)
    return left.shape == right.shape and bool(np.allclose(left, right, equal_nan=True))


def _compare_labels(first: Any, second: Any) -> tuple[bool, bool]:
    if isinstance(first, TechnicalLabels) and isinstance(second, TechnicalLabels):
        return (
            _compare_arrays(first.values, second.values),
            bool(np.array_equal(first.valid, second.valid)),
        )
    if isinstance(first, dict) and isinstance(second, dict) and {"values", "valid"} <= first.keys() <= second.keys():
        return (
            _compare_arrays(first["values"], second["values"]),
            bool(np.array_equal(_array(first["valid"]), _array(second["valid"]))),
        )
    first_valid = getattr(first, "valid", None)
    second_valid = getattr(second, "valid", None)
    values_same = _compare_arrays(first, second)
    validity_same = (
        True
        if first_valid is None or second_valid is None
        else bool(np.array_equal(_array(first_valid), _array(second_valid)))
    )
    return values_same, validity_same


def _call_window_builder(builder: Callable[..., Any], frame: pd.DataFrame, cutoff: int) -> Any:
    """Call a window builder with the cutoff when its contract supports it."""
    try:
        parameters = inspect.signature(builder).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "cutoff" in parameters or any(item.kind == inspect.Parameter.VAR_POSITIONAL for item in parameters.values()):
        return builder(frame, cutoff=cutoff)
    return builder(frame)


def _changed_future(frame: pd.DataFrame, cutoff: int, *, rng: np.random.Generator | None = None) -> pd.DataFrame:
    changed = frame.copy()
    rng = rng or np.random.default_rng(0)
    columns = [
        column for column in ("open", "high", "low", "close", "volume", "amount", "turnover") if column in changed
    ]
    if cutoff + 1 < len(changed) and columns:
        future_index = changed.index[cutoff + 1 :]
        mode = int(rng.integers(0, 5))
        # Future perturbations are floating-point transforms.  Cast the
        # copied raw-bar columns before assignment so pandas does not emit a
        # warning (or change behavior in a future pandas release) for writing
        # float values back into integer columns.
        changed[columns] = changed[columns].astype(float)
        values = changed.loc[future_index, columns].astype(float).to_numpy()
        if mode == 0:
            values *= 1000.0
        elif mode == 1:
            if "volume" in columns:
                values[:, columns.index("volume")] *= 0.01
            if "amount" in columns:
                values[:, columns.index("amount")] *= 0.01
            for column in ("open", "high", "low", "close"):
                if column in columns:
                    values[:, columns.index(column)] *= 0.01
        elif mode == 2:
            values = values[::-1]
        elif mode == 3:
            values *= 1.0 + rng.normal(0.0, 2.0, size=values.shape)
        else:
            values = values[rng.permutation(len(values))]
        changed.loc[future_index, columns] = values
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
            "status": "NOT_EVALUATED",
            "cases": 0,
            "feature_violations": 0,
            "label_violations": 0,
            "label_value_violations": 0,
            "label_validity_violations": 0,
            "window_violations": 0,
            "model_output_violations": 0,
            "total_violations": 1,
            "passed": False,
            "reason": "NO_REAL_INPUT",
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
    counts.update({"label_value_violations": 0, "label_validity_violations": 0})
    for frame, cutoff in candidates:
        changed = _changed_future(frame, cutoff, rng=rng)
        original_features = feature_builder(frame)
        changed_features = feature_builder(changed)
        original_labels = label_builder(frame)
        changed_labels = label_builder(changed)
        feature_same = _compare_arrays(
            _historical(original_features, cutoff + 1), _historical(changed_features, cutoff + 1)
        )
        label_values_same, label_validity_same = _compare_labels(
            _historical(original_labels, cutoff + 1),
            _historical(changed_labels, cutoff + 1),
        )
        counts["feature_violations"] += int(not feature_same)
        counts["label_value_violations"] += int(not label_values_same)
        counts["label_validity_violations"] += int(not label_validity_same)
        counts["label_violations"] += int(not (label_values_same and label_validity_same))
        if window_builder is not None:
            first_window = _call_window_builder(window_builder, frame, cutoff)
            second_window = _call_window_builder(window_builder, changed, cutoff)
            window_same = _compare_arrays(first_window, second_window)
            counts["window_violations"] += int(not window_same)
            if model is not None:
                model.eval()
                model_device = next(model.parameters()).device
                with torch.no_grad():
                    first_tensor = (
                        first_window if isinstance(first_window, torch.Tensor) else torch.as_tensor(first_window)
                    )
                    second_tensor = (
                        second_window if isinstance(second_window, torch.Tensor) else torch.as_tensor(second_window)
                    )
                    if first_tensor.ndim == 2:
                        first_tensor = first_tensor.unsqueeze(0)
                    if second_tensor.ndim == 2:
                        second_tensor = second_tensor.unsqueeze(0)
                    first_tensor = first_tensor.to(model_device)
                    second_tensor = second_tensor.to(model_device)
                    first_output = model(first_tensor)
                    second_output = model(second_tensor)
                output_same = _outputs_equal(first_output, second_output)
                counts["model_output_violations"] += int(not output_same)
    total = int(
        counts["feature_violations"]
        + counts["label_value_violations"]
        + counts["label_validity_violations"]
        + counts["window_violations"]
        + counts["model_output_violations"]
    )
    return {
        "status": "EVALUATED",
        "cases": len(candidates),
        **counts,
        "total_violations": total,
        "passed": bool(candidates) and total == 0,
    }


def _outputs_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return bool(torch.allclose(first, second, rtol=1e-6, atol=1e-6, equal_nan=True))
    if isinstance(first, dict) and isinstance(second, dict) and first.keys() == second.keys():
        return all(_outputs_equal(first[key], second[key]) for key in first)
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return _compare_arrays(first, second)
    return first == second
