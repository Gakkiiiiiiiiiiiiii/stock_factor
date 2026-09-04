from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch


def cosine_similarity(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    x = np.asarray(a.detach().cpu() if isinstance(a, torch.Tensor) else a, dtype=float).reshape(-1)
    y = np.asarray(b.detach().cpu() if isinstance(b, torch.Tensor) else b, dtype=float).reshape(-1)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else 0.0


def transform_price_scale(frame: pd.DataFrame, scale: float = 10.0) -> pd.DataFrame:
    result = frame.copy()
    for column in ("open", "high", "low", "close", "amount"):
        if column in result:
            result[column] = result[column].astype(float) * scale
    return result


def add_small_ohlc_noise(frame: pd.DataFrame, *, scale: float = 0.00075, seed: int = 42) -> pd.DataFrame:
    result = frame.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, scale, (len(result), 4))
    for offset, column in enumerate(("open", "high", "low", "close")):
        if column in result:
            result[column] = result[column].astype(float) * (1.0 + noise[:, offset])
    if {"open", "high", "low", "close"} <= set(result.columns):
        result["high"] = np.maximum.reduce([result["high"], result["open"], result["close"]])
        result["low"] = np.minimum.reduce([result["low"], result["open"], result["close"]])
    if "amount" in result:
        result["amount"] = result["amount"].astype(float) * (1.0 + noise[:, 3])
    return result


@torch.no_grad()
def model_embedding_invariance(model: torch.nn.Module, original: torch.Tensor, transformed: torch.Tensor) -> float:
    model.eval()
    first = model(original)["technical_embedding"]
    second = model(transformed)["technical_embedding"]
    return cosine_similarity(first, second)


def price_scale_invariance(
    model: torch.nn.Module, original: torch.Tensor, scaled: torch.Tensor, *, minimum: float = 0.98
) -> dict[str, float | bool]:
    cosine = model_embedding_invariance(model, original, scaled)
    return {"cosine": cosine, "minimum": minimum, "passed": cosine >= minimum}


@torch.no_grad()
def model_event_probability_delta(model: torch.nn.Module, original: torch.Tensor, transformed: torch.Tensor) -> float:
    model.eval()
    first = torch.sigmoid(model(original)["events"])
    second = torch.sigmoid(model(transformed)["events"])
    return float(torch.abs(first - second).median().cpu())


@torch.no_grad()
def model_phase_js_divergence(model: torch.nn.Module, original: torch.Tensor, transformed: torch.Tensor) -> float:
    model.eval()
    first = torch.softmax(model(original)["phase"], dim=-1)
    second = torch.softmax(model(transformed)["phase"], dim=-1)
    midpoint = 0.5 * (first + second)
    js = 0.5 * (first * (first.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=-1)
    js += 0.5 * (second * (second.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=-1)
    return float(js.mean().cpu())


def small_noise_stability(
    model: torch.nn.Module,
    original: torch.Tensor,
    noisy: torch.Tensor,
    *,
    cosine_minimum: float = 0.95,
    event_probability_delta_maximum: float = 0.10,
) -> dict[str, float | bool]:
    cosine = model_embedding_invariance(model, original, noisy)
    delta = model_event_probability_delta(model, original, noisy)
    return {
        "cosine": cosine,
        "event_probability_delta": delta,
        "cosine_minimum": cosine_minimum,
        "event_probability_delta_maximum": event_probability_delta_maximum,
        "passed": cosine >= cosine_minimum and delta < event_probability_delta_maximum,
    }


def _outputs_equal(first: object, second: object) -> bool:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return bool(torch.equal(first, second))
    if isinstance(first, dict) and isinstance(second, dict) and first.keys() == second.keys():
        return all(_outputs_equal(first[key], second[key]) for key in first)
    return first == second


def future_invariance(
    frame: pd.DataFrame,
    cutoff: int,
    *,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    label_builder: Callable[[pd.DataFrame], pd.DataFrame],
    window_builder: Callable[[pd.DataFrame], object] | None = None,
    model: torch.nn.Module | None = None,
) -> dict[str, bool | float]:
    """Change only future rows and verify every T-prefix artifact is stable."""
    changed = frame.copy()
    columns = [
        column for column in ("open", "high", "low", "close", "volume", "amount", "turnover") if column in changed
    ]
    changed.loc[cutoff:, columns] = changed.loc[cutoff:, columns].astype(float) * 1000.0
    original_features = feature_builder(frame).iloc[:cutoff].to_numpy()
    changed_features = feature_builder(changed).iloc[:cutoff].to_numpy()
    original_labels = label_builder(frame).iloc[:cutoff].to_numpy()
    changed_labels = label_builder(changed).iloc[:cutoff].to_numpy()
    result: dict[str, bool | float] = {
        "feature_unchanged": bool(np.allclose(original_features, changed_features, equal_nan=True)),
        "label_unchanged": bool(np.allclose(original_labels, changed_labels, equal_nan=True)),
    }
    if window_builder is not None:
        result["window_unchanged"] = (
            window_builder(frame.iloc[:cutoff]).__repr__() == window_builder(changed.iloc[:cutoff]).__repr__()
        )
    if model is not None and window_builder is not None:
        first = window_builder(frame.iloc[:cutoff])
        second = window_builder(changed.iloc[:cutoff])
        if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
            result["model_output_unchanged"] = _outputs_equal(model(first), model(second))
    result["passed"] = all(bool(value) for key, value in result.items() if key.endswith("unchanged"))
    return result
