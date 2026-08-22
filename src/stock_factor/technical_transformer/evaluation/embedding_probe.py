from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .metrics import pearson, spearman


def _split(x: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    cut = max(1, int(len(x) * 0.8))
    train, test = indices[:cut], indices[cut:]
    return x[train], x[test], y[train], y[test]


def linear_regression_probe(embedding: np.ndarray, target: np.ndarray, *, seed: int = 42, ridge: float = 1e-3) -> dict[str, float]:
    x = np.asarray(embedding, dtype=float); y = np.asarray(target, dtype=float)
    train_x, test_x, train_y, test_y = _split(x, y, seed)
    design = np.column_stack([train_x, np.ones(len(train_x))])
    weights = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ train_y)
    prediction = np.column_stack([test_x, np.ones(len(test_x))]) @ weights
    return {"pearson": pearson(prediction, test_y), "spearman": spearman(prediction, test_y), "mae": float(np.mean(np.abs(prediction - test_y)))}


def classification_probe(embedding: np.ndarray, target: np.ndarray, *, seed: int = 42, steps: int = 200) -> dict[str, float]:
    x = np.asarray(embedding, dtype=float); raw_y = np.asarray(target)
    y = np.argmax(raw_y, axis=1) if raw_y.ndim == 2 else (raw_y > 0.5).astype(int)
    train_x, test_x, train_y, test_y = _split(x, y, seed)
    classes = int(np.max(y)) + 1
    if classes <= 2:
        classes = 2
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.01, (train_x.shape[1], classes)); bias = np.zeros(classes)
    for _ in range(steps):
        logits = train_x @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits); probability /= probability.sum(axis=1, keepdims=True)
        one_hot = np.eye(classes)[train_y]
        gradient = probability - one_hot
        weights -= 0.05 * (train_x.T @ gradient / max(len(train_x), 1))
        bias -= 0.05 * gradient.mean(axis=0)
    predicted = np.argmax(test_x @ weights + bias, axis=1)
    return {"accuracy": float(np.mean(predicted == test_y)), "macro_f1": _macro_f1(test_y, predicted, classes)}


def _macro_f1(actual: np.ndarray, predicted: np.ndarray, classes: int) -> float:
    values = []
    for cls in range(classes):
        tp = np.sum((actual == cls) & (predicted == cls)); fp = np.sum((actual != cls) & (predicted == cls)); fn = np.sum((actual == cls) & (predicted != cls))
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(values))


def run_embedding_probe(
    embedding: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    raw_features: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Freeze embeddings and compare simple probes with requested baselines."""
    result: dict[str, Any] = {"frozen": True, "tasks": {}}
    random_embedding = np.random.default_rng(seed).normal(size=np.asarray(embedding).shape)
    for name, target in targets.items():
        task = {"technical_embedding": linear_regression_probe(embedding, target, seed=seed)}
        if np.asarray(target).ndim == 2 and np.allclose(np.asarray(target).sum(axis=1), 1.0, atol=1e-3):
            task["technical_embedding_classification"] = classification_probe(embedding, target, seed=seed)
            task["random_embedding_classification"] = classification_probe(random_embedding, target, seed=seed)
        else:
            task["random_embedding"] = linear_regression_probe(random_embedding, target, seed=seed)
        if raw_features is not None:
            task["last_day_raw_features"] = linear_regression_probe(np.asarray(raw_features), target, seed=seed)
        result["tasks"][name] = task
    return result


def nearest_neighbor_audit(
    embeddings: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    semantic_match: Callable[[int, int], bool] | None = None,
    k: int = 20,
) -> dict[str, Any]:
    values = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    hits = []
    neighbors: list[list[int]] = []
    for index in range(len(values)):
        scores = normalized @ normalized[index]
        scores[index] = -np.inf
        top = np.argsort(-scores)[: min(k, max(0, len(values) - 1))].tolist()
        neighbors.append(top)
        if semantic_match is not None:
            hits.extend(bool(semantic_match(index, item)) for item in top)
        elif labels is not None:
            hits.extend(bool(np.array_equal(labels[index], labels[item])) for item in top)
    return {
        "k": k, "neighbors": neighbors,
        "semantic_hit_rate": float(np.mean(hits)) if hits else None,
        "nearest_neighbor_semantic_hit": float(np.mean(hits)) if hits else None,
    }
