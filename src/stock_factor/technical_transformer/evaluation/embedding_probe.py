from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .metrics import pearson, spearman


@dataclass(frozen=True)
class LinearProbe:
    weights: np.ndarray
    ridge: float


@dataclass(frozen=True)
class ClassificationProbe:
    weights: np.ndarray
    bias: np.ndarray
    classes: int


def _design(values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.asarray(values, dtype=float), np.ones(len(values))])


def fit_linear_probe(train_embedding: np.ndarray, train_target: np.ndarray, *, ridge: float = 1e-3) -> LinearProbe:
    x = _design(train_embedding)
    y = np.asarray(train_target, dtype=float).reshape(-1)
    weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    return LinearProbe(weights=weights, ridge=float(ridge))


def evaluate_linear_probe(probe: LinearProbe, test_embedding: np.ndarray, test_target: np.ndarray) -> dict[str, float]:
    x = _design(test_embedding)
    y = np.asarray(test_target, dtype=float).reshape(-1)
    prediction = x @ probe.weights
    return {
        "pearson": pearson(prediction, y),
        "spearman": spearman(prediction, y),
        "mae": float(np.mean(np.abs(prediction - y))),
    }


def _class_ids(target: np.ndarray) -> np.ndarray:
    values = np.asarray(target)
    return np.argmax(values, axis=1) if values.ndim == 2 else (values > 0.5).astype(int)


def fit_classification_probe(
    train_embedding: np.ndarray,
    train_target: np.ndarray,
    *,
    steps: int = 200,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> ClassificationProbe:
    x = np.asarray(train_embedding, dtype=float)
    y = _class_ids(np.asarray(train_target))
    classes = max(2, int(np.max(y)) + 1 if len(y) else 2)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.01, (x.shape[1], classes))
    bias = np.zeros(classes)
    for _ in range(steps):
        logits = x @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
        gradient = probability - np.eye(classes)[y]
        weights -= learning_rate * (x.T @ gradient / max(len(x), 1))
        bias -= learning_rate * gradient.mean(axis=0)
    return ClassificationProbe(weights=weights, bias=bias, classes=classes)


def _macro_f1(actual: np.ndarray, predicted: np.ndarray, classes: int) -> float:
    values = []
    for cls in range(classes):
        tp = np.sum((actual == cls) & (predicted == cls))
        fp = np.sum((actual != cls) & (predicted == cls))
        fn = np.sum((actual == cls) & (predicted != cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(values))


def evaluate_classification_probe(
    probe: ClassificationProbe, test_embedding: np.ndarray, test_target: np.ndarray
) -> dict[str, float]:
    actual = _class_ids(np.asarray(test_target))
    predicted = np.argmax(np.asarray(test_embedding, dtype=float) @ probe.weights + probe.bias, axis=1)
    return {
        "accuracy": float(np.mean(predicted == actual)) if len(actual) else 0.0,
        "macro_f1": _macro_f1(actual, predicted, probe.classes) if len(actual) else 0.0,
    }


def _is_classification(target: np.ndarray) -> bool:
    values = np.asarray(target)
    if values.ndim == 2:
        return True
    unique = np.unique(values[np.isfinite(values)])
    return len(unique) <= 2 and set(unique.tolist()).issubset({0.0, 1.0})


def _chronological_split(
    values: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cut = min(max(1, int(len(values) * 0.8)), max(1, len(values) - 1))
    return values[:cut], values[cut:], target[:cut], target[cut:]


def linear_regression_probe(
    embedding: np.ndarray, target: np.ndarray, *, seed: int = 42, ridge: float = 1e-3
) -> dict[str, float]:
    """Compatibility helper using a chronological, never-random split."""
    train_x, test_x, train_y, test_y = _chronological_split(np.asarray(embedding), np.asarray(target))
    return evaluate_linear_probe(fit_linear_probe(train_x, train_y, ridge=ridge), test_x, test_y)


def classification_probe(
    embedding: np.ndarray, target: np.ndarray, *, seed: int = 42, steps: int = 200
) -> dict[str, float]:
    train_x, test_x, train_y, test_y = _chronological_split(np.asarray(embedding), np.asarray(target))
    return evaluate_classification_probe(
        fit_classification_probe(train_x, train_y, steps=steps, seed=seed), test_x, test_y
    )


def run_embedding_probe(
    embedding: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    train_embedding: np.ndarray | None = None,
    test_embedding: np.ndarray | None = None,
    train_targets: dict[str, np.ndarray] | None = None,
    test_targets: dict[str, np.ndarray] | None = None,
    train_valid: dict[str, np.ndarray] | None = None,
    test_valid: dict[str, np.ndarray] | None = None,
    raw_features: np.ndarray | None = None,
    train_raw_features: np.ndarray | None = None,
    test_raw_features: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit on train embeddings and evaluate on fixed OOS embeddings."""
    values = np.asarray(embedding, dtype=float)
    if train_embedding is None or test_embedding is None:
        cut = min(max(1, int(len(values) * 0.8)), max(1, len(values) - 1))
        train_embedding, test_embedding = values[:cut], values[cut:]
        train_targets = {name: np.asarray(target)[:cut] for name, target in targets.items()}
        test_targets = {name: np.asarray(target)[cut:] for name, target in targets.items()}
        if raw_features is not None:
            train_raw_features, test_raw_features = np.asarray(raw_features)[:cut], np.asarray(raw_features)[cut:]
    else:
        train_targets = train_targets or targets
        test_targets = test_targets or targets
    result: dict[str, Any] = {
        "frozen": True,
        "split": {"train": "explicit_or_chronological", "test": "explicit_or_chronological"},
        "tasks": {},
    }
    for name, target in targets.items():
        train_target = np.asarray(train_targets[name])
        test_target = np.asarray(test_targets[name])
        train_x = np.asarray(train_embedding)
        test_x = np.asarray(test_embedding)
        train_raw = np.asarray(train_raw_features) if train_raw_features is not None else None
        test_raw = np.asarray(test_raw_features) if test_raw_features is not None else None
        if train_valid is not None and name in train_valid:
            mask = np.asarray(train_valid[name], dtype=bool)
            train_x = train_x[mask]
            train_target = train_target[mask]
            if train_raw is not None:
                train_raw = train_raw[mask]
        if test_valid is not None and name in test_valid:
            mask = np.asarray(test_valid[name], dtype=bool)
            test_x = test_x[mask]
            test_target = test_target[mask]
            if test_raw is not None:
                test_raw = test_raw[mask]
        if len(train_x) == 0 or len(test_x) == 0:
            result["tasks"][name] = {"status": "NO_VALID_SAMPLES"}
            continue
        random_train = np.random.default_rng(seed).normal(size=train_x.shape)
        random_test = np.random.default_rng(seed + 1).normal(size=test_x.shape)
        if _is_classification(train_target):
            technical_probe = fit_classification_probe(train_x, train_target, seed=seed)
            random_probe = fit_classification_probe(random_train, train_target, seed=seed)
            task = {
                "technical_embedding": evaluate_classification_probe(technical_probe, test_x, test_target),
                "random_embedding": evaluate_classification_probe(random_probe, random_test, test_target),
            }
            if train_raw is not None and test_raw is not None:
                raw_probe = fit_classification_probe(train_raw, train_target, seed=seed)
                task["last_day_raw_features"] = evaluate_classification_probe(raw_probe, test_raw, test_target)
        else:
            technical_probe = fit_linear_probe(train_x, train_target)
            random_probe = fit_linear_probe(random_train, train_target)
            task = {
                "technical_embedding": evaluate_linear_probe(technical_probe, test_x, test_target),
                "random_embedding": evaluate_linear_probe(random_probe, random_test, test_target),
            }
            if train_raw is not None and test_raw is not None:
                raw_probe = fit_linear_probe(train_raw, train_target)
                task["last_day_raw_features"] = evaluate_linear_probe(raw_probe, test_raw, test_target)
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
    rate = float(np.mean(hits)) if hits else None
    return {"k": k, "neighbors": neighbors, "semantic_hit_rate": rate, "nearest_neighbor_semantic_hit": rate}


def weak_phase_neighbor_hit(embeddings: np.ndarray, phase_labels: np.ndarray, *, k: int = 20) -> float | None:
    """Nearest-neighbor agreement against dataset-derived weak phase labels."""
    values = np.asarray(phase_labels)
    valid = np.isfinite(values).all(axis=1) if values.ndim == 2 else np.isfinite(values)
    if not np.any(valid):
        return None
    labels = np.argmax(values[valid], axis=1) if values.ndim == 2 else (values[valid] > 0.5).astype(int)
    return nearest_neighbor_audit(np.asarray(embeddings)[valid], labels=labels, k=k).get("semantic_hit_rate")


def gold_neighbor_semantic_hit(embeddings: np.ndarray, event_labels: np.ndarray) -> float | None:
    """Nearest-neighbor agreement using frozen Gold event labels."""
    values = np.asarray(embeddings, dtype=float)
    labels = np.asarray(event_labels, dtype=int)
    if values.ndim != 2 or labels.ndim != 2 or len(values) < 2 or len(values) != len(labels):
        return None
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    hits: list[float] = []
    for index in range(len(values)):
        # A Gold anchor without any compatible peer is not an evaluable
        # nearest-neighbor query.  Excluding it avoids treating an inherently
        # untestable singleton event as a semantic miss.
        if not any(
            _dense_gold_semantic_match(labels[index], labels[peer]) for peer in range(len(labels)) if peer != index
        ):
            continue
        neighbor = int(np.argmax(similarity[index]))
        hits.append(float(_dense_gold_semantic_match(labels[index], labels[neighbor])))
    return float(np.mean(hits)) if hits else None


def _dense_gold_semantic_match(anchor: np.ndarray, candidate: np.ndarray) -> bool:
    positive = anchor > 0
    if np.any(positive):
        return bool(np.any(candidate[positive] > 0))
    return bool(np.array_equal(anchor, candidate))
