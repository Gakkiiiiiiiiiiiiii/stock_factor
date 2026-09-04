from __future__ import annotations

import math

import numpy as np


def _pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(a, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _pair(a, b)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    """Return average ranks (ties receive the same rank)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _pair(a, b)
    return pearson(_rank(x), _rank(y)) if len(x) >= 2 else 0.0


def mae(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _pair(a, b)
    return float(np.mean(np.abs(x - y))) if len(x) else float("nan")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _pair(a, b)
    return float(np.sqrt(np.mean((x - y) ** 2))) if len(x) else float("nan")


def sign_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _pair(a, b)
    return float(np.mean(np.sign(x) == np.sign(y))) if len(x) else 0.0


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(prediction, target),
        "rmse": rmse(prediction, target),
        "pearson": pearson(prediction, target),
        "spearman": spearman(prediction, target),
        "sign_accuracy": sign_accuracy(prediction, target),
    }


def pr_auc(target: np.ndarray, score: np.ndarray) -> float:
    y, s = _pair(target, score)
    y = (y > 0.5).astype(int)
    positives = int(y.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    previous = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous) * precision))


def precision_recall_f1(target: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y, s = _pair(target, score)
    actual = y > 0.5
    predicted = s >= threshold
    tp = float(np.sum(actual & predicted))
    fp = float(np.sum(~actual & predicted))
    fn = float(np.sum(actual & ~predicted))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


def precision_at_top(target: np.ndarray, score: np.ndarray, fraction: float) -> float:
    y, s = _pair(target, score)
    if len(y) == 0:
        return 0.0
    count = max(1, int(math.ceil(len(y) * fraction)))
    indices = np.argsort(-s, kind="mergesort")[:count]
    return float(np.mean(y[indices] > 0.5))


def ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y, p = _pair(target, probability)
    if len(y) == 0:
        return 0.0
    result = 0.0
    for left, right in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= left) & (p <= right if right == 1 else p < right)
        if mask.any():
            result += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return result


def event_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y, p = _pair(target, probability)
    prevalence = float(np.mean(y > 0.5)) if len(y) else 0.0
    area = pr_auc(y, p)
    relative_pr = min(area / max(prevalence, 1e-4), 10.0) / 10.0
    result = {
        "pr_auc": area,
        "prevalence": prevalence,
        "pr_auc_multiple_of_prevalence": area / max(prevalence, 1e-12),
        "relative_pr": relative_pr,
        "ece": ece(y, p),
    }
    result.update(precision_recall_f1(y, p))
    result.update(
        {"precision_at_top1pct": precision_at_top(y, p, 0.01), "precision_at_top5pct": precision_at_top(y, p, 0.05)}
    )
    return result


def soft_phase_metrics(target: np.ndarray, logits_or_probability: np.ndarray) -> dict[str, object]:
    y = np.asarray(target, dtype=float)
    raw = np.asarray(logits_or_probability, dtype=float)
    if raw.ndim != 2:
        raise ValueError("phase arrays must be [samples, classes]")
    shifted = raw - raw.max(axis=1, keepdims=True)
    p = np.exp(shifted)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    y = y / np.maximum(y.sum(axis=1, keepdims=True), 1e-12)
    ce = float(np.mean(-np.sum(y * np.log(np.maximum(p, 1e-12)), axis=1)))
    kl = float(np.mean(np.sum(y * np.log(np.maximum(y, 1e-12) / np.maximum(p, 1e-12)), axis=1)))
    midpoint = 0.5 * (y + p)
    js = float(
        np.mean(
            0.5 * np.sum(y * np.log(np.maximum(y, 1e-12) / np.maximum(midpoint, 1e-12)), axis=1)
            + 0.5 * np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(midpoint, 1e-12)), axis=1)
        )
    )
    actual = np.argmax(y, axis=1)
    predicted = np.argmax(p, axis=1)
    classes = range(y.shape[1])
    f1s = []
    confusion = np.zeros((y.shape[1], y.shape[1]), dtype=int)
    for a, b in zip(actual, predicted):
        confusion[a, b] += 1
    for cls in classes:
        tp = np.sum((actual == cls) & (predicted == cls))
        fp = np.sum((actual != cls) & (predicted == cls))
        fn = np.sum((actual == cls) & (predicted != cls))
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1s.append(2 * pr * rc / max(pr + rc, 1e-12))
    confidence = p.max(axis=1)
    correctness = (actual == predicted).astype(float)
    return {
        "soft_ce": ce,
        "kl_divergence": kl,
        "js_divergence": js,
        "brier_score": float(np.mean(np.sum((p - y) ** 2, axis=1))),
        "ece": ece(correctness, confidence),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": confusion.tolist(),
    }
