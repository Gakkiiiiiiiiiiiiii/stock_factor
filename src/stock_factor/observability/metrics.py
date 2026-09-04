"""Bounded, low-cardinality metrics suitable for logs or a metrics adapter."""

from __future__ import annotations

import math
from collections import defaultdict
from threading import Lock
from typing import Mapping

ALLOWED_METRICS = frozenset(
    {
        "dataset_build_latency_seconds",
        "mining_latency_seconds",
        "oos_latency_seconds",
        "artifact_sealing_latency_seconds",
        "research_error_total",
        "research_queue_depth",
        "dataset_freshness_seconds",
        "research_readiness",
    }
)
ALLOWED_LABELS = frozenset({"profile", "mode", "stage", "status", "dependency", "outcome"})
ALLOWED_LABEL_VALUES = {
    "profile": frozenset({"dev", "test", "staging", "prod"}),
    "mode": frozenset({"FORMAL", "EXPLORATORY"}),
    "stage": frozenset({"prepare", "data", "evaluate", "oos", "promotion", "seal", "queued", "complete"}),
    "status": frozenset({"alive", "ready", "not_ready", "success", "error", "pass", "fail"}),
    "dependency": frozenset({"market", "content", "database", "artifact_store", "oos", "model", "quant"}),
    "outcome": frozenset({"success", "failure", "accepted", "rejected"}),
}


class LowCardinalityMetrics:
    def __init__(self) -> None:
        self._values: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._lock = Lock()

    def observe(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        self._validate(name, value, labels)
        key = (name, tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items())))
        with self._lock:
            self._values[key] += float(value)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {f"{name}|{dict(labels)}": value for (name, labels), value in self._values.items()}

    @staticmethod
    def _validate(name: str, value: float, labels: Mapping[str, str] | None) -> None:
        if name not in ALLOWED_METRICS:
            raise ValueError("metric name is not allowlisted")
        if not math.isfinite(float(value)):
            raise ValueError("metric value must be finite")
        labels = labels or {}
        if not set(labels).issubset(ALLOWED_LABELS):
            raise ValueError("metric labels are not allowlisted")
        if any(
            not str(value) or len(str(value)) > 32 or str(value) not in ALLOWED_LABEL_VALUES.get(key, frozenset())
            for key, value in labels.items()
        ):
            raise ValueError("metric label value is invalid or unbounded")


__all__ = ["ALLOWED_LABELS", "ALLOWED_LABEL_VALUES", "ALLOWED_METRICS", "LowCardinalityMetrics"]
