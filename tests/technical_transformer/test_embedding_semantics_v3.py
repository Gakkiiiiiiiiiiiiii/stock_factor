from __future__ import annotations

import numpy as np

from stock_factor.technical_transformer.evaluation.embedding_probe import (
    gold_neighbor_semantic_hit,
    weak_phase_neighbor_hit,
)
from stock_factor.technical_transformer.evaluation.gold_evaluator import _gold_neighbor_semantic_hit


def test_weak_and_gold_neighbor_metrics_are_separately_named() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    phase = np.eye(3)
    gold_events = np.asarray([[1, 0], [1, 0], [0, 1]])
    assert weak_phase_neighbor_hit(embeddings, phase, k=1) is not None
    assert gold_neighbor_semantic_hit(embeddings, gold_events) == 1.0


def test_gold_neighbor_excludes_singleton_event_anchor() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    labels = np.asarray([[1, 0], [1, 0], [0, 1]])
    assert gold_neighbor_semantic_hit(embeddings, labels) == 1.0


def test_gold_neighbor_returns_none_when_every_anchor_is_singleton() -> None:
    embeddings = np.eye(3)
    labels = np.eye(3, dtype=int)
    assert gold_neighbor_semantic_hit(embeddings, labels) is None


def test_gold_evaluator_uses_same_singleton_anchor_semantics() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    labels = [{"spring_score": 1}, {"spring_score": 1}, {"sos_score": 1}]
    assert _gold_neighbor_semantic_hit(embeddings, labels) == 1.0
