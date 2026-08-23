from __future__ import annotations

import numpy as np

from stock_factor.technical_transformer.evaluation.embedding_probe import gold_neighbor_semantic_hit, weak_phase_neighbor_hit


def test_weak_and_gold_neighbor_metrics_are_separately_named() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    phase = np.eye(3)
    gold_events = np.asarray([[1, 0], [1, 0], [0, 1]])
    assert weak_phase_neighbor_hit(embeddings, phase, k=1) is not None
    assert gold_neighbor_semantic_hit(embeddings, gold_events) == 1.0

