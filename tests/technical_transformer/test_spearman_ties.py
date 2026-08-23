from __future__ import annotations

import numpy as np

from stock_factor.technical_transformer.evaluation.metrics import spearman


def test_spearman_uses_average_rank_for_ties() -> None:
    assert np.isclose(spearman(np.array([0, 0, 0, 1, 1]), np.array([0, 0, 0, 1, 1])), 1.0)
