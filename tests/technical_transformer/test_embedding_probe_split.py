from __future__ import annotations

import numpy as np

from stock_factor.technical_transformer.evaluation.embedding_probe import run_embedding_probe


def test_embedding_probe_accepts_explicit_train_and_oos_arrays() -> None:
    train = np.arange(40, dtype=float).reshape(20, 2)
    test = np.arange(40, 60, dtype=float).reshape(10, 2)
    result = run_embedding_probe(
        train,
        {"ma_alignment": np.arange(20, dtype=float)},
        train_embedding=train,
        test_embedding=test,
        train_targets={"ma_alignment": np.arange(20, dtype=float)},
        test_targets={"ma_alignment": np.arange(10, dtype=float)},
    )
    assert result["split"]["test"] == "explicit_or_chronological"
