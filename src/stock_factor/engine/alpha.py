from __future__ import annotations

import numpy as np

from stock_factor.engine.fitness import rank
from stock_factor.engine.vm import StackVM


def compose_alpha_scores(panel: dict[str, np.ndarray], factors: list[dict]) -> tuple[np.ndarray | None, int]:
    columns = []
    for factor in factors:
        values = StackVM().execute(factor.get("rpn") or [], panel)
        if values is not None:
            latest = rank(values[:, -1])
            if not np.isnan(latest).all():
                columns.append(latest)
    return (np.nanmean(np.vstack(columns), axis=0), len(columns)) if columns else (None, 0)
