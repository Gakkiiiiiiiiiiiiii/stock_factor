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


def compose_alpha_scores_with_evidence(
    panel: dict[str, np.ndarray], factors: list[dict]
) -> tuple[np.ndarray | None, list[dict]]:
    """带逐因子贡献分解的 alpha 合成（设计文档 §14.2）。

    返回 (scores, contributions)：contributions[i] = {"factor_id", "column"}，
    column 为该因子最新截面的 rank 值向量（等权合成的证据来源）。
    """
    contributions: list[dict] = []
    for factor in factors:
        values = StackVM().execute(factor.get("rpn") or [], panel)
        if values is None:
            continue
        latest = rank(values[:, -1])
        if np.isnan(latest).all():
            continue
        contributions.append(
            {
                "factor_id": factor.get("factor_id") or "",
                "column": latest,
            }
        )
    if not contributions:
        return None, []
    matrix = np.vstack([item["column"] for item in contributions])
    scores = np.nanmean(matrix, axis=0)
    return scores, contributions
