"""Promotion gate orchestration for candidate-level evidence."""

from __future__ import annotations

from stock_factor.engine.promotion_gate import evaluate_promotion_gate


def evaluate_candidate_promotion(**evidence) -> dict:
    """Evaluate the promotion/tradability state without persistence side effects."""
    return evaluate_promotion_gate(**evidence).model_dump()


__all__ = ["evaluate_candidate_promotion"]
