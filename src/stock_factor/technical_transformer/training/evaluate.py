"""Compatibility entry points for frozen checkpoint evaluation."""

from ..evaluation.evaluator import EvaluationResult, evaluate_all_splits, evaluate_checkpoint, evaluate_split, freeze_model

__all__ = ["EvaluationResult", "evaluate_all_splits", "evaluate_checkpoint", "evaluate_split", "freeze_model"]
