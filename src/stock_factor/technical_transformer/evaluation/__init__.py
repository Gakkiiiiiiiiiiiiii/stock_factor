"""Formal Technical Transformer reliability evaluation and lifecycle gates."""

from .leakage import audit_shortcut_leakage
from .reliability_gate import evaluate_reliability_gate
from .report import build_reliability_report


def run_reliability_evaluation(*args, **kwargs):
    """Load the full runner lazily to keep training/inference imports acyclic."""
    from .run import run_reliability_evaluation as _run_reliability_evaluation

    return _run_reliability_evaluation(*args, **kwargs)


__all__ = [
    "audit_shortcut_leakage",
    "evaluate_reliability_gate",
    "build_reliability_report",
    "run_reliability_evaluation",
]
