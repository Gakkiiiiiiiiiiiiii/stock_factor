from __future__ import annotations

from stock_factor.technical_transformer.evaluation.gold_set import validate_gold_set


def test_gold_set_without_agreement_evidence_is_invalid() -> None:
    assert validate_gold_set([])["passed"] is False
