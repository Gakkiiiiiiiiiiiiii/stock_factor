from __future__ import annotations

from stock_factor.technical_transformer.training.selection import select_stage_score


def test_epoch_with_lower_mask_loss_wins() -> None:
    assert select_stage_score("masked_pretraining", {"mask": 0.5}).score > select_stage_score("masked_pretraining", {"mask": 0.6}).score
