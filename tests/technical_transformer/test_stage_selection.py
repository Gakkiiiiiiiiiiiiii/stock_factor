from __future__ import annotations

from stock_factor.technical_transformer.training.selection import select_stage_score


def _metrics(ma: float, boll: float, primitive: float, phase: float, event: float) -> dict:
    return {"summary": {
        "ma_slope_mean_pearson": ma, "ma_slope_mean_sign_accuracy": ma, "ma_alignment_spearman": ma, "ma_compression_pearson": ma,
        "bollinger_percent_b_pearson": boll, "bollinger_boll_zscore_pearson": boll, "bollinger_bandwidth_pearson": boll,
        "bollinger_squeeze_spearman": boll, "bollinger_expansion_spearman": boll,
        "wyckoff_primitive_mean_spearman": primitive, "phase_macro_f1": phase, "phase_js_divergence": 0.0, "phase_ece": 0.0,
        "event_mean_relative_pr": event, "event_mean_f1": event, "event_precision_at_top1pct": event, "event_precision_at_top5pct": event,
    }}


def test_stage_d_selection_uses_sequence_event_score() -> None:
    epoch_one = select_stage_score("wyckoff_phase_events", _metrics(0.99, 0.99, 0.10, 0.10, 0.10))
    epoch_two = select_stage_score("wyckoff_phase_events", _metrics(0.90, 0.90, 0.80, 0.70, 0.80))
    assert epoch_two.score > epoch_one.score


def test_masked_pretraining_selection_is_negative_mask_loss() -> None:
    first = select_stage_score("masked_pretraining", {"mask": 0.8})
    second = select_stage_score("masked_pretraining", {"mask": 0.5})
    assert second.score > first.score
