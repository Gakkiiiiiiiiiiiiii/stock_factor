from __future__ import annotations

import pytest

from stock_factor.technical_transformer.evaluation.report import build_reliability_report


def _split(score: float) -> dict:
    return {"sample_count": 10, "technical_composite": score, "summary": {
        "ma_slope_mean_pearson": score, "ma_slope_mean_sign_accuracy": score, "ma_alignment_spearman": score, "ma_compression_pearson": score,
        "bollinger_percent_b_pearson": score, "bollinger_boll_zscore_pearson": score, "bollinger_bandwidth_pearson": score,
        "bollinger_squeeze_spearman": score, "bollinger_expansion_spearman": score,
        "wyckoff_primitive_mean_spearman": score, "phase_macro_f1": score, "phase_js_divergence": 0.0, "phase_ece": 0.0,
        "event_mean_relative_pr": score, "event_mean_f1": score, "event_precision_at_top1pct": score, "event_precision_at_top5pct": score,
    }}


def test_oos_degradation_uses_technical_composite() -> None:
    report = build_reliability_report(
        checkpoint_identity={"checkpoint_id": "c", "dataset_id": "d"},
        dataset_manifest={"dataset_id": "d", "source_market_snapshot_id": "s", "feature_schema_version": "f", "label_schema_version": "l", "split_overlap": 0, "leakage_audit": {"violations": []}},
        splits={"valid": _split(1.0), "time_test": _split(0.9), "instrument_test": _split(0.8), "double_oos": _split(0.7)},
        mode="RESEARCH", causality={"status": "EVALUATED", "total_violations": 0},
    )
    assert report["oos"]["time_degradation"] == pytest.approx(0.1)
    assert report["oos"]["double_oos_degradation"] == pytest.approx(0.3)
