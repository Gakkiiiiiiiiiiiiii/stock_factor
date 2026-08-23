from __future__ import annotations

import json
from pathlib import Path

from stock_factor.technical_transformer.evaluation.reliability_gate import promote_checkpoint
from stock_factor.technical_transformer.evaluation.report import build_reliability_report


def _split() -> dict:
    return {"sample_count": 10, "technical_composite": 1.0, "summary": {
        "ma_slope_mean_pearson": 1.0, "ma_slope_mean_sign_accuracy": 1.0, "ma_alignment_spearman": 1.0, "ma_compression_pearson": 1.0,
        "bollinger_percent_b_pearson": 1.0, "bollinger_boll_zscore_pearson": 1.0, "bollinger_bandwidth_pearson": 1.0,
        "bollinger_squeeze_spearman": 1.0, "bollinger_expansion_spearman": 1.0,
        "wyckoff_primitive_mean_spearman": 1.0, "phase_macro_f1": 1.0, "phase_js_divergence": 0.0, "phase_ece": 0.0,
        "event_mean_relative_pr": 1.0, "event_mean_f1": 1.0, "event_precision_at_top1pct": 1.0, "event_precision_at_top5pct": 1.0,
    }}


def test_complete_production_fixture_passes_and_promotes() -> None:
    checkpoint = {"checkpoint_id": "c", "dataset_id": "d"}
    manifest = {"dataset_id": "d", "source_market_snapshot_id": "s", "feature_schema_version": "f", "label_schema_version": "l", "split_overlap": 0, "leakage_audit": {"violations": []}}
    report = build_reliability_report(
        checkpoint_identity=checkpoint, dataset_manifest=manifest,
        splits={"valid": _split(), "time_test": _split(), "instrument_test": _split(), "double_oos": _split()},
        mode="PRODUCTION", causality={"status": "EVALUATED", "total_violations": 0},
        gold_set={"status": "EVALUATED", "passed": True, "kappa": {"events": 1.0}, "allowed_splits": ["double_oos"], "allowed_split_passed": True, "coverage_passed": True, "phase": {"macro_f1": 1.0}, "event": {"pr_auc_multiple_of_prevalence": 2.0}},
        embedding={"status": "EVALUATED", "passed": True, "weak_phase_neighbor_hit": 1.0, "gold_neighbor_semantic_hit": 1.0},
        invariance={"status": "EVALUATED", "passed": True, "raw_source_available": True, "feature_noise_invariance": {"cosine": 1.0}, "raw_noise_invariance": {"embedding_cosine": 1.0, "phase_js_divergence": 0.0, "event_median_probability_delta": 0.0}, "price_scale_cosine": 1.0},
        baseline={"status": "EVALUATED", "passed": True, "transformer_double_oos_composite_relative_gain": 0.1},
    )
    assert report["reliability_gate"]["status"] == "PASS"
    checkpoint_dir = Path(__file__).parent / "_promotion_fixture_tmp"
    checkpoint_dir.mkdir(exist_ok=True)
    try:
        (checkpoint_dir / "checkpoint_manifest.json").write_text(json.dumps({**checkpoint, "checkpoint_status": "CANDIDATE"}), encoding="utf-8")
        promoted = promote_checkpoint(checkpoint_dir, "ACTIVE", report)
        assert promoted["checkpoint_status"] == "ACTIVE"
    finally:
        (checkpoint_dir / "checkpoint_manifest.json").unlink(missing_ok=True)
        checkpoint_dir.rmdir()
