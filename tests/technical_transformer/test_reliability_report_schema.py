from __future__ import annotations

import json
from pathlib import Path

from stock_factor.technical_transformer.evaluation.report import build_reliability_report, validate_reliability_report


def test_v2_report_has_required_sections() -> None:
    report = build_reliability_report(
        checkpoint_identity={"checkpoint_id": "c", "dataset_id": "d"},
        dataset_manifest={
            "dataset_id": "d",
            "source_market_snapshot_id": "s",
            "feature_schema_version": "f",
            "label_schema_version": "l",
            "split_overlap": 0,
            "leakage_audit": {"violations": []},
        },
        splits={},
        mode="SMOKE",
    )
    validate_reliability_report(report)
    schema = json.loads(Path("config/technical_reliability_report_v2.schema.json").read_text(encoding="utf-8"))
    assert "reliability_gate" in schema["required"]
