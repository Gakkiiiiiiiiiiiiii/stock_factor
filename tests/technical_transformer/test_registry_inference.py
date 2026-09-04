from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from stock_factor.technical_transformer.registry import ModelArtifact, ModelRegistry, RegistryError
from stock_factor.technical_transformer.training.inference import load_checkpoint, predict, predict_registered
from stock_factor.technical_transformer.training.train import TechnicalTransformerSystem

H = "a" * 64


def _report(checkpoint_sha256: str) -> dict:
    return {
        "report_version": "technical-reliability-report.v2",
        "mode": "PRODUCTION",
        "checkpoint": {
            "checkpoint_sha256": checkpoint_sha256,
            "feature_schema_hash": H,
            "label_schema_hash": "b" * 64,
            "market_snapshot_id": "snapshot-inference",
            "split_manifest_hash": "c" * 64,
        },
        "required_evidence": {
            "gold_set": True,
            "baseline": True,
            "causality": True,
            "invariance": True,
            "embedding_probe": True,
            "double_oos": True,
        },
        "reliability_gate": {"status": "PASS"},
        "data_integrity": {
            "causality": {"status": "EVALUATED", "total_violations": 0},
            "leakage": {"passed": True, "violations": []},
            "split": {"overlap": 0},
        },
        "gold_set": {"status": "EVALUATED", "passed": True},
        "baseline": {"status": "EVALUATED", "passed": True},
        "embedding": {"status": "EVALUATED", "passed": True},
        "invariance": {"status": "EVALUATED", "passed": True},
        "splits": {"double_oos": {"sample_count": 2}},
    }


def _fixture(tmp_path: Path) -> tuple[ModelRegistry, ModelArtifact, np.ndarray, Path, Path]:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    config = {
        "input_dim": 51,
        "hidden_size": 16,
        "layers": 1,
        "heads": 1,
        "ffn_size": 32,
        "dropout": 0.0,
        "embedding_dim": 8,
        "sequence_length": 4,
    }
    (directory / "model_config.json").write_text(json.dumps(config), encoding="utf-8")
    model = TechnicalTransformerSystem(config).eval()
    torch.save(model.state_dict(), directory / "model.pt")
    checkpoint_hash = hashlib.sha256((directory / "model.pt").read_bytes()).hexdigest()
    report = _report(checkpoint_hash)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    artifact = ModelArtifact.from_checkpoint(
        directory / "model.pt",
        model_id="inference-model",
        feature_schema_hash=H,
        label_schema_hash="b" * 64,
        training_snapshot_id="snapshot-inference",
        split_manifest_hash="c" * 64,
        reliability_report_hash=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        dependency_lock_hash=hashlib.sha256(b"offline-lock").hexdigest(),
        hardware_profile={"device": "cpu"},
        determinism_profile={
            "seed": 42,
            "deterministic_algorithms": True,
            "tolerance_abs": 1e-6,
            "tolerance_rel": 1e-5,
        },
        created_at=datetime(2026, 9, 4, tzinfo=UTC),
    )
    lock = tmp_path / "requirements.lock"
    lock.write_bytes(b"offline-lock")
    registry = ModelRegistry(
        promotion_policy={"hardware_tolerances": {"cpu": {"tolerance_abs": 1e-6, "tolerance_rel": 1e-5}}}
    )
    registry.register(artifact, checkpoint=directory / "model.pt")
    return registry, artifact, np.zeros((4, 51), dtype=np.float32), report_path, lock


def test_formal_inference_requires_promoted_registry_and_traces_record(tmp_path):
    registry, artifact, window, report_path, lock = _fixture(tmp_path)
    with pytest.raises(RegistryError, match="PROMOTED"):
        predict_registered(registry, artifact.model_id, window)
    exploratory = predict(load_checkpoint(tmp_path / "checkpoint"), window)
    assert exploratory["formal_ineligible"] is True
    with pytest.raises(ValueError, match="registered"):
        predict(load_checkpoint(tmp_path / "checkpoint"), window, model_artifact_id="sha256:" + "f" * 64)

    registry.evaluate(artifact.model_id, report_path)
    registry.seal(artifact.model_id)
    promoted = registry.promote(
        artifact.model_id,
        checkpoint=tmp_path / "checkpoint" / "model.pt",
        dependency_lock=lock,
        feature_schema_hash=H,
        label_schema_hash="b" * 64,
        training_snapshot_id="snapshot-inference",
        hardware_profile={"device": "cpu", "max_abs_error": 0.0, "max_rel_error": 0.0},
    )
    result = predict_registered(registry, artifact.model_id, window)
    assert result["formal_ineligible"] is False
    assert result["model_artifact_id"] == artifact.artifact_id
    assert result["record_id"] == promoted.record_id
    assert promoted.status.value == "PROMOTED"
    assert promoted.verify_record()
