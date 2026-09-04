from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stock_factor.technical_transformer.registry import ModelArtifact, ModelArtifactStatus, ModelRegistry, RegistryError
from stock_factor.technical_transformer.registry.registry import hardware_compatible, verify_dependency_lock

H = "a" * 64
CHECKPOINT_SHA256 = "4f5ea895a373f8c374c2be7600aeed5ae7e15c98d0843fafbfbcf181b333e1af"


def _report() -> dict:
    return {
        "report_version": "technical-reliability-report.v2",
        "mode": "PRODUCTION",
        "checkpoint": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "feature_schema_hash": H,
            "label_schema_hash": "b" * 64,
            "market_snapshot_id": "snapshot-1",
            "split_manifest_hash": "c" * 64,
        },
        "required_evidence": {
            "gold_set": True,
            "baseline": True,
            "causality": True,
            "embedding_probe": True,
            "invariance": True,
            "double_oos": True,
        },
        "reliability_gate": {"status": "PASS"},
        "data_integrity": {
            "causality": {"status": "EVALUATED", "total_violations": 0},
            "leakage": {"status": "EVALUATED", "passed": True, "violations": []},
            "split": {"overlap": 0},
        },
        "gold_set": {"status": "EVALUATED", "passed": True},
        "baseline": {"status": "EVALUATED", "passed": True},
        "embedding": {"status": "EVALUATED", "passed": True},
        "invariance": {"status": "EVALUATED", "passed": True},
        "splits": {"double_oos": {"sample_count": 2}},
    }


def _artifact(tmp_path: Path, *, report_hash: str, model_id: str = "model-1") -> tuple[ModelArtifact, Path, Path]:
    checkpoint = tmp_path / f"{model_id}.weights"
    checkpoint.write_bytes(b"deterministic checkpoint bytes")
    lock = tmp_path / "requirements.lock"
    lock.write_text("torch==2.6.0\n", encoding="utf-8")
    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
    artifact = ModelArtifact.from_checkpoint(
        checkpoint,
        model_id=model_id,
        feature_schema_hash=H,
        label_schema_hash="b" * 64,
        training_snapshot_id="snapshot-1",
        split_manifest_hash="c" * 64,
        reliability_report_hash=report_hash,
        dependency_lock_hash=lock_hash,
        hardware_profile={"device": "cpu"},
        determinism_profile={
            "seed": 42,
            "deterministic_algorithms": True,
            "tolerance_abs": 1e-6,
            "tolerance_rel": 1e-5,
        },
        created_at=datetime(2026, 9, 4, tzinfo=UTC),
    )
    return artifact, checkpoint, lock


def _promote(registry: ModelRegistry, artifact: ModelArtifact, checkpoint: Path, lock: Path) -> ModelArtifact:
    registry.evaluate(artifact.model_id, _report())
    registry.seal(artifact.model_id)
    return registry.promote(
        artifact.model_id,
        checkpoint=checkpoint,
        dependency_lock=lock,
        feature_schema_hash=artifact.feature_schema_hash,
        label_schema_hash=artifact.label_schema_hash,
        training_snapshot_id=artifact.training_snapshot_id,
        hardware_profile={"device": "cpu", "max_abs_error": 0.0, "max_rel_error": 0.0},
    )


def test_registry_requires_full_production_reliability_and_tracks_append_only_audit(tmp_path):
    report = _report()
    report_path = tmp_path / "reliability.json"
    report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    artifact, checkpoint, lock = _artifact(tmp_path, report_hash=hashlib.sha256(report_path.read_bytes()).hexdigest())
    registry = ModelRegistry()
    registry.register(artifact, checkpoint=checkpoint)
    with pytest.raises(RegistryError, match="requires SEALED"):
        registry.promote(
            artifact.model_id,
            checkpoint=checkpoint,
            dependency_lock=lock,
            feature_schema_hash=H,
            label_schema_hash="b" * 64,
            training_snapshot_id="snapshot-1",
            hardware_profile={"device": "cpu"},
        )
    evaluated = registry.evaluate(artifact.model_id, report_path)
    assert evaluated.status == ModelArtifactStatus.EVALUATED
    sealed = registry.seal(artifact.model_id)
    assert sealed.status == ModelArtifactStatus.SEALED
    promoted = registry.promote(
        artifact.model_id,
        checkpoint=checkpoint,
        dependency_lock=lock,
        feature_schema_hash=H,
        label_schema_hash="b" * 64,
        training_snapshot_id="snapshot-1",
        hardware_profile={"device": "cpu", "max_abs_error": 0.0, "max_rel_error": 0.0},
    )
    assert promoted.status == ModelArtifactStatus.PROMOTED
    assert [event.to_status for event in registry.events()] == ["CANDIDATE", "EVALUATED", "SEALED", "PROMOTED"]


def test_smoke_or_failed_reliability_report_cannot_seal_or_promote(tmp_path):
    report = _report()
    report["mode"] = "SMOKE"
    report_path = tmp_path / "smoke.json"
    report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    artifact, checkpoint, _lock = _artifact(tmp_path, report_hash=hashlib.sha256(report_path.read_bytes()).hexdigest())
    registry = ModelRegistry()
    registry.register(artifact, checkpoint=checkpoint)
    registry.evaluate(artifact.model_id, report_path)
    with pytest.raises(RegistryError, match="PASS"):
        registry.seal(artifact.model_id)


def test_reliability_report_identity_is_bound_to_artifact(tmp_path):
    report = _report()
    report["checkpoint"]["feature_schema_hash"] = "d" * 64
    report_path = tmp_path / "wrong-identity.json"
    report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    artifact, checkpoint, _lock = _artifact(tmp_path, report_hash=hashlib.sha256(report_path.read_bytes()).hexdigest())
    registry = ModelRegistry()
    registry.register(artifact, checkpoint=checkpoint)
    with pytest.raises(RegistryError, match="identity"):
        registry.evaluate(artifact.model_id, report_path)


@pytest.mark.parametrize("missing", ["gold_set", "baseline", "embedding", "invariance"])
def test_reliability_report_missing_section_cannot_evaluate(tmp_path, missing):
    report = _report()
    report.pop(missing)
    report_path = tmp_path / f"missing-{missing}.json"
    report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    artifact, checkpoint, _lock = _artifact(tmp_path, report_hash=hashlib.sha256(report_path.read_bytes()).hexdigest())
    registry = ModelRegistry()
    registry.register(artifact, checkpoint=checkpoint)
    evaluated = registry.evaluate(artifact.model_id, report_path)
    assert evaluated.reliability_seal["status"] == "FAIL"
    with pytest.raises(RegistryError, match="PASS"):
        registry.seal(artifact.model_id)


@pytest.mark.parametrize("evidence", ["gold_set", "baseline", "embedding", "invariance"])
def test_reliability_report_evidence_must_explicitly_pass(tmp_path, evidence):
    report = _report()
    report[evidence]["passed"] = None
    report_path = tmp_path / f"not-passed-{evidence}.json"
    report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    artifact, checkpoint, _lock = _artifact(tmp_path, report_hash=hashlib.sha256(report_path.read_bytes()).hexdigest())
    registry = ModelRegistry()
    registry.register(artifact, checkpoint=checkpoint)
    evaluated = registry.evaluate(artifact.model_id, report_path)
    assert evaluated.reliability_seal["status"] == "FAIL"
    with pytest.raises(RegistryError, match="PASS"):
        registry.seal(artifact.model_id)


def test_checkpoint_tamper_and_illegal_transitions_are_fail_closed(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report(), sort_keys=True, separators=(",", ":")), encoding="utf-8")
    artifact, checkpoint, lock = _artifact(tmp_path, report_hash=hashlib.sha256(report_path.read_bytes()).hexdigest())
    registry = ModelRegistry()
    registry.register(artifact, checkpoint=checkpoint)
    with pytest.raises(RegistryError, match="PROMOTED"):
        registry.activate(artifact.model_id)
    registry.evaluate(artifact.model_id, report_path)
    registry.seal(artifact.model_id)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(RegistryError, match="checkpoint hash"):
        registry.promote(
            artifact.model_id,
            checkpoint=checkpoint,
            dependency_lock=lock,
            feature_schema_hash=H,
            label_schema_hash="b" * 64,
            training_snapshot_id="snapshot-1",
            hardware_profile={"device": "cpu"},
        )


def test_content_addressing_duplicate_model_and_rollback_keep_history(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report(), sort_keys=True, separators=(",", ":")), encoding="utf-8")
    report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
    artifact, checkpoint, lock = _artifact(tmp_path, report_hash=report_hash)
    registry = ModelRegistry()
    assert registry.register(artifact, checkpoint=checkpoint) == artifact
    assert registry.register(artifact, checkpoint=checkpoint) == artifact
    other_checkpoint = tmp_path / "other.weights"
    other_checkpoint.write_bytes(b"different checkpoint bytes")
    other = ModelArtifact.from_checkpoint(
        other_checkpoint,
        model_id="model-1",
        feature_schema_hash=H,
        label_schema_hash="b" * 64,
        training_snapshot_id="snapshot-1",
        split_manifest_hash="c" * 64,
        reliability_report_hash=report_hash,
        dependency_lock_hash=hashlib.sha256((tmp_path / "requirements.lock").read_bytes()).hexdigest(),
        hardware_profile={"device": "cpu"},
        determinism_profile={
            "seed": 42,
            "deterministic_algorithms": True,
            "tolerance_abs": 1e-6,
            "tolerance_rel": 1e-5,
        },
        created_at=datetime(2026, 9, 4, tzinfo=UTC),
    )
    with pytest.raises(RegistryError, match="append-only"):
        registry.register(other)
    promoted = _promote(registry, artifact, checkpoint, lock)
    registry.activate(promoted.model_id)
    assert registry.active().model_id == "model-1"
    assert registry.rollback(promoted.model_id).status == ModelArtifactStatus.PROMOTED
    retired = registry.retire(promoted.model_id)
    assert retired.status == ModelArtifactStatus.RETIRED
    with pytest.raises(RegistryError, match="PROMOTED"):
        registry.rollback(promoted.model_id)
    assert len(registry.events()) >= 6
    assert [item.status for item in registry.history(promoted.model_id)] == [
        ModelArtifactStatus.CANDIDATE,
        ModelArtifactStatus.EVALUATED,
        ModelArtifactStatus.SEALED,
        ModelArtifactStatus.PROMOTED,
        ModelArtifactStatus.RETIRED,
    ]
    snapshots = registry.history(promoted.model_id)
    assert all(item.verify_record() for item in snapshots)
    assert len({item.record_id for item in snapshots}) == len(snapshots)
    assert len(registry.snapshots()) == len(snapshots)


def test_hardware_tolerance_and_dependency_lock_are_explicit(tmp_path):
    policy = {
        "hardware_tolerances": {"cpu": {"tolerance_abs": 1e-6, "tolerance_rel": 1e-5}},
    }
    assert hardware_compatible(
        {"device": "cpu"}, {"device": "cpu", "max_abs_error": 1e-6, "max_rel_error": 1e-5}, policy
    )
    assert not hardware_compatible({"device": "cpu"}, {"device": "cpu", "max_abs_error": 1e-3}, policy)
    assert not hardware_compatible({"device": "cpu"}, {"device": "cuda"}, policy)
    assert not hardware_compatible({"device": "cpu"}, {"device": "cpu", "max_abs_error": 0.0}, policy)
    assert not hardware_compatible(
        {"device": "cpu"}, {"device": "cpu", "max_abs_error": float("nan"), "max_rel_error": 0.0}, policy
    )
    assert not hardware_compatible(
        {"device": "cpu"}, {"device": "cpu", "max_abs_error": 0.0, "max_rel_error": float("inf")}, policy
    )
    cross_policy = {"hardware_cross_tolerances": {"cpu": {"cuda": {"tolerance_abs": 1e-4, "tolerance_rel": 1e-3}}}}
    assert hardware_compatible(
        {"device": "cpu"}, {"device": "cuda", "max_abs_error": 1e-5, "max_rel_error": 1e-4}, cross_policy
    )
    assert not hardware_compatible({"device": "cpu"}, {"device": "cuda", "max_abs_error": 1e-2}, cross_policy)
    lock = tmp_path / "lock"
    lock.write_bytes(b"offline-lock")
    expected = hashlib.sha256(lock.read_bytes()).hexdigest()
    assert verify_dependency_lock(lock, expected)
    lock.write_bytes(b"changed")
    assert not verify_dependency_lock(lock, expected)


def test_artifact_profiles_are_deeply_immutable_and_reject_non_finite_values(tmp_path):
    artifact, _checkpoint, _lock = _artifact(tmp_path, report_hash=H)
    with pytest.raises(TypeError):
        artifact.hardware_profile["device"] = "cuda"
    with pytest.raises(TypeError):
        artifact.determinism_profile["nested"] = {"value": 1}

    with pytest.raises(ValueError, match="infinity"):
        ModelArtifact.from_checkpoint(
            tmp_path / "model-1.weights",
            model_id="non-finite",
            feature_schema_hash=H,
            label_schema_hash="b" * 64,
            training_snapshot_id="snapshot-1",
            split_manifest_hash="c" * 64,
            reliability_report_hash=H,
            dependency_lock_hash="d" * 64,
            hardware_profile={"device": "cpu"},
            determinism_profile={
                "seed": 42,
                "deterministic_algorithms": True,
                "tolerance_abs": float("nan"),
                "tolerance_rel": 1e-5,
            },
            created_at=datetime(2026, 9, 4, tzinfo=UTC),
        )


def test_checkpoint_manifest_hashes_are_recomputed_from_schema_files(tmp_path):
    directory = tmp_path / "checkpoint-manifest"
    directory.mkdir()
    (directory / "model.pt").write_bytes(b"weights")
    feature_path = directory / "feature_schema.json"
    label_path = directory / "label_schema.json"
    feature_path.write_text('{"features":["close"]}', encoding="utf-8")
    label_path.write_text('{"labels":["return"]}', encoding="utf-8")
    feature_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    label_hash = hashlib.sha256(label_path.read_bytes()).hexdigest()
    split_manifest = {"train": ["snapshot-1"]}
    (directory / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "source_market_snapshot_id": "snapshot-1",
                "feature_schema_hash": feature_hash,
                "label_schema_hash": label_hash,
                "split_manifest": split_manifest,
            }
        ),
        encoding="utf-8",
    )
    (directory / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "manifest-model",
                "market_snapshot_id": "snapshot-1",
                "feature_schema_hash": "0" * 64,
                "label_schema_hash": label_hash,
                "created_at": "2026-09-04T00:00:00Z",
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(RegistryError, match="feature schema"):
        ModelRegistry().register_from_checkpoint_manifest(
            directory,
            reliability_report=report,
            dependency_lock_hash="d" * 64,
        )
