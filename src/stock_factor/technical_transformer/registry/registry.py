"""Append-only registry and promotion gates for Technical Transformer models."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from stock_factor.config.schema import load_config

from .model_artifact import ModelArtifact, ModelArtifactStatus, canonical_json, sha256_file


class RegistryError(ValueError):
    """Raised when a model registry invariant is violated."""


_ALLOWED = {
    ModelArtifactStatus.CANDIDATE: {ModelArtifactStatus.EVALUATED},
    ModelArtifactStatus.EVALUATED: {ModelArtifactStatus.SEALED},
    ModelArtifactStatus.SEALED: {ModelArtifactStatus.PROMOTED},
    ModelArtifactStatus.PROMOTED: {ModelArtifactStatus.RETIRED},
    ModelArtifactStatus.RETIRED: set(),
}


def hash_report(report: Mapping[str, Any] | str | Path) -> str:
    if isinstance(report, (str, Path)):
        return sha256_file(report)
    return hashlib.sha256(canonical_json(dict(report))).hexdigest()


def verify_dependency_lock(path: str | Path, expected_hash: str) -> bool:
    return sha256_file(path) == expected_hash


def hardware_compatible(
    recorded: Mapping[str, Any], observed: Mapping[str, Any], policy: Mapping[str, Any] | None = None
) -> bool:
    recorded_device = str(recorded.get("device", "")).lower()
    observed_device = str(observed.get("device", "")).lower()
    if any(key not in observed for key in ("max_abs_error", "max_rel_error")):
        return False
    try:
        observed_abs = float(observed["max_abs_error"])
        observed_rel = float(observed["max_rel_error"])
    except (TypeError, ValueError):
        return False
    if not math.isfinite(observed_abs) or not math.isfinite(observed_rel) or observed_abs < 0 or observed_rel < 0:
        return False
    configured = (policy or {}).get("hardware_tolerances", {})
    cross = (policy or {}).get("hardware_cross_tolerances", {})
    if recorded_device == observed_device:
        limits = configured.get(recorded_device, {}) if isinstance(configured, Mapping) else {}
    else:
        limits = cross.get(recorded_device, {}).get(observed_device, {}) if isinstance(cross, Mapping) else {}
        if not limits:
            return False
    abs_limit = float(limits.get("tolerance_abs", recorded.get("tolerance_abs", 0.0)))
    rel_limit = float(limits.get("tolerance_rel", recorded.get("tolerance_rel", 0.0)))
    if not math.isfinite(abs_limit) or not math.isfinite(rel_limit) or abs_limit < 0 or rel_limit < 0:
        return False
    return observed_abs <= abs_limit and observed_rel <= rel_limit


def _evidence_passes(report: Mapping[str, Any]) -> tuple[bool, str]:
    if report.get("report_version") != "technical-reliability-report.v2":
        return False, "unsupported reliability report version"
    if str(report.get("mode", "")).upper() != "PRODUCTION":
        return False, "reliability report is not PRODUCTION"
    if (report.get("reliability_gate") or {}).get("status") != "PASS":
        return False, "reliability gate is not PASS"
    required = report.get("required_evidence") or {}
    # Keep this inventory in lock-step with build_reliability_report's
    # production contract.  ``embedding_probe`` is the report key while its
    # materialized result lives under ``embedding``.
    expected_evidence = {
        "gold_set",
        "baseline",
        "causality",
        "embedding_probe",
        "invariance",
        "double_oos",
    }
    if set(required) != expected_evidence or any(value is not True for value in required.values()):
        return False, "required reliability evidence is incomplete"
    data_integrity = report.get("data_integrity") or {}
    causality = data_integrity.get("causality") or report.get("causality") or {}
    leakage = data_integrity.get("leakage") or report.get("leakage_audit") or {}
    split = data_integrity.get("split") or {}
    if causality.get("status") != "EVALUATED" or causality.get("total_violations") != 0:
        return False, "causality evidence is incomplete or failed"
    if leakage.get("passed") is not True or leakage.get("violations") != []:
        return False, "leakage evidence is incomplete or failed"
    if split.get("overlap") not in (0, 0.0):
        return False, "split overlap evidence failed"
    for name in ("gold_set", "baseline", "embedding", "invariance"):
        section = report.get(name) or {}
        if section.get("status") != "EVALUATED" or section.get("passed") is not True:
            return False, f"{name} evidence is incomplete or failed"
    double_oos = (report.get("splits") or {}).get("double_oos") or {}
    if int(double_oos.get("sample_count", 0)) <= 0:
        return False, "double_oos evidence is missing"
    return True, ""


def _report_identity(report: Mapping[str, Any], key: str) -> Any:
    checkpoint = report.get("checkpoint") or {}
    dataset = report.get("dataset") or {}
    aliases = {
        "checkpoint_sha256": ("checkpoint_sha256", "checkpoint_hash"),
        "feature_schema_hash": ("feature_schema_hash",),
        "label_schema_hash": ("label_schema_hash",),
        "training_snapshot_id": ("training_snapshot_id", "market_snapshot_id", "snapshot_id"),
        "split_manifest_hash": ("split_manifest_hash",),
    }
    for container in (checkpoint, dataset, report):
        for candidate in aliases.get(key, (key,)):
            if candidate in container:
                return container[candidate]
    return None


def _report_matches_artifact(report: Mapping[str, Any], artifact: ModelArtifact) -> bool:
    expected = {
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "feature_schema_hash": artifact.feature_schema_hash,
        "label_schema_hash": artifact.label_schema_hash,
        "training_snapshot_id": artifact.training_snapshot_id,
        "split_manifest_hash": artifact.split_manifest_hash,
    }
    return all(_report_identity(report, key) == value for key, value in expected.items())


@dataclass(frozen=True)
class RegistryEvent:
    sequence: int
    model_id: str
    from_status: str | None
    to_status: str
    at: datetime
    reason: str


class ModelRegistry:
    """Thread-safe append-only registry with an atomic active-model pointer."""

    def __init__(self, *, promotion_policy: Mapping[str, Any] | str | Path | None = None) -> None:
        if promotion_policy is None:
            policy: Mapping[str, Any] = {}
        elif isinstance(promotion_policy, Mapping):
            policy = dict(promotion_policy)
        else:
            policy = load_config(promotion_policy).payload
        self._policy = dict(policy)
        self._records: dict[str, ModelArtifact] = {}
        self._snapshots: dict[str, ModelArtifact] = {}
        self._checkpoint_paths: dict[str, Path] = {}
        self._bound_manifests: dict[str, tuple[Path, Path, Path]] = {}
        self._history: dict[str, list[ModelArtifact]] = {}
        self._events: list[RegistryEvent] = []
        self._active_model_id: str | None = None
        self._lock = threading.RLock()

    def register(self, artifact: ModelArtifact, *, checkpoint: str | Path | None = None) -> ModelArtifact:
        with self._lock:
            if checkpoint is not None and not artifact.verify_checkpoint(checkpoint):
                raise RegistryError("checkpoint hash mismatch during registration")
            existing = self._records.get(artifact.artifact_id)
            if existing is not None:
                if existing != artifact:
                    raise RegistryError("artifact id collision with different model identity")
                if checkpoint is not None:
                    self._checkpoint_paths[artifact.artifact_id] = Path(checkpoint)
                return existing
            if artifact.model_id in {item.model_id for item in self._records.values()}:
                raise RegistryError("model_id already exists; registry records are append-only")
            self._records[artifact.artifact_id] = artifact
            self._snapshots[artifact.record_id] = artifact
            self._history[artifact.artifact_id] = [artifact]
            if checkpoint is not None:
                self._checkpoint_paths[artifact.artifact_id] = Path(checkpoint)
            self._append_event(artifact.model_id, None, artifact.status, "REGISTERED")
            return artifact

    def register_from_checkpoint_manifest(
        self,
        checkpoint_dir: str | Path,
        *,
        reliability_report: str | Path,
        dependency_lock_hash: str,
        model_id: str | None = None,
    ) -> ModelArtifact:
        """Build a candidate from the manifests emitted by the training pipeline."""
        directory = Path(checkpoint_dir)
        checkpoint_manifest = json.loads((directory / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        dataset_manifest = json.loads((directory / "dataset_manifest.json").read_text(encoding="utf-8"))
        checkpoint = next(
            (directory / name for name in ("model.safetensors", "model.pt") if (directory / name).is_file()), None
        )
        if checkpoint is None:
            raise RegistryError("checkpoint manifest directory has no model.safetensors or model.pt")
        report_hash = hash_report(reliability_report)
        split_manifest = dataset_manifest.get("split_manifest") or {
            key: dataset_manifest.get(key)
            for key in ("splits", "split_overlap", "split_config")
            if key in dataset_manifest
        }
        computed_split_hash = hashlib.sha256(canonical_json(split_manifest)).hexdigest()
        declared_split_hash = dataset_manifest.get("split_manifest_hash")
        if declared_split_hash is not None and declared_split_hash != computed_split_hash:
            raise RegistryError("dataset split manifest hash does not match its contents")
        split_hash = computed_split_hash
        feature_path = directory / "feature_schema.json"
        label_path = directory / "label_schema.json"
        feature_hash = sha256_file(feature_path)
        label_hash = sha256_file(label_path)
        for name, declared, actual in (
            ("feature schema", dataset_manifest.get("feature_schema_hash"), feature_hash),
            ("label schema", dataset_manifest.get("label_schema_hash"), label_hash),
            ("checkpoint feature schema", checkpoint_manifest.get("feature_schema_hash"), feature_hash),
            ("checkpoint label schema", checkpoint_manifest.get("label_schema_hash"), label_hash),
        ):
            if declared is not None and declared != actual:
                raise RegistryError(f"dataset {name} hash does not match its contents")
        checkpoint_snapshot = checkpoint_manifest.get("market_snapshot_id") or checkpoint_manifest.get(
            "training_snapshot_id"
        )
        dataset_snapshot = dataset_manifest.get("source_market_snapshot_id") or dataset_manifest.get(
            "training_snapshot_id"
        )
        if checkpoint_snapshot and dataset_snapshot and checkpoint_snapshot != dataset_snapshot:
            raise RegistryError("checkpoint and dataset snapshot identities differ")
        artifact = ModelArtifact.from_checkpoint(
            checkpoint,
            model_id=model_id or str(checkpoint_manifest.get("checkpoint_id") or directory.name),
            feature_schema_hash=feature_hash,
            label_schema_hash=label_hash,
            training_snapshot_id=str(
                checkpoint_manifest.get("market_snapshot_id") or dataset_manifest.get("source_market_snapshot_id") or ""
            ),
            split_manifest_hash=split_hash,
            reliability_report_hash=report_hash,
            dependency_lock_hash=dependency_lock_hash,
            hardware_profile={"device": str(checkpoint_manifest.get("device", "cpu")).lower()},
            determinism_profile={
                "seed": int(checkpoint_manifest.get("seed", 42)),
                "deterministic_algorithms": True,
                "tolerance_abs": float(checkpoint_manifest.get("tolerance_abs", 1e-6)),
                "tolerance_rel": float(checkpoint_manifest.get("tolerance_rel", 1e-5)),
            },
            created_at=datetime.fromisoformat(
                str(checkpoint_manifest.get("created_at", datetime.now(UTC).isoformat())).replace("Z", "+00:00")
            ),
        )
        registered = self.register(artifact, checkpoint=checkpoint)
        self._bound_manifests[registered.artifact_id] = (
            feature_path,
            label_path,
            directory / "checkpoint_manifest.json",
        )
        return registered

    def get(self, model_id: str) -> ModelArtifact:
        with self._lock:
            for artifact in self._records.values():
                if artifact.model_id == model_id or artifact.artifact_id == model_id:
                    return artifact
        raise RegistryError(f"model is not registered: {model_id}")

    def list(self) -> tuple[ModelArtifact, ...]:
        with self._lock:
            return tuple(self._records.values())

    def events(self) -> tuple[RegistryEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def history(self, model_id: str) -> tuple[ModelArtifact, ...]:
        with self._lock:
            current = self.get(model_id)
            return tuple(self._history[current.artifact_id])

    def snapshots(self) -> tuple[ModelArtifact, ...]:
        """Return every immutable record snapshot in insertion order."""
        with self._lock:
            return tuple(self._snapshots.values())

    def _append_event(
        self, model_id: str, old: ModelArtifactStatus | None, new: ModelArtifactStatus, reason: str
    ) -> None:
        self._events.append(
            RegistryEvent(
                len(self._events) + 1, model_id, old.value if old else None, new.value, datetime.now(UTC), reason
            )
        )

    def _replace(self, current: ModelArtifact, updated: ModelArtifact, reason: str) -> ModelArtifact:
        if updated.record_id in self._snapshots:
            raise RegistryError("record id collision; registry snapshots are append-only")
        self._snapshots[updated.record_id] = updated
        self._records[updated.artifact_id] = updated
        self._history[current.artifact_id].append(updated)
        self._append_event(current.model_id, current.status, updated.status, reason)
        return updated

    def evaluate(self, model_id: str, report: Mapping[str, Any] | str | Path) -> ModelArtifact:
        with self._lock:
            current = self.get(model_id)
            if current.status != ModelArtifactStatus.CANDIDATE:
                raise RegistryError(f"evaluation requires CANDIDATE, got {current.status.value}")
            if hash_report(report) != current.reliability_report_hash:
                raise RegistryError("reliability report hash does not match model artifact")
            payload = (
                json.loads(Path(report).read_text(encoding="utf-8")) if isinstance(report, (str, Path)) else report
            )
            if not _report_matches_artifact(payload, current):
                raise RegistryError("reliability report identity does not match model artifact")
            passed, reason = _evidence_passes(payload)
            seal = {"status": "PASS" if passed else "FAIL", "mode": str(payload.get("mode", "")), "reason": reason}
            return self._replace(
                current, current.with_status(ModelArtifactStatus.EVALUATED, reliability_seal=seal), "EVALUATED"
            )

    def seal(self, model_id: str) -> ModelArtifact:
        with self._lock:
            current = self.get(model_id)
            if current.status != ModelArtifactStatus.EVALUATED:
                raise RegistryError(f"sealing requires EVALUATED, got {current.status.value}")
            if current.reliability_seal.get("status") != "PASS":
                raise RegistryError("only a full PRODUCTION PASS report may seal")
            return self._replace(current, current.with_status(ModelArtifactStatus.SEALED), "SEALED")

    def promote(
        self,
        model_id: str,
        *,
        checkpoint: str | Path,
        dependency_lock: str | Path,
        feature_schema_hash: str,
        label_schema_hash: str,
        training_snapshot_id: str,
        hardware_profile: Mapping[str, Any],
    ) -> ModelArtifact:
        with self._lock:
            current = self.get(model_id)
            if current.status != ModelArtifactStatus.SEALED:
                raise RegistryError(f"promotion requires SEALED, got {current.status.value}")
            if not current.verify_checkpoint(checkpoint):
                raise RegistryError("checkpoint hash mismatch; promotion denied")
            if not verify_dependency_lock(dependency_lock, current.dependency_lock_hash):
                raise RegistryError("dependency lock hash mismatch; promotion denied")
            if (feature_schema_hash, label_schema_hash, training_snapshot_id) != (
                current.feature_schema_hash,
                current.label_schema_hash,
                current.training_snapshot_id,
            ):
                raise RegistryError("model schema or training snapshot identity mismatch")
            bound = self._bound_manifests.get(current.artifact_id)
            if bound is not None:
                feature_path, label_path, checkpoint_manifest_path = bound
                if sha256_file(feature_path) != current.feature_schema_hash:
                    raise RegistryError("feature schema file was modified")
                if sha256_file(label_path) != current.label_schema_hash:
                    raise RegistryError("label schema file was modified")
                checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
                snapshot_id = checkpoint_manifest.get("market_snapshot_id") or checkpoint_manifest.get(
                    "training_snapshot_id"
                )
                if snapshot_id != current.training_snapshot_id:
                    raise RegistryError("training snapshot manifest was modified")
            if not hardware_compatible(current.hardware_profile, hardware_profile, self._policy):
                raise RegistryError("hardware profile is outside the configured tolerance")
            return self._replace(current, current.with_status(ModelArtifactStatus.PROMOTED), "PROMOTED")

    def retire(self, model_id: str) -> ModelArtifact:
        with self._lock:
            current = self.get(model_id)
            if current.status != ModelArtifactStatus.PROMOTED:
                raise RegistryError(f"retirement requires PROMOTED, got {current.status.value}")
            if self._active_model_id == current.model_id:
                self._active_model_id = None
            return self._replace(current, current.with_status(ModelArtifactStatus.RETIRED), "RETIRED")

    def activate(self, model_id: str) -> ModelArtifact:
        with self._lock:
            artifact = self.get(model_id)
            if artifact.status != ModelArtifactStatus.PROMOTED:
                raise RegistryError("only a PROMOTED artifact may become active")
            self._active_model_id = artifact.model_id
            self._append_event(artifact.model_id, artifact.status, artifact.status, "ACTIVE_SWITCH")
            return artifact

    def rollback(self, model_id: str) -> ModelArtifact:
        return self.activate(model_id)

    def active(self) -> ModelArtifact:
        if self._active_model_id is None:
            raise RegistryError("no active promoted model")
        return self.get(self._active_model_id)

    def require_promoted(self, model_id: str) -> ModelArtifact:
        artifact = self.get(model_id)
        if artifact.status != ModelArtifactStatus.PROMOTED or artifact.reliability_seal.get("status") != "PASS":
            raise RegistryError("formal inference requires a PROMOTED model with a PASS reliability seal")
        return artifact

    def checkpoint_path(self, model_id: str) -> Path:
        artifact = self.require_promoted(model_id)
        try:
            path = self._checkpoint_paths[artifact.artifact_id]
        except KeyError as exc:
            raise RegistryError("registered model has no checkpoint path") from exc
        if not artifact.verify_checkpoint(path):
            raise RegistryError("registered checkpoint was modified")
        return path
