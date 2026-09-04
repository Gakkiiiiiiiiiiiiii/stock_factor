"""Content-addressed identity for a Technical Transformer model artifact."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any


class ModelArtifactStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    EVALUATED = "EVALUATED"
    SEALED = "SEALED"
    PROMOTED = "PROMOTED"
    RETIRED = "RETIRED"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("model artifact timestamps must be timezone-aware")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("model artifact cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported model artifact value: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("model artifact cannot contain NaN or infinity")
    return value


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    checkpoint_sha256: str
    feature_schema_hash: str
    label_schema_hash: str
    training_snapshot_id: str
    split_manifest_hash: str
    reliability_report_hash: str
    dependency_lock_hash: str
    hardware_profile: dict[str, Any]
    determinism_profile: dict[str, Any]
    created_at: datetime
    model_version: str = "technical-transformer.v1-reliability-v2"
    status: ModelArtifactStatus = ModelArtifactStatus.CANDIDATE
    reliability_seal: dict[str, Any] = field(default_factory=lambda: {"status": "NOT_EVALUATED"})
    artifact_id: str = field(init=False, default="")
    record_id: str = field(init=False, default="")

    def __post_init__(self) -> None:
        for name in ("model_id", "training_snapshot_id", "model_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"model artifact requires non-empty {name}")
        for name in (
            "checkpoint_sha256",
            "feature_schema_hash",
            "label_schema_hash",
            "split_manifest_hash",
            "reliability_report_hash",
            "dependency_lock_hash",
        ):
            _hash(getattr(self, name), name)
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("model artifact created_at must be timezone-aware")
        if not isinstance(self.hardware_profile, Mapping) or str(
            self.hardware_profile.get("device", "")
        ).lower() not in {
            "cpu",
            "cuda",
        }:
            raise ValueError("hardware_profile.device must be cpu or cuda")
        required_determinism = {"seed", "deterministic_algorithms", "tolerance_abs", "tolerance_rel"}
        if not isinstance(self.determinism_profile, Mapping) or not required_determinism <= set(
            self.determinism_profile
        ):
            raise ValueError("determinism_profile is incomplete")
        if float(self.determinism_profile["tolerance_abs"]) < 0 or float(self.determinism_profile["tolerance_rel"]) < 0:
            raise ValueError("determinism tolerances must be non-negative")
        if not isinstance(self.reliability_seal, Mapping) or self.reliability_seal.get("status") not in {
            "NOT_EVALUATED",
            "PASS",
            "FAIL",
        }:
            raise ValueError("invalid reliability seal status")
        object.__setattr__(self, "hardware_profile", _freeze(self.hardware_profile))
        object.__setattr__(self, "determinism_profile", _freeze(self.determinism_profile))
        object.__setattr__(self, "reliability_seal", _freeze(self.reliability_seal))
        material = self.identity_payload()
        object.__setattr__(self, "artifact_id", "sha256:" + hashlib.sha256(canonical_json(material)).hexdigest())
        object.__setattr__(
            self, "record_id", "sha256:" + hashlib.sha256(canonical_json(self._record_material())).hexdigest()
        )

    @property
    def model_artifact_id(self) -> str:
        """Compatibility alias used by the public model-artifact.v1 schema."""

        return self.artifact_id

    @property
    def model_hash(self) -> str:
        return "sha256:" + self.checkpoint_sha256

    def identity_payload(self) -> dict[str, Any]:
        return _canonical(
            {
                "model_id": self.model_id,
                "checkpoint_sha256": self.checkpoint_sha256,
                "feature_schema_hash": self.feature_schema_hash,
                "label_schema_hash": self.label_schema_hash,
                "training_snapshot_id": self.training_snapshot_id,
                "split_manifest_hash": self.split_manifest_hash,
                "reliability_report_hash": self.reliability_report_hash,
                "dependency_lock_hash": self.dependency_lock_hash,
                "hardware_profile": self.hardware_profile,
                "determinism_profile": self.determinism_profile,
                "created_at": self.created_at,
                "model_version": self.model_version,
            }
        )

    def to_payload(self) -> dict[str, Any]:
        return _canonical({**self._record_material(), "record_id": self.record_id})

    def _record_material(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "contract_version": "model-artifact.v1",
            "artifact_id": self.artifact_id,
            "model_artifact_id": self.model_artifact_id,
            "model_hash": self.model_hash,
            "status": self.status.value,
            "reliability_seal": self.reliability_seal,
            "reliability_gate": {"passed": self.reliability_seal.get("status") == "PASS"},
            "producer": "stock_factor",
        }

    def verify_record(self) -> bool:
        return self.record_id == "sha256:" + hashlib.sha256(canonical_json(self._record_material())).hexdigest()

    def with_status(
        self, status: ModelArtifactStatus, *, reliability_seal: dict[str, Any] | None = None
    ) -> "ModelArtifact":
        return replace(self, status=status, reliability_seal=reliability_seal or self.reliability_seal)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        model_id: str,
        feature_schema_hash: str,
        label_schema_hash: str,
        training_snapshot_id: str,
        split_manifest_hash: str,
        reliability_report_hash: str,
        dependency_lock_hash: str,
        hardware_profile: dict[str, Any],
        determinism_profile: dict[str, Any],
        created_at: datetime,
        model_version: str = "technical-transformer.v1-reliability-v2",
    ) -> "ModelArtifact":
        path = Path(checkpoint)
        if not path.is_file():
            raise ValueError(f"checkpoint file not found: {path}")
        return cls(
            model_id=model_id,
            checkpoint_sha256=sha256_file(path),
            feature_schema_hash=feature_schema_hash,
            label_schema_hash=label_schema_hash,
            training_snapshot_id=training_snapshot_id,
            split_manifest_hash=split_manifest_hash,
            reliability_report_hash=reliability_report_hash,
            dependency_lock_hash=dependency_lock_hash,
            hardware_profile=dict(hardware_profile),
            determinism_profile=dict(determinism_profile),
            created_at=created_at,
            model_version=model_version,
        )

    def verify_checkpoint(self, checkpoint: str | Path) -> bool:
        return sha256_file(checkpoint) == self.checkpoint_sha256
