"""Immutable, content-addressed formal research evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any


def _canonical(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("artifact datetimes must be timezone-aware")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact payload cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported artifact value type: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"artifact requires non-empty {name}")
    return value


CONTRACT_VERSION = "research-artifact.v2"
REQUIRED_CONTRACT_CHECKSUMS = frozenset(
    {"factor.v1", "market-snapshot.v1", "content-factor-signal.v5.1", CONTRACT_VERSION}
)
_HASH_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_CHECKSUM_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _hash_text(value: Any, name: str) -> str:
    value = _required_text(value, name)
    if _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"artifact {name} must be a canonical sha256 hash")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ResearchArtifactV2:
    experiment_id: str
    research_question: str
    hypothesis: str
    dataset_manifest: dict[str, Any]
    market_ref: dict[str, Any]
    content_ref: dict[str, Any]
    candidate_set_hash: str
    statistical_experiment: dict[str, Any]
    final_oos_evidence: dict[str, Any]
    tradability_assessment: dict[str, Any]
    promotion_decision: dict[str, Any]
    promotion_policy_version: str
    producer_commit: str
    dependency_lock_hash: str
    contract_checksums: dict[str, str]
    created_at: datetime
    artifact_status: str = "SEALED"
    factor_set: dict[str, Any] | None = None
    readiness_evidence: dict[str, Any] | None = None
    artifact_id: str = field(init=False, default="")

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "research_question",
            "hypothesis",
            "promotion_policy_version",
            "producer_commit",
        ):
            _required_text(getattr(self, name), name)
        _hash_text(self.candidate_set_hash, "candidate_set_hash")
        _hash_text(self.dependency_lock_hash, "dependency_lock_hash")
        if self.artifact_status != "SEALED":
            raise ValueError("formal ResearchArtifactV2 must be SEALED")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("artifact created_at must be timezone-aware")
        for name in (
            "dataset_manifest",
            "market_ref",
            "content_ref",
            "statistical_experiment",
            "final_oos_evidence",
            "tradability_assessment",
            "promotion_decision",
        ):
            if not isinstance(getattr(self, name), dict) or not getattr(self, name):
                raise ValueError(f"artifact requires non-empty {name}")
        if not self.contract_checksums or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
            for key, value in self.contract_checksums.items()
        ):
            raise ValueError("artifact requires contract checksums")
        if any(_CHECKSUM_PATTERN.fullmatch(value) is None for value in self.contract_checksums.values()):
            raise ValueError("artifact contract checksums must be canonical sha256 hashes")
        if not REQUIRED_CONTRACT_CHECKSUMS.issubset(self.contract_checksums):
            raise ValueError("artifact contract checksums missing required inventory")
        if not any(key in self.dataset_manifest for key in ("universe", "universe_identity", "universe_snapshot_id")):
            raise ValueError("dataset manifest must bind universe identity")
        for name, reference in (("market_ref", self.market_ref), ("content_ref", self.content_ref)):
            if "ref_hash" not in reference:
                raise ValueError(f"artifact {name} must bind ref_hash")
            _hash_text(reference["ref_hash"], f"{name}.ref_hash")
        for key in ("multiple_testing", "dsr", "pbo"):
            if key not in self.statistical_experiment:
                raise ValueError(f"statistical experiment missing {key}")
        if self.final_oos_evidence.get("status") != "SEALED":
            raise ValueError("final OOS evidence must be SEALED")
        if "cohort_artifact_hash" not in self.final_oos_evidence:
            raise ValueError("sealed Final OOS evidence requires cohort_artifact_hash")
        _hash_text(self.final_oos_evidence["cohort_artifact_hash"], "final_oos_evidence.cohort_artifact_hash")
        if not isinstance(self.final_oos_evidence.get("run_id"), str) or not self.final_oos_evidence["run_id"].strip():
            raise ValueError("sealed Final OOS evidence requires run_id")
        if not any(
            key in self.tradability_assessment for key in ("reference", "evidence", "assessment_ref", "artifact_id")
        ):
            raise ValueError("tradability assessment reference/evidence is required")
        if self.readiness_evidence is not None:
            required_readiness = {
                "ready",
                "runtime_profile",
                "checks",
                "blocking_reasons",
                "threshold_version",
                "frozen_at",
                "evidence_hash",
            }
            if not isinstance(self.readiness_evidence, dict) or not required_readiness.issubset(
                self.readiness_evidence
            ):
                raise ValueError("readiness evidence is incomplete")
            if not isinstance(self.readiness_evidence["ready"], bool):
                raise ValueError("readiness evidence ready must be boolean")
            _hash_text(self.readiness_evidence["evidence_hash"], "readiness_evidence.evidence_hash")
            _required_text(self.readiness_evidence["runtime_profile"], "readiness_evidence.runtime_profile")
            _required_text(self.readiness_evidence["threshold_version"], "readiness_evidence.threshold_version")
            _required_text(self.readiness_evidence["frozen_at"], "readiness_evidence.frozen_at")
            if not isinstance(self.readiness_evidence["checks"], dict) or not isinstance(
                self.readiness_evidence["blocking_reasons"], list
            ):
                raise ValueError("readiness evidence checks/blocking_reasons have invalid types")
        for name in (
            "dataset_manifest",
            "market_ref",
            "content_ref",
            "statistical_experiment",
            "final_oos_evidence",
            "tradability_assessment",
            "promotion_decision",
            "contract_checksums",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        if self.factor_set is not None:
            object.__setattr__(self, "factor_set", _freeze(self.factor_set))
        if self.readiness_evidence is not None:
            object.__setattr__(self, "readiness_evidence", _freeze(self.readiness_evidence))
        material = self.to_payload(include_id=False)
        object.__setattr__(self, "artifact_id", hashlib.sha256(canonical_json(material)).hexdigest())

    def to_payload(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "contract_version": CONTRACT_VERSION,
            "experiment_id": self.experiment_id,
            "research_question": self.research_question,
            "hypothesis": self.hypothesis,
            "dataset_manifest": self.dataset_manifest,
            "market_ref": self.market_ref,
            "content_ref": self.content_ref,
            "candidate_set_hash": self.candidate_set_hash,
            "statistical_experiment": self.statistical_experiment,
            "final_oos_evidence": self.final_oos_evidence,
            "tradability_assessment": self.tradability_assessment,
            "promotion_decision": self.promotion_decision,
            "promotion_policy_version": self.promotion_policy_version,
            "producer_commit": self.producer_commit,
            "dependency_lock_hash": self.dependency_lock_hash,
            "contract_checksums": self.contract_checksums,
            "created_at": self.created_at,
            "artifact_status": self.artifact_status,
        }
        if self.factor_set is not None:
            payload["factor_set"] = self.factor_set
        if self.readiness_evidence is not None:
            payload["readiness_evidence"] = self.readiness_evidence
        if include_id:
            payload["artifact_id"] = self.artifact_id
        return _canonical(payload)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ResearchArtifactV2":
        if not isinstance(payload, dict):
            raise TypeError("research artifact payload must be an object")
        data = dict(payload)
        if data.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("unsupported research artifact contract")
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("artifact created_at is not a valid datetime") from exc
        data.pop("artifact_id", None)
        data.pop("contract_version", None)
        data["created_at"] = created_at
        artifact = cls(**data)
        supplied_id = payload.get("artifact_id")
        if supplied_id is not None and supplied_id != artifact.artifact_id:
            raise ValueError("research artifact hash mismatch")
        return artifact

    def verify(self) -> bool:
        return self.artifact_id == hashlib.sha256(canonical_json(self.to_payload(include_id=False))).hexdigest()


__all__ = ["CONTRACT_VERSION", "REQUIRED_CONTRACT_CHECKSUMS", "ResearchArtifactV2", "canonical_json"]
