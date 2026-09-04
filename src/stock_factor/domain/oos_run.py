"""Resumable, fenced Final OOS run identities.

The records in this module are deliberately small immutable/value objects.  A
repository owns the mutable state transitions and is responsible for CAS or
transaction semantics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum


def _hash(material: dict) -> str:
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def canonical_candidate_set_hash(candidate_ids: list[str] | tuple[str, ...]) -> str:
    if any(value is None for value in candidate_ids):
        raise ValueError("candidate set contains a null candidate id")
    ids = [str(value) for value in candidate_ids]
    if not ids or any(not value.strip() for value in ids):
        raise ValueError("candidate set must contain non-empty ids")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate set contains duplicate candidate ids")
    return _hash({"candidate_ids": sorted(ids)})


def canonical_dataset_ref_hash(
    *,
    final_oos_snapshot_id: str,
    final_oos_start: int,
    final_oos_end: int,
    market_ref_hash: str,
    content_ref_hash: str | None = None,
) -> str:
    if not str(final_oos_snapshot_id).strip():
        raise ValueError("Final OOS snapshot identity is required")
    if not str(market_ref_hash).strip():
        raise ValueError("market dataset reference identity is required")
    if final_oos_end <= final_oos_start:
        raise ValueError("Final OOS range must be forward")
    return _hash(
        {
            "final_oos_snapshot_id": final_oos_snapshot_id,
            "final_oos_start": final_oos_start,
            "final_oos_end": final_oos_end,
            "market_ref_hash": market_ref_hash,
            "content_ref_hash": content_ref_hash,
        }
    )


def _now(value: datetime | None = None) -> datetime:
    result = value or datetime.now(UTC)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("OOS timestamps must be timezone-aware")
    return result.astimezone(UTC)


class AuthorizationStatus(StrEnum):
    AUTHORIZED = "AUTHORIZED"
    EVALUATING = "EVALUATING"
    EVALUATING_INTERRUPTED = "EVALUATING_INTERRUPTED"
    CONSUMED = "CONSUMED"
    INVALIDATED = "INVALIDATED"


class OosRunStatus(StrEnum):
    STARTED = "STARTED"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    SEALING = "SEALING"
    SEALED = "SEALED"
    FAILED_TERMINAL = "FAILED_TERMINAL"


class CheckpointStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_TERMINAL = "FAILED_TERMINAL"


@dataclass(frozen=True)
class OosAuthorization:
    authorization_id: str
    experiment_id: str
    candidate_set_hash: str
    dataset_ref_hash: str
    market_snapshot_id: str
    content_ref_hash: str | None = None
    status: AuthorizationStatus = AuthorizationStatus.AUTHORIZED
    active_run_id: str | None = None
    authorized_at: datetime = field(default_factory=lambda: _now())
    consumed_at: datetime | None = None
    invalidated_at: datetime | None = None
    identity_hash: str = field(init=False)
    version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "authorization_id",
            "experiment_id",
            "candidate_set_hash",
            "dataset_ref_hash",
            "market_snapshot_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"OOS authorization requires non-empty {name}")
        material = {
            "experiment_id": self.experiment_id,
            "candidate_set_hash": self.candidate_set_hash,
            "dataset_ref_hash": self.dataset_ref_hash,
            "market_snapshot_id": self.market_snapshot_id,
            "content_ref_hash": self.content_ref_hash,
        }
        object.__setattr__(self, "identity_hash", _hash(material))


@dataclass(frozen=True)
class OosEvaluationRun:
    run_id: str
    authorization_id: str
    owner_id: str
    fencing_token: int = 1
    status: OosRunStatus = OosRunStatus.STARTED
    lease_expires_at: datetime = field(default_factory=lambda: _now() + timedelta(minutes=15))
    evaluator_version: str = "final-oos-v1"
    cohort_artifact_hash: str | None = None
    started_at: datetime = field(default_factory=lambda: _now())
    sealed_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "authorization_id", "owner_id", "evaluator_version"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"OOS run requires non-empty {name}")
        if self.fencing_token < 1:
            raise ValueError("OOS fencing token must be positive")
        _now(self.lease_expires_at)
        _now(self.started_at)
        if self.sealed_at is not None:
            _now(self.sealed_at)


@dataclass(frozen=True)
class OosCandidateCheckpoint:
    run_id: str
    candidate_id: str
    input_hash: str
    status: CheckpointStatus = CheckpointStatus.PENDING
    result: dict | None = None
    result_hash: str | None = None
    error: str | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "candidate_id", "input_hash"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"checkpoint requires non-empty {name}")
        if self.completed_at is not None:
            _now(self.completed_at)

    @classmethod
    def completed(cls, run_id: str, candidate_id: str, input_hash: str, result: dict) -> "OosCandidateCheckpoint":
        return cls(
            run_id=run_id,
            candidate_id=candidate_id,
            input_hash=input_hash,
            status=CheckpointStatus.COMPLETED,
            result=result,
            result_hash=_hash(result),
            completed_at=_now(),
        )


def canonical_cohort_hash(results: list[dict], *, authorization: OosAuthorization, run: OosEvaluationRun) -> str:
    entries = sorted(results, key=lambda value: str(value["candidate_id"]))
    candidate_ids = [str(value["candidate_id"]) for value in entries]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("cohort artifact contains duplicate candidate ids")
    return _hash(
        {
            "authorization_identity": authorization.identity_hash,
            "run_id": run.run_id,
            "evaluator_version": run.evaluator_version,
            "dataset_ref_hash": authorization.dataset_ref_hash,
            "market_snapshot_id": authorization.market_snapshot_id,
            "content_ref_hash": authorization.content_ref_hash,
            "candidates": entries,
        }
    )


__all__ = [
    "AuthorizationStatus",
    "OosRunStatus",
    "CheckpointStatus",
    "OosAuthorization",
    "OosEvaluationRun",
    "OosCandidateCheckpoint",
    "canonical_cohort_hash",
    "canonical_candidate_set_hash",
    "canonical_dataset_ref_hash",
]
