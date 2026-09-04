"""Persistence port for resumable Final OOS runs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import uuid4

from stock_factor.domain.oos_run import (
    AuthorizationStatus,
    OosAuthorization,
    OosCandidateCheckpoint,
    OosEvaluationRun,
    OosRunStatus,
)


class OosRunError(RuntimeError):
    pass


class OosLeaseError(OosRunError):
    pass


class OosIdentityError(OosRunError):
    pass


class OosCheckpointConflict(OosRunError):
    pass


class OosRunRepository(Protocol):
    def save_authorization(self, authorization: OosAuthorization) -> OosAuthorization: ...

    def get_authorization(self, authorization_id: str) -> OosAuthorization | None: ...

    def invalidate_authorization(self, authorization_id: str) -> OosAuthorization: ...

    def start_or_resume(
        self,
        authorization_id: str,
        run_id: str | None,
        owner_id: str,
        lease_seconds: int,
        *,
        evaluator_version: str = "final-oos-v1",
        now: datetime | None = None,
    ) -> OosEvaluationRun: ...

    def get_run(self, run_id: str) -> OosEvaluationRun | None: ...

    def renew(
        self, run_id: str, owner_id: str, fencing_token: int, lease_seconds: int, *, now: datetime | None = None
    ) -> OosEvaluationRun: ...

    def get_checkpoint(self, run_id: str, candidate_id: str) -> OosCandidateCheckpoint | None: ...

    def put_checkpoint(
        self, checkpoint: OosCandidateCheckpoint, owner_id: str, fencing_token: int
    ) -> OosCandidateCheckpoint: ...

    def interrupt(
        self, run_id: str, owner_id: str, fencing_token: int, *, terminal: bool = False
    ) -> OosEvaluationRun: ...

    def seal(
        self, run_id: str, owner_id: str, fencing_token: int, cohort_artifact_hash: str, artifact: dict | None = None
    ) -> OosEvaluationRun: ...


class InMemoryOosRunRepository:
    """Thread-safe deterministic repository used by unit tests and local runs."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.RLock()
        self._authorizations: dict[str, OosAuthorization] = {}
        self._runs: dict[str, OosEvaluationRun] = {}
        self._checkpoints: dict[tuple[str, str], OosCandidateCheckpoint] = {}
        self._artifacts: dict[str, dict] = {}

    def save_authorization(self, authorization: OosAuthorization) -> OosAuthorization:
        with self._lock:
            existing = self._authorizations.get(authorization.authorization_id)
            if existing is not None:
                if existing.identity_hash != authorization.identity_hash:
                    raise OosIdentityError("authorization identity drift")
                if existing.status in {AuthorizationStatus.CONSUMED, AuthorizationStatus.INVALIDATED}:
                    return existing
                return existing
            self._authorizations[authorization.authorization_id] = authorization
            return authorization

    def get_authorization(self, authorization_id: str) -> OosAuthorization | None:
        with self._lock:
            return self._authorizations.get(authorization_id)

    def invalidate_authorization(self, authorization_id: str) -> OosAuthorization:
        with self._lock:
            auth = self._authorizations.get(authorization_id)
            if auth is None:
                raise OosIdentityError("authorization is missing")
            if auth.status != AuthorizationStatus.CONSUMED:
                auth = _replace_auth(auth, status=AuthorizationStatus.INVALIDATED, version=auth.version + 1)
                self._authorizations[authorization_id] = auth
            return auth

    def start_or_resume(
        self, authorization_id, run_id, owner_id, lease_seconds, *, evaluator_version="final-oos-v1", now=None
    ):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        current = _utc(now)
        with self._lock:
            auth = self._authorizations.get(authorization_id)
            if auth is None or auth.status in {AuthorizationStatus.CONSUMED, AuthorizationStatus.INVALIDATED}:
                raise OosIdentityError("authorization is not available for evaluation")
            active = self._runs.get(auth.active_run_id) if auth.active_run_id else None
            if active is not None and run_id is not None and run_id != active.run_id:
                raise OosLeaseError("authorization is already bound to a different run")
            run_id = run_id or (active.run_id if active is not None else f"oosrun-{uuid4().hex}")
            if active is not None and active.evaluator_version != evaluator_version:
                raise OosIdentityError("evaluator version mismatch on resume")
            if active is not None:
                if active.run_id == run_id and active.owner_id == owner_id and active.lease_expires_at > current:
                    run = _replace_run(
                        active, status=OosRunStatus.RUNNING, lease_expires_at=current + timedelta(seconds=lease_seconds)
                    )
                    self._runs[run_id] = run
                    self._authorizations[authorization_id] = _replace_auth(
                        auth, status=AuthorizationStatus.EVALUATING, version=auth.version + 1
                    )
                    return run
                if active.lease_expires_at > current:
                    raise OosLeaseError("authorization has an active lease owned by another worker")
                run = _replace_run(
                    active,
                    owner_id=owner_id,
                    fencing_token=active.fencing_token + 1,
                    status=OosRunStatus.RUNNING,
                    lease_expires_at=current + timedelta(seconds=lease_seconds),
                )
                self._runs[run_id] = run
                self._authorizations[authorization_id] = _replace_auth(
                    auth, active_run_id=run_id, status=AuthorizationStatus.EVALUATING, version=auth.version + 1
                )
                return run
            run = OosEvaluationRun(
                run_id=run_id,
                authorization_id=authorization_id,
                owner_id=owner_id,
                status=OosRunStatus.RUNNING,
                lease_expires_at=current + timedelta(seconds=lease_seconds),
                evaluator_version=evaluator_version,
            )
            self._runs[run_id] = run
            self._authorizations[authorization_id] = _replace_auth(
                auth, active_run_id=run_id, status=AuthorizationStatus.EVALUATING, version=auth.version + 1
            )
            return run

    def get_run(self, run_id):
        with self._lock:
            return self._runs.get(run_id)

    def renew(self, run_id, owner_id, fencing_token, lease_seconds, *, now=None):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        current = _utc(now)
        with self._lock:
            run = self._owned(run_id, owner_id, fencing_token, current)
            result = _replace_run(
                run, lease_expires_at=current + timedelta(seconds=lease_seconds), status=OosRunStatus.RUNNING
            )
            self._runs[run_id] = result
            return result

    def get_checkpoint(self, run_id, candidate_id):
        with self._lock:
            return self._checkpoints.get((run_id, candidate_id))

    def put_checkpoint(self, checkpoint, owner_id, fencing_token):
        with self._lock:
            run = self._owned(checkpoint.run_id, owner_id, fencing_token, _utc())
            if run.status in {OosRunStatus.SEALED, OosRunStatus.FAILED_TERMINAL}:
                raise OosRunError("run is terminal")
            key = (checkpoint.run_id, checkpoint.candidate_id)
            existing = self._checkpoints.get(key)
            if existing is not None:
                if existing.input_hash != checkpoint.input_hash or existing.result_hash != checkpoint.result_hash:
                    raise OosCheckpointConflict("checkpoint input/result hash conflict")
                return existing
            self._checkpoints[key] = checkpoint
            return checkpoint

    def interrupt(self, run_id, owner_id, fencing_token, *, terminal=False):
        with self._lock:
            run = self._owned(run_id, owner_id, fencing_token, _utc(), allow_expired=True)
            result = _replace_run(run, status=OosRunStatus.FAILED_TERMINAL if terminal else OosRunStatus.INTERRUPTED)
            self._runs[run_id] = result
            auth = self._authorizations[run.authorization_id]
            self._authorizations[run.authorization_id] = _replace_auth(
                auth,
                status=AuthorizationStatus.INVALIDATED if terminal else AuthorizationStatus.EVALUATING_INTERRUPTED,
                version=auth.version + 1,
            )
            return result

    def seal(self, run_id, owner_id, fencing_token, cohort_artifact_hash, artifact=None):
        with self._lock:
            run = self._owned(run_id, owner_id, fencing_token, _utc())
            if not cohort_artifact_hash:
                raise OosRunError("cohort artifact hash is required")
            result = _replace_run(
                run, status=OosRunStatus.SEALED, cohort_artifact_hash=cohort_artifact_hash, sealed_at=_utc()
            )
            self._runs[run_id] = result
            self._artifacts[cohort_artifact_hash] = artifact or {}
            auth = self._authorizations[run.authorization_id]
            self._authorizations[run.authorization_id] = _replace_auth(
                auth, status=AuthorizationStatus.CONSUMED, consumed_at=_utc(), version=auth.version + 1
            )
            return result

    def get_artifact(self, cohort_artifact_hash: str) -> dict | None:
        with self._lock:
            return self._artifacts.get(cohort_artifact_hash)

    def _owned(self, run_id, owner_id, fencing_token, now, *, allow_expired=False):
        run = self._runs.get(run_id)
        if run is None or run.owner_id != owner_id or run.fencing_token != fencing_token:
            raise OosLeaseError("owner or fencing token is stale")
        auth = self._authorizations.get(run.authorization_id)
        if auth is None or auth.active_run_id != run_id:
            raise OosLeaseError("run is no longer active for its authorization")
        if not allow_expired and run.lease_expires_at <= now:
            raise OosLeaseError("OOS lease expired")
        return run


def _utc(value=None):
    result = value or datetime.now(UTC)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("OOS timestamps must be timezone-aware")
    return result.astimezone(UTC)


def _replace_run(run, **changes):
    from dataclasses import replace

    return replace(run, **changes)


def _replace_auth(auth, **changes):
    from dataclasses import replace

    return replace(auth, **changes)


__all__ = [
    "OosRunRepository",
    "InMemoryOosRunRepository",
    "OosRunError",
    "OosLeaseError",
    "OosIdentityError",
    "OosCheckpointConflict",
]
