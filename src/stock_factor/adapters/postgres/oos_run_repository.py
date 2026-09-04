"""Transactional SQLAlchemy repository for resumable OOS runs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select

from stock_factor.adapters.postgres.models import (
    OosAuthorizationRow,
    OosCandidateCheckpointRow,
    OosCohortArtifactRow,
    OosEvaluationRunRow,
)
from stock_factor.domain.oos_run import (
    AuthorizationStatus,
    CheckpointStatus,
    OosAuthorization,
    OosCandidateCheckpoint,
    OosEvaluationRun,
    OosRunStatus,
)
from stock_factor.ports.oos_run_repository import (
    OosCheckpointConflict,
    OosIdentityError,
    OosLeaseError,
    OosRunRepository,
)


class PostgresOosRunRepository(OosRunRepository):
    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def save_authorization(self, authorization):
        with self._sessions.begin() as session:
            row = session.get(OosAuthorizationRow, authorization.authorization_id)
            if row is not None:
                if _auth_identity(row) != authorization.identity_hash:
                    raise OosIdentityError("authorization identity drift")
                return _auth(row)
            session.add(
                OosAuthorizationRow(
                    authorization_id=authorization.authorization_id,
                    experiment_id=authorization.experiment_id,
                    final_oos_snapshot_id=authorization.market_snapshot_id,
                    candidate_set_hash=authorization.candidate_set_hash,
                    dataset_ref_hash=authorization.dataset_ref_hash,
                    market_snapshot_id=authorization.market_snapshot_id,
                    content_ref_hash=authorization.content_ref_hash,
                    status=authorization.status.value,
                )
            )
            session.flush()
            return _auth(session.get(OosAuthorizationRow, authorization.authorization_id))

    def get_authorization(self, authorization_id):
        with self._sessions() as session:
            row = session.get(OosAuthorizationRow, authorization_id)
            return _auth(row) if row else None

    def invalidate_authorization(self, authorization_id):
        with self._sessions.begin() as session:
            row = session.get(OosAuthorizationRow, authorization_id, with_for_update=True)
            if row is None:
                raise OosIdentityError("authorization is missing")
            if row.status != AuthorizationStatus.CONSUMED.value:
                row.status = AuthorizationStatus.INVALIDATED.value
                row.version += 1
            session.flush()
            return _auth(row)

    def start_or_resume(
        self, authorization_id, run_id, owner_id, lease_seconds, *, evaluator_version="final-oos-v1", now=None
    ):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        current = _utc(now)
        with self._sessions.begin() as session:
            auth = session.get(OosAuthorizationRow, authorization_id, with_for_update=True)
            if auth is None or auth.status in {
                AuthorizationStatus.CONSUMED.value,
                AuthorizationStatus.INVALIDATED.value,
            }:
                raise OosIdentityError("authorization is not available for evaluation")
            active = (
                session.get(OosEvaluationRunRow, auth.active_run_id, with_for_update=True)
                if auth.active_run_id
                else None
            )
            if active is not None and run_id is not None and run_id != active.run_id:
                raise OosLeaseError("authorization is already bound to a different run")
            run_id = run_id or (active.run_id if active is not None else f"oosrun-{uuid4().hex}")
            if active is not None and active.evaluator_version != evaluator_version:
                raise OosIdentityError("evaluator version mismatch on resume")
            if (
                active is not None
                and _utc(active.lease_expires_at) > current
                and not (active.run_id == run_id and active.owner_id == owner_id)
            ):
                raise OosLeaseError("authorization has an active lease owned by another worker")
            if (
                active is not None
                and active.run_id == run_id
                and active.owner_id == owner_id
                and _utc(active.lease_expires_at) > current
            ):
                active.status = OosRunStatus.RUNNING.value
                active.lease_expires_at = current + timedelta(seconds=lease_seconds)
            elif active is not None and active.run_id == run_id:
                active.owner_id = owner_id
                active.fencing_token += 1
                active.status = OosRunStatus.RUNNING.value
                active.lease_expires_at = current + timedelta(seconds=lease_seconds)
            else:
                token = active.fencing_token + 1 if active is not None else 1
                active = OosEvaluationRunRow(
                    run_id=run_id,
                    authorization_id=authorization_id,
                    owner_id=owner_id,
                    fencing_token=token,
                    status=OosRunStatus.RUNNING.value,
                    lease_expires_at=current + timedelta(seconds=lease_seconds),
                    evaluator_version=evaluator_version,
                )
                session.add(active)
                auth.active_run_id = run_id
            auth.status = AuthorizationStatus.EVALUATING.value
            auth.version += 1
            session.flush()
            return _run(active)

    def get_run(self, run_id):
        with self._sessions() as session:
            row = session.get(OosEvaluationRunRow, run_id)
            return _run(row) if row else None

    def renew(self, run_id, owner_id, fencing_token, lease_seconds, *, now=None):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        current = _utc(now)
        with self._sessions.begin() as session:
            row = _owned(session, run_id, owner_id, fencing_token, current)
            row.status = OosRunStatus.RUNNING.value
            row.lease_expires_at = current + timedelta(seconds=lease_seconds)
            session.flush()
            return _run(row)

    def get_checkpoint(self, run_id, candidate_id):
        with self._sessions() as session:
            row = session.scalar(
                select(OosCandidateCheckpointRow).where(
                    OosCandidateCheckpointRow.run_id == run_id,
                    OosCandidateCheckpointRow.candidate_id == candidate_id,
                )
            )
            return _checkpoint(row) if row else None

    def put_checkpoint(self, checkpoint, owner_id, fencing_token):
        with self._sessions.begin() as session:
            _owned(session, checkpoint.run_id, owner_id, fencing_token, _utc())
            row = session.scalar(
                select(OosCandidateCheckpointRow)
                .where(OosCandidateCheckpointRow.run_id == checkpoint.run_id)
                .where(OosCandidateCheckpointRow.candidate_id == checkpoint.candidate_id)
                .with_for_update()
            )
            if row is not None:
                if row.input_hash != checkpoint.input_hash or row.result_hash != checkpoint.result_hash:
                    raise OosCheckpointConflict("checkpoint input/result hash conflict")
                return _checkpoint(row)
            session.add(
                OosCandidateCheckpointRow(
                    run_id=checkpoint.run_id,
                    candidate_id=checkpoint.candidate_id,
                    input_hash=checkpoint.input_hash,
                    status=checkpoint.status.value,
                    result=checkpoint.result,
                    result_hash=checkpoint.result_hash,
                    error=checkpoint.error,
                    completed_at=checkpoint.completed_at,
                )
            )
            session.flush()
            return checkpoint

    def interrupt(self, run_id, owner_id, fencing_token, *, terminal=False):
        with self._sessions.begin() as session:
            row = _owned(session, run_id, owner_id, fencing_token, _utc(), allow_expired=True)
            row.status = (OosRunStatus.FAILED_TERMINAL if terminal else OosRunStatus.INTERRUPTED).value
            auth = session.get(OosAuthorizationRow, row.authorization_id, with_for_update=True)
            auth.status = (
                AuthorizationStatus.INVALIDATED if terminal else AuthorizationStatus.EVALUATING_INTERRUPTED
            ).value
            auth.version += 1
            session.flush()
            return _run(row)

    def seal(self, run_id, owner_id, fencing_token, cohort_artifact_hash, artifact=None):
        with self._sessions.begin() as session:
            row = _owned(session, run_id, owner_id, fencing_token, _utc())
            if not cohort_artifact_hash:
                raise OosIdentityError("cohort artifact hash is required")
            row.status = OosRunStatus.SEALED.value
            row.cohort_artifact_hash = cohort_artifact_hash
            row.sealed_at = _utc()
            auth = session.get(OosAuthorizationRow, row.authorization_id, with_for_update=True)
            session.add(
                OosCohortArtifactRow(
                    artifact_hash=cohort_artifact_hash,
                    run_id=run_id,
                    authorization_id=row.authorization_id,
                    artifact=artifact or {},
                )
            )
            auth.status = AuthorizationStatus.CONSUMED.value
            auth.consumed_at = _utc()
            auth.version += 1
            session.flush()
            return _run(row)

    def get_artifact(self, cohort_artifact_hash):
        with self._sessions() as session:
            row = session.get(OosCohortArtifactRow, cohort_artifact_hash)
            return row.artifact if row else None


def _utc(value=None):
    result = value or datetime.now(UTC)
    if result.tzinfo is None or result.utcoffset() is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _owned(session, run_id, owner_id, fencing_token, now, *, allow_expired=False):
    row = session.get(OosEvaluationRunRow, run_id, with_for_update=True)
    if row is None or row.owner_id != owner_id or row.fencing_token != fencing_token:
        raise OosLeaseError("owner or fencing token is stale")
    auth = session.get(OosAuthorizationRow, row.authorization_id, with_for_update=True)
    if auth is None or auth.active_run_id != run_id:
        raise OosLeaseError("run is no longer active for its authorization")
    if not allow_expired and _utc(row.lease_expires_at) <= now:
        raise OosLeaseError("OOS lease expired")
    return row


def _auth(row):
    return OosAuthorization(
        authorization_id=row.authorization_id,
        experiment_id=row.experiment_id,
        candidate_set_hash=row.candidate_set_hash,
        dataset_ref_hash=row.dataset_ref_hash or row.final_oos_snapshot_id,
        market_snapshot_id=row.market_snapshot_id or row.final_oos_snapshot_id,
        content_ref_hash=row.content_ref_hash,
        status=AuthorizationStatus(row.status),
        active_run_id=row.active_run_id,
        authorized_at=row.authorized_at,
        consumed_at=row.consumed_at,
        invalidated_at=row.invalidated_at,
        version=row.version,
    )


def _run(row):
    return OosEvaluationRun(
        run_id=row.run_id,
        authorization_id=row.authorization_id,
        owner_id=row.owner_id,
        fencing_token=row.fencing_token,
        status=OosRunStatus(row.status),
        lease_expires_at=_utc(row.lease_expires_at),
        evaluator_version=row.evaluator_version,
        cohort_artifact_hash=row.cohort_artifact_hash,
        started_at=_utc(row.started_at),
        sealed_at=_utc(row.sealed_at) if row.sealed_at else None,
    )


def _checkpoint(row):
    return OosCandidateCheckpoint(
        run_id=row.run_id,
        candidate_id=row.candidate_id,
        input_hash=row.input_hash,
        status=CheckpointStatus(row.status),
        result=row.result,
        result_hash=row.result_hash,
        error=row.error,
        completed_at=_utc(row.completed_at) if row.completed_at else None,
    )


def _auth_identity(row):
    return OosAuthorization(
        authorization_id=row.authorization_id,
        experiment_id=row.experiment_id,
        candidate_set_hash=row.candidate_set_hash,
        dataset_ref_hash=row.dataset_ref_hash or row.final_oos_snapshot_id,
        market_snapshot_id=row.market_snapshot_id or row.final_oos_snapshot_id,
        content_ref_hash=row.content_ref_hash,
    ).identity_hash


__all__ = ["PostgresOosRunRepository"]
