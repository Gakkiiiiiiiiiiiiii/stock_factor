"""Exactly-once/resumable Final OOS orchestration."""

from __future__ import annotations

import hashlib
import json
from typing import Callable

import numpy as np

from stock_factor.domain.oos_run import (
    AuthorizationStatus,
    CheckpointStatus,
    OosAuthorization,
    OosCandidateCheckpoint,
    canonical_candidate_set_hash,
    canonical_cohort_hash,
)
from stock_factor.engine.oos_seal import CandidateUnfrozenError, OosWindowInvalidatedError
from stock_factor.ports.oos_run_repository import (
    OosCheckpointConflict,
    OosIdentityError,
    OosLeaseError,
    OosRunRepository,
)


class OosRunService:
    def __init__(self, repository: OosRunRepository) -> None:
        self._repository = repository

    def authorize(
        self,
        experiment_id: str,
        candidate_set_hash: str,
        dataset_ref_hash: str,
        market_snapshot_id: str,
        *,
        content_ref_hash: str | None = None,
    ) -> OosAuthorization:
        authorization = OosAuthorization(
            authorization_id=f"oosa-{experiment_id}",
            experiment_id=experiment_id,
            candidate_set_hash=candidate_set_hash,
            dataset_ref_hash=dataset_ref_hash,
            market_snapshot_id=market_snapshot_id,
            content_ref_hash=content_ref_hash,
        )
        return self._repository.save_authorization(authorization)

    def start_or_resume(
        self,
        authorization_id: str,
        *,
        run_id: str | None = None,
        owner_id: str,
        lease_seconds: int = 900,
        evaluator_version: str = "final-oos-v1",
    ):
        return self._repository.start_or_resume(
            authorization_id,
            run_id,
            owner_id,
            lease_seconds,
            evaluator_version=evaluator_version,
        )

    def evaluate_cohort(
        self,
        authorization_id: str,
        candidates: list[dict],
        evaluator: Callable[[dict], dict],
        *,
        run_id: str | None = None,
        owner_id: str = "oos-worker",
        lease_seconds: int = 900,
        evaluator_version: str = "final-oos-v1",
        preflight: Callable[[], None] | None = None,
    ) -> dict:
        if not candidates:
            raise ValueError("OOS cohort cannot be empty")
        auth = self._repository.get_authorization(authorization_id)
        if auth is None or auth.status in {AuthorizationStatus.CONSUMED, AuthorizationStatus.INVALIDATED}:
            raise RuntimeError("OOS authorization is unavailable")
        run = self.start_or_resume(
            authorization_id,
            run_id=run_id,
            owner_id=owner_id,
            lease_seconds=lease_seconds,
            evaluator_version=evaluator_version,
        )
        if run.evaluator_version != evaluator_version:
            raise ValueError("evaluator version mismatch")
        results: list[dict] = []
        try:
            try:
                expected_set = canonical_candidate_set_hash(
                    [str(candidate["candidate_id"]) for candidate in candidates]
                )
            except (KeyError, ValueError) as exc:
                raise OosIdentityError(f"invalid candidate set: {exc}") from exc
            if auth.candidate_set_hash != expected_set:
                raise OosIdentityError("candidate set identity mismatch")
            for candidate in candidates:
                for field in ("dataset_ref_hash", "market_snapshot_id", "content_ref_hash"):
                    if field in candidate and candidate[field] != getattr(auth, field):
                        raise OosIdentityError(f"{field} identity mismatch")
            if preflight is not None:
                preflight()
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                input_hash = _input_hash(candidate)
                checkpoint = self._repository.get_checkpoint(run.run_id, candidate_id)
                if checkpoint is not None:
                    if checkpoint.input_hash != input_hash:
                        raise OosIdentityError(f"candidate {candidate_id} input identity mismatch")
                    if checkpoint.status == CheckpointStatus.COMPLETED:
                        results.append(checkpoint.result or {})
                        continue
                    if checkpoint.status == CheckpointStatus.FAILED_TERMINAL:
                        raise OosIdentityError(f"candidate {candidate_id} has terminal checkpoint")
                try:
                    result = evaluator(candidate)
                except (CandidateUnfrozenError, OosWindowInvalidatedError, OosIdentityError):
                    raise
                except Exception as exc:
                    # Evaluator failures are retryable regardless of the concrete
                    # exception type (including ValueError from a backend).
                    raise _RetryableEvaluatorError(str(exc)) from exc
                completed = OosCandidateCheckpoint.completed(run.run_id, candidate_id, input_hash, result)
                self._repository.put_checkpoint(completed, owner_id, run.fencing_token)
                results.append(result)
            auth = self._repository.get_authorization(authorization_id) or auth
            artifact_rows = []
            for candidate in candidates:
                checkpoint = self._repository.get_checkpoint(run.run_id, str(candidate["candidate_id"]))
                if checkpoint is None or checkpoint.status != CheckpointStatus.COMPLETED:
                    raise OosIdentityError("all cohort candidates must be completed before sealing")
                artifact_rows.append(
                    {
                        "candidate_id": checkpoint.candidate_id,
                        "input_hash": checkpoint.input_hash,
                        "result_hash": checkpoint.result_hash,
                        "result": checkpoint.result,
                    }
                )
            artifact_hash = canonical_cohort_hash(artifact_rows, authorization=auth, run=run)
            self._repository.seal(
                run.run_id,
                owner_id,
                run.fencing_token,
                artifact_hash,
                {
                    "authorization_id": authorization_id,
                    "dataset_ref_hash": auth.dataset_ref_hash,
                    "market_snapshot_id": auth.market_snapshot_id,
                    "content_ref_hash": auth.content_ref_hash,
                    "evaluator_version": run.evaluator_version,
                    "candidates": artifact_rows,
                },
            )
            return {"run_id": run.run_id, "cohort_artifact_hash": artifact_hash, "results": results, "status": "SEALED"}
        except OosLeaseError:
            raise
        except (OosCheckpointConflict, OosIdentityError, CandidateUnfrozenError, OosWindowInvalidatedError):
            self._repository.interrupt(run.run_id, owner_id, run.fencing_token, terminal=True)
            raise
        except Exception:
            self._repository.interrupt(run.run_id, owner_id, run.fencing_token, terminal=False)
            raise


class _RetryableEvaluatorError(RuntimeError):
    """Internal wrapper preserving evaluator failures as retryable errors."""


def _input_hash(candidate: dict) -> str:
    material = {key: value for key, value in candidate.items() if key not in {"result", "metrics", "loader", "load"}}
    values = material.get("values")
    closes = material.get("closes")
    if isinstance(values, np.ndarray):
        material["values"] = {"dtype": str(values.dtype), "shape": values.shape, "bytes": values.tobytes().hex()}
    if isinstance(closes, np.ndarray):
        material["closes"] = {"dtype": str(closes.dtype), "shape": closes.shape, "bytes": closes.tobytes().hex()}
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


__all__ = ["OosRunService"]
