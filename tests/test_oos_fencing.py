from datetime import UTC, datetime

import pytest

from stock_factor.application.oos_run_service import OosRunService
from stock_factor.domain.oos_run import (
    OosAuthorization,
    OosCandidateCheckpoint,
    OosEvaluationRun,
    canonical_candidate_set_hash,
    canonical_cohort_hash,
)
from stock_factor.ports.oos_run_repository import (
    InMemoryOosRunRepository,
    OosCheckpointConflict,
    OosIdentityError,
    OosLeaseError,
)


def test_lease_takeover_fences_old_owner_and_token():
    repo = InMemoryOosRunRepository()
    service = OosRunService(repo)
    auth = service.authorize("exp-fence", "set", "dataset", "market")
    started = repo.start_or_resume(
        auth.authorization_id,
        "run-1",
        "owner-a",
        1,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    taken = repo.start_or_resume(
        auth.authorization_id,
        "run-1",
        "owner-b",
        60,
        now=datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC),
    )
    assert taken.fencing_token == started.fencing_token + 1
    checkpoint = OosCandidateCheckpoint.completed("run-1", "c1", "input", {"ok": True})
    with pytest.raises(OosLeaseError):
        repo.put_checkpoint(checkpoint, "owner-a", started.fencing_token)
    with pytest.raises(OosLeaseError):
        repo.renew("run-1", "owner-a", started.fencing_token, 60)
    with pytest.raises(OosLeaseError):
        repo.seal("run-1", "owner-a", started.fencing_token, "artifact")


def test_checkpoint_is_idempotent_for_same_hash_and_rejects_conflict():
    repo = InMemoryOosRunRepository()
    service = OosRunService(repo)
    auth = service.authorize("exp-checkpoint", "set", "dataset", "market")
    run = service.start_or_resume(auth.authorization_id, run_id="run-1", owner_id="owner")
    first = OosCandidateCheckpoint.completed("run-1", "c1", "input", {"score": 1})
    assert repo.put_checkpoint(first, "owner", run.fencing_token) == first
    assert repo.put_checkpoint(first, "owner", run.fencing_token) == first
    conflict = OosCandidateCheckpoint.completed("run-1", "c1", "different", {"score": 2})
    with pytest.raises(OosCheckpointConflict):
        repo.put_checkpoint(conflict, "owner", run.fencing_token)


def test_only_one_active_owner_can_start_run():
    repo = InMemoryOosRunRepository()
    service = OosRunService(repo)
    auth = service.authorize("exp-concurrent", "set", "dataset", "market")
    service.start_or_resume(auth.authorization_id, run_id="run-1", owner_id="owner-a")
    with pytest.raises(OosLeaseError):
        service.start_or_resume(auth.authorization_id, run_id="run-2", owner_id="owner-b")


def test_different_run_after_expiry_is_rejected_and_candidate_set_mismatch_invalidates():
    repo = InMemoryOosRunRepository()
    service = OosRunService(repo)
    auth = service.authorize("exp-run", canonical_candidate_set_hash(["c1"]), "dataset", "market")
    repo.start_or_resume(auth.authorization_id, "run-1", "owner-a", 1, now=datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(OosLeaseError):
        repo.start_or_resume(
            auth.authorization_id, "run-2", "owner-b", 60, now=datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)
        )
    with pytest.raises(OosIdentityError):
        service.evaluate_cohort(auth.authorization_id, [{"candidate_id": "wrong"}], lambda _: {"ok": True})
    assert repo.get_authorization(auth.authorization_id).status.value == "INVALIDATED"


def test_mismatch_from_other_owner_cannot_invalidate_active_authorization():
    repo = InMemoryOosRunRepository()
    service = OosRunService(repo)
    auth = service.authorize("exp-owner-mismatch", canonical_candidate_set_hash(["c1"]), "dataset", "market")
    run = service.start_or_resume(auth.authorization_id, owner_id="owner-a")
    with pytest.raises(OosLeaseError):
        service.evaluate_cohort(
            auth.authorization_id,
            [{"candidate_id": "wrong"}],
            lambda _: {"ok": True},
            owner_id="owner-b",
        )
    current = repo.get_authorization(auth.authorization_id)
    assert current.status.value == "EVALUATING"
    assert current.active_run_id == run.run_id
    result = service.evaluate_cohort(
        auth.authorization_id,
        [{"candidate_id": "c1"}],
        lambda _: {"ok": True},
        owner_id="owner-a",
    )
    assert result["status"] == "SEALED"


def test_non_positive_lease_is_rejected():
    repo = InMemoryOosRunRepository()
    service = OosRunService(repo)
    auth = service.authorize("exp-lease-value", canonical_candidate_set_hash(["c1"]), "dataset", "market")
    with pytest.raises(ValueError):
        service.start_or_resume(auth.authorization_id, owner_id="owner", lease_seconds=0)


def test_cohort_hash_is_order_stable_and_evaluator_version_is_persisted():
    repo = InMemoryOosRunRepository()
    service = OosRunService(repo)
    ids_hash = canonical_candidate_set_hash(["c1", "c2"])
    assert ids_hash == canonical_candidate_set_hash(["c2", "c1"])
    auth = service.authorize("exp-version", ids_hash, "dataset", "market")
    run = service.start_or_resume(auth.authorization_id, owner_id="owner", evaluator_version="eval-v2")
    assert run.evaluator_version == "eval-v2"
    assert (
        service.start_or_resume(auth.authorization_id, owner_id="owner", evaluator_version="eval-v2").run_id
        == run.run_id
    )
    with pytest.raises(OosIdentityError):
        service.start_or_resume(auth.authorization_id, owner_id="owner", evaluator_version="eval-v1")
    a = OosAuthorization("a", "e", ids_hash, "dataset", "market")
    r = OosEvaluationRun("r", "a", "owner", evaluator_version="eval-v1")
    left = [
        {"candidate_id": "c2", "input_hash": "i2", "result_hash": "r2", "result": {"v": 2}},
        {"candidate_id": "c1", "input_hash": "i1", "result_hash": "r1", "result": {"v": 1}},
    ]
    right = list(reversed(left))
    assert canonical_cohort_hash(left, authorization=a, run=r) == canonical_cohort_hash(right, authorization=a, run=r)


def test_evaluator_value_error_is_retryable_not_terminal():
    repo = InMemoryOosRunRepository()
    service = OosRunService(repo)
    auth = service.authorize("exp-retryable", canonical_candidate_set_hash(["c1"]), "dataset", "market")

    with pytest.raises(RuntimeError):
        service.evaluate_cohort(
            auth.authorization_id,
            [{"candidate_id": "c1", "value": 1}],
            lambda _: (_ for _ in ()).throw(ValueError("temporary backend failure")),
            run_id="run-retryable",
            owner_id="owner",
        )

    assert repo.get_authorization(auth.authorization_id).status.value == "EVALUATING_INTERRUPTED"
    result = service.evaluate_cohort(
        auth.authorization_id,
        [{"candidate_id": "c1", "value": 1}],
        lambda _: {"ok": True},
        run_id="run-retryable",
        owner_id="owner",
    )
    assert result["status"] == "SEALED"
