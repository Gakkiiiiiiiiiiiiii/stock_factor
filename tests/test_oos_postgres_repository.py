from sqlalchemy import event

from stock_factor.adapters.postgres.database import Database
from stock_factor.adapters.postgres.oos_run_repository import PostgresOosRunRepository
from stock_factor.application.oos_run_service import OosRunService
from stock_factor.domain.oos_run import canonical_candidate_set_hash


def test_sqlite_create_all_supports_resumable_oos_repository():
    database = Database("sqlite://")
    database.create_schema()
    service = OosRunService(PostgresOosRunRepository(database.session_factory))
    authorization = service.authorize(
        "exp-sqlite", canonical_candidate_set_hash(["candidate-1"]), "dataset-ref", "market-1"
    )
    result = service.evaluate_cohort(
        authorization.authorization_id,
        [{"candidate_id": "candidate-1", "value": 1}],
        lambda candidate: {"value": candidate["value"]},
        run_id="run-sqlite",
        owner_id="worker-1",
    )
    assert result["status"] == "SEALED"


def test_postgres_seal_artifact_failure_rolls_back_consumption():
    database = Database("sqlite://")
    database.create_schema()
    repository = PostgresOosRunRepository(database.session_factory)
    service = OosRunService(repository)
    authorization = service.authorize(
        "exp-seal-rollback", canonical_candidate_set_hash(["candidate-1"]), "dataset-ref", "market-1"
    )
    run = repository.start_or_resume(authorization.authorization_id, "run-rollback", "worker-1", 900)

    def fail_artifact_insert(_connection, _cursor, statement, _parameters, _context, _executemany):
        if "oos_cohort_artifacts" in statement:
            raise RuntimeError("artifact storage unavailable")

    event.listen(database.engine, "before_cursor_execute", fail_artifact_insert)
    try:
        try:
            repository.seal(run.run_id, "worker-1", run.fencing_token, "artifact-hash", {})
        except RuntimeError as exc:
            assert "artifact storage unavailable" in str(exc)
        else:
            raise AssertionError("seal should fail when artifact persistence fails")
    finally:
        event.remove(database.engine, "before_cursor_execute", fail_artifact_insert)

    assert repository.get_authorization(authorization.authorization_id).status.value == "EVALUATING"
    assert repository.get_run(run.run_id).status.value == "RUNNING"
