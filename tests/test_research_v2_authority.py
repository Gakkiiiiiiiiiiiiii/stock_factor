from sqlalchemy import select

from stock_factor.adapters.postgres.models import FactorFinalOosRow, FactorOosAuditRow, FactorStatisticalTestRow
from stock_factor.api.dependencies import build_application
from tests.test_integration import FixtureContent, FixtureMarket


def test_research_statistics_oos_and_audit_have_normalized_authority(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'research.db'}", FixtureMarket(), FixtureContent())
    job = application.create_mining_job(
        {"symbols": [f"6000{index:02d}" for index in range(20)], "candidates": [{"name": "r", "rpn": ["ret", "cs_rank"]}]}
    )
    assert application.process_next("authority-test")["status"] == "SUCCEEDED"
    repository = application._factors
    with repository._sessions() as session:
        assert session.scalars(select(FactorStatisticalTestRow)).all()
        assert session.scalars(select(FactorFinalOosRow)).all()
        assert session.scalars(select(FactorOosAuditRow)).all()
    assert application.get_mining_job(job["job_id"])["status"] == "SUCCEEDED"
