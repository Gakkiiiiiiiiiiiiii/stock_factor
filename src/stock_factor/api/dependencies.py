from __future__ import annotations

from stock_factor.adapters.http import (
    ContentSignalProviderRouter,
    HttpModelClient,
    MarketDataProviderRouter,
    QuantPaperClient,
)
from stock_factor.adapters.postgres import Database
from stock_factor.adapters.postgres.oos_run_repository import PostgresOosRunRepository
from stock_factor.adapters.postgres.repositories import (
    PostgresCandidateSealStore,
    PostgresFactorJobRepository,
    PostgresFactorRepository,
)
from stock_factor.adapters.postgres.research_artifact_repository import PostgresResearchArtifactRepository
from stock_factor.application.artifacts.seal import ResearchArtifactService
from stock_factor.application.mining import FactorMiningService
from stock_factor.application.oos.evaluate import FinalOosEvaluationService
from stock_factor.application.readiness import ReadinessService
from stock_factor.application.service import FactorApplication
from stock_factor.config.runtime import RuntimeConfig, RuntimeConfigurationError
from stock_factor.domain.authority import PaperAuthority


def build_application(
    database_url: str | None = None,
    market=None,
    content=None,
    model=None,
    *,
    runtime_config: RuntimeConfig | None = None,
    paper_authority: PaperAuthority | str | None = None,
    expected_execution_cost_calibration=None,
) -> FactorApplication:
    config = runtime_config or RuntimeConfig.from_env(paper_authority=paper_authority)
    if config.paper_authority is not PaperAuthority.QUANT:
        raise RuntimeConfigurationError(
            "production composition requires Quant Paper Authority; local paper is not reachable from src"
        )
    database = Database(database_url)
    database.create_schema()
    market_provider = market or MarketDataProviderRouter()
    content_provider = content or ContentSignalProviderRouter()
    factors = PostgresFactorRepository(database.session_factory)
    jobs = PostgresFactorJobRepository(database.session_factory)
    model_client = model or HttpModelClient()
    # Final OOS 真隔离：独立冻结/评估存储（设计文档 §13/§78）。
    oos_repository = PostgresOosRunRepository(database.session_factory)
    final_oos_service = FinalOosEvaluationService(
        PostgresCandidateSealStore(database.session_factory), run_repository=oos_repository
    )
    research_artifact_service = ResearchArtifactService(PostgresResearchArtifactRepository(database.session_factory))
    readiness_service = ReadinessService(
        config, database=database, artifact_store=research_artifact_service, oos_repository=oos_repository
    )
    mining = FactorMiningService(
        market_provider,
        content_provider,
        factors,
        model_client,
        final_oos_service=final_oos_service,
        research_artifact_service=research_artifact_service,
        expected_execution_cost_calibration=expected_execution_cost_calibration,
        readiness_service=readiness_service,
    )
    # Paper authority is always Quant in the production composition root.
    # Local paper execution is intentionally not importable from this path.
    paper_service = QuantPaperClient(config.quant_base_url)
    return FactorApplication(
        jobs,
        factors,
        mining,
        market_provider,
        content_provider,
        paper_service,
        config,
        research_artifact_service=research_artifact_service,
        readiness_service=readiness_service,
    )
