from __future__ import annotations

from stock_factor.adapters.http import HttpContentSignalProvider, HttpMarketDataProvider, HttpModelClient
from stock_factor.adapters.postgres import Database
from stock_factor.adapters.postgres.repositories import (
    PostgresFactorJobRepository,
    PostgresFactorRepository,
    PostgresPaperRepository,
)
from stock_factor.application.mining import FactorMiningService
from stock_factor.application.paper import PaperTradingService
from stock_factor.application.service import FactorApplication
from stock_factor.ports.trading_calendar import ExchangeTradingCalendar


def build_application(database_url: str | None = None, market=None, content=None, model=None) -> FactorApplication:
    database = Database(database_url)
    database.create_schema()
    market_provider = market or HttpMarketDataProvider()
    content_provider = content or HttpContentSignalProvider()
    factors = PostgresFactorRepository(database.session_factory)
    jobs = PostgresFactorJobRepository(database.session_factory)
    paper_repository = PostgresPaperRepository(database.session_factory)
    model_client = model or HttpModelClient()
    mining = FactorMiningService(market_provider, content_provider, factors, model_client)
    return FactorApplication(
        jobs,
        factors,
        mining,
        market_provider,
        content_provider,
        PaperTradingService(
            paper_repository,
            ExchangeTradingCalendar(allow_weekday_fallback=bool(database_url and database_url.startswith("sqlite"))),
        ),
    )
