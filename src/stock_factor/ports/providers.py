from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from stock_factor.domain.content_signal_v5 import FormalContentQuery, FormalContentRef
from stock_factor.domain.factor import FactorDefinition, FactorJob
from stock_factor.domain.market import MarketDataSnapshot
from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef


class MarketDataProvider(Protocol):
    def get_daily_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        adjust: str,
        *,
        formal_market_ref: FormalMarketDatasetRef | None = None,
        expected_ref: FormalMarketDatasetRef | None = None,
        as_of: datetime | None = None,
    ) -> MarketDataSnapshot: ...


class ContentSignalProvider(Protocol):
    def load_signals(
        self,
        symbols: list[str],
        start: str,
        end: str,
        *,
        query: FormalContentQuery | None = None,
        expected_ref: FormalContentRef | None = None,
    ) -> list[dict]: ...


class ModelClient(Protocol):
    def complete(self, prompt: str, system: str | None = None, temperature: float | None = None) -> str: ...


class FactorRepository(Protocol):
    def list_active(self, limit: int = 20) -> list[dict]: ...

    def get(self, factor_id: str) -> dict | None: ...

    def save(self, factor: FactorDefinition) -> dict: ...


class FactorJobRepository(Protocol):
    def create(self, job: FactorJob) -> FactorJob: ...

    def get(self, job_id: str) -> FactorJob | None: ...

    def cancel(self, job_id: str) -> FactorJob | None: ...

    def claim_pending(self, worker_id: str, lease_seconds: int) -> FactorJob | None: ...

    def progress(self, job_id: str, stage: str, progress: int) -> None: ...

    def succeed(self, job_id: str, result: dict[str, Any]) -> None: ...

    def fail(self, job_id: str, stage: str, error: str) -> None: ...
