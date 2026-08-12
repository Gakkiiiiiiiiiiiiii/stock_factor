from __future__ import annotations

from typing import Protocol

from stock_factor.domain.market import MarketDataSnapshot


class MarketDataProvider(Protocol):
    def get_daily_bars(self, symbols: list[str], start: str, end: str, adjust: str) -> MarketDataSnapshot: ...


class ContentSignalProvider(Protocol):
    def load_signals(self, symbols: list[str], start: str, end: str) -> list[dict]: ...


class ModelClient(Protocol):
    def complete(self, prompt: str, system: str | None = None, temperature: float | None = None) -> str: ...


class FactorRepository(Protocol):
    def list_active(self, limit: int = 20) -> list[dict]: ...
