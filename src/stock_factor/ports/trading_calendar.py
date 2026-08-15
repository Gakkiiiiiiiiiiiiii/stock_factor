from __future__ import annotations

from datetime import date, timedelta
from typing import Protocol


class TradingCalendar(Protocol):
    def is_trading_day(self, exchange: str, value: date) -> bool: ...
    def next_trading_day(self, exchange: str, value: date) -> date: ...
    def previous_trading_day(self, exchange: str, value: date) -> date: ...


class WeekdayTradingCalendar:
    """Deterministic fallback calendar; production adapters may add exchange holidays."""

    def is_trading_day(self, exchange: str, value: date) -> bool:
        return value.weekday() < 5

    def next_trading_day(self, exchange: str, value: date) -> date:
        candidate = value + timedelta(days=1)
        while not self.is_trading_day(exchange, candidate):
            candidate += timedelta(days=1)
        return candidate

    def previous_trading_day(self, exchange: str, value: date) -> date:
        candidate = value - timedelta(days=1)
        while not self.is_trading_day(exchange, candidate):
            candidate -= timedelta(days=1)
        return candidate
