from __future__ import annotations

import os
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


class ExchangeTradingCalendar:
    """Exchange-aware calendar adapter.

    ``exchange_calendars`` is used when deployed; a calendar outage is an
    explicit error instead of silently trading through a holiday.  The small
    deterministic fallback is retained only for development when opted in.
    """

    _NAMES = {"CN": "XSHG", "HK": "XHKG", "US": "XNYS"}

    def __init__(self, allow_weekday_fallback: bool | None = None) -> None:
        self._fallback = WeekdayTradingCalendar()
        self._allow_fallback = (
            bool(int(os.getenv("FACTOR_ALLOW_WEEKDAY_CALENDAR", "0")))
            if allow_weekday_fallback is None
            else allow_weekday_fallback
        )
        try:
            import exchange_calendars as xcals

            self._xcals = xcals
        except ImportError:
            self._xcals = None

    def _calendar(self, exchange: str):
        if self._xcals is None:
            if self._allow_fallback:
                return None
            raise RuntimeError("TRADING_CALENDAR_UNAVAILABLE")
        name = self._NAMES.get(exchange.upper())
        if not name:
            raise ValueError("INVALID_SYMBOL")
        return self._xcals.get_calendar(name)

    def is_trading_day(self, exchange: str, value: date) -> bool:
        calendar = self._calendar(exchange)
        return self._fallback.is_trading_day(exchange, value) if calendar is None else calendar.is_session(str(value))

    def next_trading_day(self, exchange: str, value: date) -> date:
        calendar = self._calendar(exchange)
        if calendar is None:
            return self._fallback.next_trading_day(exchange, value)
        return calendar.next_session(str(value)).date()

    def previous_trading_day(self, exchange: str, value: date) -> date:
        calendar = self._calendar(exchange)
        if calendar is None:
            return self._fallback.previous_trading_day(exchange, value)
        return calendar.previous_session(str(value)).date()
