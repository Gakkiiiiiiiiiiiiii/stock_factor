from datetime import date

from stock_factor.ports.trading_calendar import WeekdayTradingCalendar


def test_weekday_calendar_skips_weekend():
    assert WeekdayTradingCalendar().next_trading_day("CN", date(2026, 8, 14)) == date(2026, 8, 17)
