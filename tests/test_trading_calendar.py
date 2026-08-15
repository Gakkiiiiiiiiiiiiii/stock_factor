from datetime import date

from stock_factor.application.paper import PaperTradingService
from stock_factor.ports.trading_calendar import WeekdayTradingCalendar


def test_weekday_calendar_skips_weekend():
    assert WeekdayTradingCalendar().next_trading_day("CN", date(2026, 8, 14)) == date(2026, 8, 17)


def test_paper_uses_canonical_hk_symbol_exchange():
    class Calendar:
        exchange = None

        def next_trading_day(self, exchange, value):
            self.exchange = exchange
            return value

    class Repository:
        def state(self):
            return {"positions": {}}

        def freeze(self, orders, snapshot_id):
            return {"orders": orders, "data_snapshot_id": snapshot_id}

    calendar = Calendar()
    PaperTradingService(Repository(), calendar).generate_orders(
        [{"symbol": "HK.00700", "score": 1.0}], "2026-08-14", "snapshot", 1
    )
    assert calendar.exchange == "HK"
