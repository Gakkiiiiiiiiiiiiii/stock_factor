from sqlalchemy import select

from stock_factor.adapters.postgres.models import PaperCashLedgerRow, PaperFillRow, PaperPositionLotRow
from stock_factor.api.dependencies import build_application
from tests.test_integration import FixtureContent, FixtureMarket


def test_paper_execution_persists_lots_fills_and_ledger_and_consumes_fifo(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'paper.db'}", FixtureMarket(), FixtureContent())
    application.generate_paper_orders(
        [{"symbol": "600000", "score": 1.0}], "2026-08-12", "snapshot-1", 1
    )
    bought = application.run_paper(
        "2026-08-13", "snapshot-1", {"600000": {"open": 10.0, "close": 10.0, "volume": 1_000_000}}
    )
    assert bought["filled_order_count"] == 1
    application.generate_paper_orders([], "2026-08-14", "snapshot-1", 1)
    sold = application.run_paper(
        "2026-08-17", "snapshot-1", {"600000": {"open": 11.0, "close": 11.0, "volume": 1_000_000}}
    )
    assert sold["filled_order_count"] == 1
    repository = application._paper._repository
    with repository._sessions() as session:
        fills = session.scalars(select(PaperFillRow)).all()
        ledgers = session.scalars(select(PaperCashLedgerRow)).all()
        lots = session.scalars(select(PaperPositionLotRow)).all()
    assert len(fills) == 2
    assert {item.event_type for item in ledgers} >= {"BUY", "SELL", "COMMISSION"}
    assert lots and all(item.available_quantity == 0 for item in lots)
