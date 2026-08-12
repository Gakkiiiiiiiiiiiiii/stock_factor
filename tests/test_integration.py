from datetime import UTC, datetime, timedelta

import numpy as np

from stock_factor.api.dependencies import build_application
from stock_factor.domain.market import MarketDataSnapshot


class FixtureMarket:
    def get_daily_bars(self, symbols, start, end, adjust):
        days = 90
        dates = [(datetime.now(UTC).date() - timedelta(days=days - index)).isoformat() for index in range(days)]
        rng = np.random.default_rng(4)
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, (len(symbols), days)), axis=1)
        volume = rng.uniform(1e6, 2e6, close.shape)
        return MarketDataSnapshot(symbols, dates, {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": volume, "amount": volume * close, "turnover": np.ones_like(close)}, "fixture-v1", "snapshot-1", "fixture")


class FixtureContent:
    def load_signals(self, symbols, start, end):
        return [{"symbol": symbols[0], "available_from": end, "kind": "CATALYST", "sentiment": "BULLISH", "confidence": 0.8}]


def test_job_worker_persists_factor_and_real_apis(tmp_path):
    url = f"sqlite:///{tmp_path / 'factor.db'}"
    application = build_application(url, FixtureMarket(), FixtureContent())
    job = application.create_mining_job({"symbols": [f"6000{index:02d}" for index in range(20)], "candidates": [{"name": "reversal", "rpn": ["ret", "ts_mean_5", "neg", "cs_rank"]}]})
    result = application.process_next("test-worker")
    assert result["status"] == "SUCCEEDED"
    assert application.get_mining_job(job["job_id"])["status"] == "SUCCEEDED"
    evaluation = application.evaluate(None, ["ret", "ts_mean_5", "neg", "cs_rank"], [f"6000{index:02d}" for index in range(20)], None, None, 5)
    assert evaluation["data_snapshot_id"] == "snapshot-1"
    frozen = application.generate_paper_orders([{"symbol": "600000", "score": 0.9, "next_trading_day": "2026-08-13"}], "2026-08-12", "snapshot-1", 10)
    assert frozen["orders"][0]["status"] == "FROZEN"
    assert application.paper_state()["data_snapshot_id"] == "snapshot-1"


def test_job_survives_application_restart(tmp_path):
    url = f"sqlite:///{tmp_path / 'factor.db'}"
    first = build_application(url, FixtureMarket(), FixtureContent())
    job = first.create_mining_job({"symbols": ["600000"]})
    second = build_application(url, FixtureMarket(), FixtureContent())
    assert second.get_mining_job(job["job_id"])["status"] == "PENDING"
