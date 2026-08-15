from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np

from stock_factor.application.mining import FactorMiningService
from stock_factor.application.panel import build_feature_panel
from stock_factor.application.paper import PaperTradingService
from stock_factor.domain.factor import FactorJob
from stock_factor.engine.alpha import compose_alpha_scores
from stock_factor.engine.fitness import evaluate_factor
from stock_factor.engine.vm import StackVM
from stock_factor.ports.providers import (
    ContentSignalProvider,
    FactorJobRepository,
    FactorRepository,
    MarketDataProvider,
)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class FactorApplication:
    def __init__(self, jobs: FactorJobRepository, factors: FactorRepository, mining: FactorMiningService, market: MarketDataProvider, content: ContentSignalProvider, paper: PaperTradingService) -> None:
        self._jobs, self._factors, self._mining = jobs, factors, mining
        self._market, self._content, self._paper = market, content, paper

    def create_mining_job(self, payload: dict) -> dict:
        job = FactorJob(job_id=uuid4().hex, request=payload)
        return self._jobs.create(job).to_dict()

    def get_mining_job(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def cancel_mining_job(self, job_id: str) -> dict | None:
        job = self._jobs.cancel(job_id)
        return {"job_id": job_id, "cancelled": job.status == "CANCELLED", "status": job.status} if job else None

    def process_next(self, worker_id: str, lease_seconds: int = 900) -> dict | None:
        job = self._jobs.claim_pending(worker_id, lease_seconds)
        if job is None:
            return None
        stage = "prepare"
        try:
            result = self._mining.run(job.request, lambda name, value: self._jobs.progress(job.job_id, name, value))
            result = _json_safe(result)
            self._jobs.succeed(job.job_id, result)
            return {"job_id": job.job_id, "status": "SUCCEEDED", **result}
        except Exception as exc:
            current = self._jobs.get(job.job_id)
            stage = current.stage if current else stage
            self._jobs.fail(job.job_id, stage, f"{type(exc).__name__}: {exc}")
            return {"job_id": job.job_id, "status": "FAILED", "stage": stage, "error": str(exc)}

    def list_factors(self, limit: int) -> list[dict]:
        return self._factors.list_active(limit)

    def get_factor(self, factor_id: str) -> dict | None:
        return self._factors.get(factor_id)

    def _panel(self, symbols: list[str], start: str | None, end: str | None):
        resolved_end = end or datetime.now(UTC).date().isoformat()
        resolved_start = start or (datetime.fromisoformat(resolved_end) - timedelta(days=730)).date().isoformat()
        snapshot = self._market.get_daily_bars(symbols, resolved_start, resolved_end, "qfq")
        signals = self._content.load_signals(symbols, resolved_start, resolved_end)
        return snapshot, build_feature_panel(snapshot, signals)

    def evaluate(self, factor_id: str | None, rpn: list[str], symbols: list[str], start: str | None, end: str | None, horizon: int) -> dict:
        definition = self._factors.get(factor_id) if factor_id else None
        formula = list(definition["rpn"] if definition else rpn)
        if not formula or not symbols:
            raise ValueError("rpn/factor_id and symbols are required")
        snapshot, panel = self._panel(symbols, start, end)
        values = StackVM().execute(formula, panel)
        if values is None:
            raise ValueError("invalid or non-computable factor formula")
        metrics = _json_safe(evaluate_factor(values, panel["close"], horizon=horizon))
        return {"factor_id": factor_id, "rpn": formula, "metrics": metrics, "data_version": snapshot.data_version, "data_snapshot_id": snapshot.data_snapshot_id}

    def alpha_score(self, symbols: list[str], as_of: str | None) -> dict:
        snapshot, panel = self._panel(symbols, None, as_of)
        factors = self._factors.list_active(100)
        scores, count = compose_alpha_scores(panel, factors)
        items = [] if scores is None else [{"symbol": symbol, "score": None if np.isnan(scores[index]) else round(float(scores[index]), 8)} for index, symbol in enumerate(snapshot.symbols)]
        return {"as_of": snapshot.dates[-1] if snapshot.dates else as_of, "factor_count": count, "data_version": snapshot.data_version, "data_snapshot_id": snapshot.data_snapshot_id, "items": items}

    def generate_paper_orders(self, scores: list[dict], as_of: str, snapshot_id: str, top_k: int) -> dict:
        return self._paper.generate_orders(scores, as_of, snapshot_id, top_k)

    def run_paper(self, as_of: str, snapshot_id: str, market_prices: dict[str, dict] | None = None) -> dict:
        return self._paper.run(as_of, snapshot_id, market_prices)

    def paper_state(self) -> dict:
        return self._paper.state()

    def paper_equity(self) -> list[dict]:
        return self._paper.equity()
