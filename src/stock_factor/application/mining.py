from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from stock_factor.application.panel import build_feature_panel
from stock_factor.domain.factor import FactorDefinition
from stock_factor.engine.fitness import evaluate_factor
from stock_factor.engine.lookback import max_lookback_from_rpn
from stock_factor.engine.validation import run_purged_walkforward
from stock_factor.engine.vm import StackVM
from stock_factor.engine.vocab import is_valid_token
from stock_factor.ports.providers import (
    ContentSignalProvider,
    FactorRepository,
    MarketDataProvider,
    ModelClient,
)

SEEDS = [
    ("momentum_20", ["close", "close", "ts_delay_20", "div", "cs_rank"], "20-day price momentum"),
    ("volume_price_corr", ["close", "volume", "ts_corr_10", "neg", "cs_rank"], "price-volume divergence"),
    ("short_reversal", ["ret", "ts_mean_5", "neg", "cs_rank"], "short-term reversal"),
    ("content_sentiment", ["theme_sentiment", "ts_sum_5", "cs_rank"], "verified content sentiment"),
]


class FactorMiningService:
    def __init__(
        self,
        market: MarketDataProvider,
        content: ContentSignalProvider,
        factors: FactorRepository,
        model: ModelClient | None = None,
    ) -> None:
        self._market, self._content, self._factors, self._model = market, content, factors, model

    def _candidates(self, request: dict) -> list[dict]:
        supplied = list(request.get("candidates") or [])
        if supplied:
            return supplied
        if request.get("use_model") and self._model is not None:
            prompt = (
                "Generate JSON array of factor candidates. Each item must contain name, hypothesis and "
                "an RPN token array using only the documented stock_factor vocabulary."
            )
            parsed = json.loads(self._model.complete(prompt, system="You are a quantitative factor researcher."))
            candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else parsed
            valid = [
                item
                for item in candidates
                if isinstance(item, dict)
                and item.get("rpn")
                and all(is_valid_token(str(token)) for token in item["rpn"])
            ]
            if valid:
                return valid
        return [{"name": name, "rpn": rpn, "hypothesis": hypothesis} for name, rpn, hypothesis in SEEDS]

    def run(self, request: dict, progress=lambda stage, value: None) -> dict:
        symbols = request.get("symbols") or []
        if not symbols:
            raise ValueError("symbols must not be empty")
        end = request.get("end") or datetime.now(UTC).date().isoformat()
        start = request.get("start") or (datetime.fromisoformat(end) - timedelta(days=int(request.get("days") or 365) * 2)).date().isoformat()
        progress("data", 10)
        snapshot = self._market.get_daily_bars(symbols, start, end, "qfq")
        signals = self._content.load_signals(symbols, start, end)
        panel = build_feature_panel(snapshot, signals)
        candidates = self._candidates(request)
        accepted = []
        for index, candidate in enumerate(candidates):
            rpn = list(candidate.get("rpn") or [])
            max_lookback_from_rpn(rpn)
            values = StackVM().execute(rpn, panel)
            if values is None:
                continue
            metrics = evaluate_factor(values, panel["close"], horizon=int(request.get("horizon") or 5), eval_window=request.get("eval_window"))
            walkforward = run_purged_walkforward(values, panel["close"], horizon=int(request.get("horizon") or 5))
            metrics["walkforward"] = walkforward
            digest = hashlib.sha256(" ".join(rpn).encode()).hexdigest()
            definition = FactorDefinition(factor_id=uuid4().hex, name=candidate.get("name") or digest[:12], rpn=rpn, hypothesis=candidate.get("hypothesis", ""), status="ACTIVE" if metrics.get("passed") and walkforward.get("passed") else "CANDIDATE", metrics=metrics, candidate_hash=digest)
            accepted.append(self._factors.save(definition))
            progress("evaluate", 20 + int((index + 1) * 70 / max(len(candidates), 1)))
        return {"factor_count": len(accepted), "factors": accepted, "data_version": snapshot.data_version, "data_snapshot_id": snapshot.data_snapshot_id, "content_signal_count": len(signals)}
