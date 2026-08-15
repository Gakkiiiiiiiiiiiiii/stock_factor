from __future__ import annotations

import hashlib
import json
from math import erfc, sqrt
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np

from stock_factor.application.panel import build_feature_panel
from stock_factor.domain.factor import FactorDefinition
from stock_factor.engine.diagnostics import (
    compute_capacity_proxy,
    compute_factor_autocorrelation,
    compute_ic_decay,
    compute_turnover,
)
from stock_factor.engine.exposure import compute_factor_exposures
from stock_factor.engine.fitness import evaluate_factor, evaluate_factor_range
from stock_factor.engine.lookback import max_lookback_from_rpn
from stock_factor.engine.promotion_gate import evaluate_promotion_gate
from stock_factor.engine.purged_walkforward import run_purged_walkforward
from stock_factor.engine.research_split import build_research_split
from stock_factor.engine.statistical_validation import validate_factor_statistics
from stock_factor.engine.vm import StackVM
from stock_factor.engine.vocab import is_valid_token
from stock_factor.research_config import get_research_config
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

    @staticmethod
    def _canonical_candidates(candidates: list[dict], budget: int) -> list[dict]:
        """Apply a deterministic candidate budget before any evaluation.

        The candidate hash is also the experiment identity used by persistence,
        preventing the same DSL expression from being counted multiple times in
        multiple-testing statistics.
        """
        selected: list[dict] = []
        seen: set[str] = set()
        for candidate in candidates:
            rpn = [str(token) for token in candidate.get("rpn") or []]
            if not rpn or not all(is_valid_token(token) for token in rpn):
                continue
            digest = hashlib.sha256(" ".join(rpn).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            selected.append({**candidate, "rpn": rpn, "candidate_hash": digest})
            if len(selected) >= budget:
                break
        return selected

    @staticmethod
    def _statistics(metrics: dict, candidate_count: int) -> dict:
        observations = int(metrics.get("valid_ic_days") or 0)
        rank_ic = float(metrics.get("rank_ic") or 0.0)
        z_score = abs(rank_ic) * sqrt(max(observations, 1))
        p_value = erfc(z_score / sqrt(2))
        # PBO requires multiple independent trial return series.  The current
        # candidate's scalar evaluation cannot fabricate one, therefore a
        # conservative one-row series makes the gate fail closed until the
        # experiment runner supplies the full candidate return matrix.
        returns_by_trial = np.asarray([[rank_ic] * max(observations, 1)], dtype=float)
        return validate_factor_statistics(
            [p_value],
            sharpe=float(metrics.get("icir") or 0.0),
            observations=observations,
            trials=max(candidate_count, 1),
            returns_by_trial=returns_by_trial,
        )

    @staticmethod
    def _diagnostics(values: np.ndarray, panel: dict) -> tuple[dict, dict, dict]:
        volume = np.asarray(panel.get("volume"), dtype=float)
        latest = values[:, -1] if values.ndim == 2 and values.shape[1] else np.asarray([])
        liquidity = volume[:, -1] if volume.ndim == 2 and volume.shape[1] else None
        diagnostics = {
            "autocorrelation_lag_1": compute_factor_autocorrelation(values),
            "ic_decay": compute_ic_decay(values, np.asarray(panel["close"], dtype=float), max_horizon=5),
            "turnover": compute_turnover(values, top_k=max(5, int(values.shape[0] * 0.01))),
        }
        exposure = compute_factor_exposures(latest, liquidity=liquidity)
        capacity = compute_capacity_proxy(volume)
        return diagnostics, exposure, capacity

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
        config = get_research_config()
        budget = max(1, min(int(request.get("candidate_budget") or 50), 200))
        candidates = self._canonical_candidates(self._candidates(request), budget)
        accepted = []
        for index, candidate in enumerate(candidates):
            rpn = candidate["rpn"]
            max_lookback_from_rpn(rpn)
            values = StackVM().execute(rpn, panel)
            if values is None:
                continue
            horizon = int(request.get("horizon") or config.evaluation.horizon_days)
            split = build_research_split(values.shape[1], config.data_split, horizon)
            preliminary = evaluate_factor(values, panel["close"], horizon=horizon, eval_window=request.get("eval_window"))
            if split is None:
                walkforward = {"passed": False, "reason": "INSUFFICIENT_RESEARCH_HISTORY"}
                final_oos = {"passed": False, "reason": "INSUFFICIENT_RESEARCH_HISTORY"}
            else:
                walkforward = run_purged_walkforward(
                    values,
                    panel["close"],
                    eval_start=split.discovery_start,
                    eval_end=split.discovery_end,
                    horizon=horizon,
                )
                final_oos = evaluate_factor_range(
                    values,
                    panel["close"],
                    split.final_oos_start,
                    split.final_oos_end,
                    horizon=horizon,
                )
            diagnostics, exposure, capacity = self._diagnostics(values, panel)
            statistics = self._statistics(preliminary, len(candidates))
            promotion = evaluate_promotion_gate(
                walkforward=walkforward,
                statistics=statistics,
                diagnostics=diagnostics,
                exposure=exposure,
                capacity=capacity,
                data_snapshot_id=snapshot.data_snapshot_id,
            ).model_dump()
            # Mining never activates a factor.  A passing research gate only
            # makes it eligible for the separate paper-trading approval flow.
            status = "PAPER_TRADING" if promotion["passed"] and final_oos.get("passed") else "CANDIDATE"
            metrics = {
                "in_sample": preliminary,
                "walkforward": walkforward,
                "final_oos": final_oos,
                "statistics": statistics,
                "diagnostics": diagnostics,
                "exposure": exposure,
                "capacity": capacity,
                "promotion_gate": promotion,
                "research_split": split.diagnostics(horizon, values.shape[1]) if split else None,
                "data_version": snapshot.data_version,
                "data_snapshot_id": snapshot.data_snapshot_id,
            }
            definition = FactorDefinition(factor_id=uuid4().hex, name=candidate.get("name") or candidate["candidate_hash"][:12], rpn=rpn, hypothesis=candidate.get("hypothesis", ""), status=status, metrics=metrics, candidate_hash=candidate["candidate_hash"])
            accepted.append(self._factors.save(definition))
            progress("evaluate", 20 + int((index + 1) * 70 / max(len(candidates), 1)))
        return {"factor_count": len(accepted), "factors": accepted, "data_version": snapshot.data_version, "data_snapshot_id": snapshot.data_snapshot_id, "content_signal_count": len(signals)}
