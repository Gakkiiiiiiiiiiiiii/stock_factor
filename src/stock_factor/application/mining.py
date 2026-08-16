from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np

from stock_factor.application.final_oos_evaluation import (
    FinalOosEvaluationService,
    InMemoryCandidateSealStore,
)
from stock_factor.application.panel import build_feature_panel
from stock_factor.application.seed_library import Alpha191SeedLibrary
from stock_factor.application.statistical_experiment import (
    rank_ic_series,
    validate_statistical_experiment,
)
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
from stock_factor.engine.oos_audit import audit_final_oos
from stock_factor.engine.oos_seal import (
    DSL_VERSION,
    CandidateFreeze,
    OosWindowInvalidatedError,
    derive_snapshot_refs,
    feature_set_version,
)
from stock_factor.engine.promotion_gate import evaluate_promotion_gate
from stock_factor.engine.purged_walkforward import run_purged_walkforward
from stock_factor.engine.research_split import build_research_split
from stock_factor.engine.vm import StackVM
from stock_factor.engine.vocab import is_valid_token
from stock_factor.ports.providers import (
    ContentSignalProvider,
    FactorRepository,
    MarketDataProvider,
    ModelClient,
)
from stock_factor.research_config import get_research_config


class ResearchIntegrityError(ValueError):
    """Raised before persisting a candidate whose identity no longer matches its formula."""


class FactorMiningService:
    def __init__(
        self,
        market: MarketDataProvider,
        content: ContentSignalProvider,
        factors: FactorRepository,
        model: ModelClient | None = None,
        seeds: Alpha191SeedLibrary | None = None,
        final_oos_service: FinalOosEvaluationService | None = None,
    ) -> None:
        self._market, self._content, self._factors, self._model = market, content, factors, model
        self._seeds = seeds or Alpha191SeedLibrary()
        # Final OOS 必须独立于候选搜索（设计文档 §13/§78）。
        self._oos_service = final_oos_service or FinalOosEvaluationService(InMemoryCandidateSealStore())

    def _candidates(self, request: dict) -> list[dict]:
        supplied = list(request.get("candidates") or [])
        if supplied:
            return supplied
        if request.get("use_model") and self._model is not None:
            valid = self._model_candidates()
            if valid:
                return valid
        return self._seeds.load()

    def _model_candidates(self, feedback: dict | None = None, previous: list[dict] | None = None) -> list[dict]:
        if self._model is None:
            return []
        prompt = (
            "Generate JSON array of factor candidates. Each item must contain name, hypothesis and "
            "an RPN token array using only the documented stock_factor vocabulary."
        )
        if feedback:
            prompt += (
                " Improve the previous round using this structured feedback; do not repeat a formula or a "
                f"known failure. feedback={json.dumps(feedback, sort_keys=True)} "
                f"previous={json.dumps(previous or [], sort_keys=True)[:6000]}"
            )
        try:
            parsed = json.loads(self._model.complete(prompt, system="You are a quantitative factor researcher."))
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else parsed
        return [
            item
            for item in candidates
            if isinstance(item, dict)
            and item.get("rpn")
            and all(is_valid_token(str(token)) for token in item["rpn"])
        ]

    @staticmethod
    def _feedback(evaluated: list[dict]) -> dict:
        if not evaluated:
            return {"reason": "EMPTY_ROUND", "quality": 0.0}
        quality = max(float(item["preliminary"].get("fitness") or 0.0) for item in evaluated)
        duplicates = len({item["candidate"]["candidate_hash"] for item in evaluated}) != len(evaluated)
        return {"reason": "LOW_FITNESS" if quality <= 0 else "IMPROVING", "quality": quality, "duplicates": duplicates}

    @staticmethod
    def _recent_alpha(values: np.ndarray, close: np.ndarray, horizon: int) -> dict:
        # 只能在 Discovery 窗口内计算（§13.2：Recent Alpha 不得引用 Final OOS）；
        # 调用方必须在传入前截断面板。
        start = max(0, values.shape[1] - max(20, horizon * 4))
        metrics = evaluate_factor_range(values, close, start, values.shape[1] - horizon, horizon=horizon)
        return {
            "recent_rank_ic": metrics.get("rank_ic"),
            "recent_icir": metrics.get("icir"),
            "recent_topk_excess": metrics.get("topk_excess_annual_return"),
            "recent_hit_rate": metrics.get("positive_window_ratio"),
            "recent_coverage": metrics.get("coverage"),
            "recent_decay": compute_ic_decay(values[:, start:], close[:, start:], max_horizon=min(5, horizon)),
            "recent_turnover": compute_turnover(values[:, start:], top_k=max(5, int(values.shape[0] * 0.01))),
            "passed": bool(metrics.get("passed")) and float(metrics.get("rank_ic") or 0.0) > 0,
        }

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

    @staticmethod
    def _mutate(candidate: dict, feedback: dict, generation_round: int) -> dict | None:
        """Deterministic mutation fallback when an LLM proposer is unavailable."""
        rpn = list(candidate["rpn"])
        replacements = {"ts_mean_5": "ts_mean_10", "ts_mean_10": "ts_mean_20", "ts_delay_3": "ts_delay_5"}
        for index, token in enumerate(rpn):
            if token in replacements:
                rpn[index] = replacements[token]
                break
        else:
            if not rpn or rpn[-1] != "cs_rank":
                return None
            rpn.insert(-1, "neg")
        return {
            "name": f"{candidate.get('name', 'candidate')}_r{generation_round}",
            "hypothesis": f"mutation after {feedback['reason']}: {candidate.get('hypothesis', '')}",
            "rpn": rpn,
            "parent_candidate_hash": candidate["candidate_hash"],
            "generation_round": generation_round,
            "generation_strategy": "feedback_mutation",
            "generation_feedback": feedback,
        }

    @staticmethod
    def _correlation_deduplicate(evaluated: list[dict], threshold: float = 0.995) -> tuple[list[dict], list[dict]]:
        """Keep one representative from each near-identical VM output cluster."""
        retained: list[dict] = []
        rejected: list[dict] = []
        def fitness(value: dict) -> float:
            result = value["preliminary"].get("fitness")
            return float(result) if result is not None else float("-inf")

        for item in sorted(evaluated, key=fitness, reverse=True):
            values = np.asarray(item["values"], dtype=float).reshape(-1)
            duplicate_of = None
            for representative in retained:
                other = np.asarray(representative["values"], dtype=float).reshape(-1)
                valid = np.isfinite(values) & np.isfinite(other)
                if valid.sum() >= 10 and abs(float(np.corrcoef(values[valid], other[valid])[0, 1])) >= threshold:
                    duplicate_of = representative["candidate"]["candidate_hash"]
                    break
            if duplicate_of:
                rejected.append({"candidate_hash": item["candidate"]["candidate_hash"], "representative": duplicate_of})
            else:
                retained.append(item)
        return retained, rejected

    def run(self, request: dict, progress=lambda stage, value: None) -> dict:
        symbols = request.get("symbols") or []
        if not symbols:
            raise ValueError("symbols must not be empty")
        end = request.get("end") or datetime.now(UTC).date().isoformat()
        start = (
            request.get("start")
            or (datetime.fromisoformat(end) - timedelta(days=int(request.get("days") or 365) * 2)).date().isoformat()
        )
        progress("data", 10)
        snapshot = self._market.get_daily_bars(symbols, start, end, "qfq")
        signals = self._content.load_signals(symbols, start, end)
        panel = build_feature_panel(snapshot, signals)
        panel_feature_version = feature_set_version(list(panel.keys()))
        config = get_research_config()
        budget = max(1, min(int(request.get("candidate_budget") or 50), 200))
        candidates = self._canonical_candidates(self._candidates(request), budget)
        horizon = int(request.get("horizon") or config.evaluation.horizon_days)
        evaluated: list[dict] = []
        explicit_candidates = bool(request.get("candidates") or request.get("use_model"))
        round_limit = max(1, min(int(request.get("rounds") or (1 if explicit_candidates else 3)), 10))
        candidates_per_round = max(1, int(request.get("candidates_per_round") or len(candidates)))
        pending = candidates
        seen = {candidate["candidate_hash"] for candidate in candidates}
        search_rounds: list[dict] = []
        for generation_round in range(1, round_limit + 1):
            round_evaluated: list[dict] = []
            for candidate in pending:
                rpn = candidate["rpn"]
                max_lookback_from_rpn(rpn)
                values = StackVM().execute(rpn, panel)
                if values is None:
                    continue
                split = build_research_split(values.shape[1], config.data_split, horizon)
                freeze: CandidateFreeze | None = None
                if split is None:
                    preliminary = evaluate_factor(
                        values, panel["close"], horizon=horizon, eval_window=request.get("eval_window")
                    )
                    walkforward = {"passed": False, "reason": "INSUFFICIENT_RESEARCH_HISTORY"}
                    final_oos = {"passed": False, "reason": "INSUFFICIENT_RESEARCH_HISTORY"}
                else:
                    # §13.2：Candidate Search 只能看到 Discovery Snapshot。
                    # Preliminary Fitness 严格限制在 Discovery 窗口内。
                    eval_window = request.get("eval_window")
                    preliminary_start = (
                        max(split.discovery_start, split.discovery_end - int(eval_window))
                        if eval_window
                        else split.discovery_start
                    )
                    preliminary = evaluate_factor_range(
                        values, panel["close"], preliminary_start, split.discovery_end, horizon=horizon
                    )
                    walkforward = run_purged_walkforward(
                        values, panel["close"], split.discovery_start, split.discovery_end, horizon
                    )
                    # §13.3 Candidate Freeze → §13.4 独立一次性 Final OOS 评估。
                    refs = derive_snapshot_refs(
                        snapshot.data_snapshot_id,
                        (split.discovery_start, split.discovery_end),
                        (split.final_oos_start, split.final_oos_end),
                    )
                    freeze = CandidateFreeze(
                        candidate_hash=candidate["candidate_hash"],
                        formula=list(rpn),
                        dsl_version=DSL_VERSION,
                        feature_set_version=panel_feature_version,
                        discovery_snapshot_id=refs.discovery_snapshot_id,
                        final_oos_snapshot_id=refs.final_oos_snapshot_id,
                    )
                    self._oos_service.freeze_candidate(freeze)
                    try:
                        final_oos = self._oos_service.evaluate(
                            candidate["candidate_hash"], values, panel["close"], split, horizon
                        )
                    except OosWindowInvalidatedError:
                        final_oos = {"passed": False, "reason": "OOS_WINDOW_INVALIDATED"}
                discovery_values = values[:, : split.discovery_end] if split else values
                discovery_close = panel["close"][:, : split.discovery_end] if split else panel["close"]
                diagnostics, exposure, capacity = self._diagnostics(values, panel)
                item = {
                    "candidate": {**candidate, "generation_round": generation_round},
                    "values": values,
                    "preliminary": preliminary,
                    "walkforward": walkforward,
                    "final_oos": final_oos,
                    "diagnostics": diagnostics,
                    "exposure": exposure,
                    "capacity": capacity,
                    # §13.2：Recent Alpha 仅在 Discovery 窗口内计算。
                    "recent_alpha": self._recent_alpha(discovery_values, discovery_close, horizon),
                    "split": split,
                    "freeze": freeze.to_dict() if freeze else None,
                }
                evaluated.append(item)
                round_evaluated.append(item)
                progress("evaluate", 20 + int(len(evaluated) * 55 / max(budget, 1)))
            feedback = self._feedback(round_evaluated)
            search_rounds.append(
                {
                    "round": generation_round,
                    "candidate_count": len(pending),
                    "evaluated_count": len(round_evaluated),
                    "feedback": feedback,
                }
            )
            if generation_round == round_limit or len(evaluated) >= budget or not round_evaluated:
                break
            previous = [
                {
                    "candidate_hash": item["candidate"]["candidate_hash"],
                    "rpn": item["candidate"]["rpn"],
                    "fitness": item["preliminary"].get("fitness"),
                    "passed": item["preliminary"].get("passed"),
                }
                for item in round_evaluated
            ]
            model_proposed = self._model_candidates(feedback, previous) if request.get("use_model") else []
            proposed = model_proposed or [
                self._mutate(item["candidate"], feedback, generation_round + 1) for item in round_evaluated
            ]
            pending = self._canonical_candidates([item for item in proposed if item], budget - len(evaluated))[
                :candidates_per_round
            ]
            pending = [item for item in pending if item["candidate_hash"] not in seen]
            seen.update(item["candidate_hash"] for item in pending)
            if not pending:
                break

        evaluated, correlation_duplicates = self._correlation_deduplicate(evaluated)
        # §13.2：FDR / PBO / DSR 队列统计验证只能使用 Discovery 窗口的 rank-IC。
        cohort_statistics = validate_statistical_experiment(
            {
                item["candidate"]["candidate_hash"]: rank_ic_series(
                    item["values"][:, : item["split"].discovery_end] if item["split"] else item["values"],
                    panel["close"][:, : item["split"].discovery_end] if item["split"] else panel["close"],
                    horizon,
                )
                for item in evaluated
            }
        )
        accepted = []
        for index, item in enumerate(evaluated):
            candidate = item["candidate"]
            rpn = list(candidate["rpn"])
            if hashlib.sha256(" ".join(rpn).encode()).hexdigest() != candidate["candidate_hash"]:
                raise ResearchIntegrityError("candidate_hash does not match the candidate RPN")
            values = item["values"]
            preliminary = item["preliminary"]
            walkforward = item["walkforward"]
            final_oos = item["final_oos"]
            diagnostics = item["diagnostics"]
            exposure = item["exposure"]
            capacity = item["capacity"]
            recent_alpha = item["recent_alpha"]
            split = item["split"]
            freeze = item.get("freeze")
            statistics = cohort_statistics[candidate["candidate_hash"]]
            oos_audit = audit_final_oos(
                split=split,
                final_oos=final_oos,
                data_snapshot_id=snapshot.data_snapshot_id,
            )
            promotion = evaluate_promotion_gate(
                walkforward=walkforward,
                statistics=statistics,
                final_oos=final_oos,
                oos_audit=oos_audit,
                diagnostics=diagnostics,
                exposure=exposure,
                capacity=capacity,
                recent_alpha=recent_alpha,
                data_snapshot_id=snapshot.data_snapshot_id,
            ).model_dump()
            # Mining never activates a factor.  A passing research gate only
            # makes it eligible for the separate paper-trading approval flow.
            status = "PAPER_TRADING" if promotion["passed"] and final_oos.get("passed") else "CANDIDATE"
            metrics = {
                "in_sample": preliminary,
                "walkforward": walkforward,
                "final_oos": final_oos,
                "oos_audit": oos_audit,
                "statistics": statistics,
                "diagnostics": diagnostics,
                "exposure": exposure,
                "capacity": capacity,
                "recent_alpha": recent_alpha,
                "promotion_gate": promotion,
                "research_split": split.diagnostics(horizon, values.shape[1]) if split else None,
                "data_version": snapshot.data_version,
                "data_snapshot_id": snapshot.data_snapshot_id,
                # §86：Experiment 必须分别保存 discovery / final OOS 快照引用。
                "discovery_snapshot_id": freeze["discovery_snapshot_id"] if freeze else None,
                "final_oos_snapshot_id": freeze["final_oos_snapshot_id"] if freeze else None,
                "candidate_frozen_at": freeze["candidate_frozen_at"] if freeze else None,
                "dsl_version": DSL_VERSION,
                "feature_set_version": panel_feature_version,
                "candidate_search_window": "discovery_only",
                "generation_round": candidate.get("generation_round", 1),
                "parent_candidate_id": candidate.get("parent_candidate_hash"),
                "generation_strategy": candidate.get("generation_strategy", "seed"),
                "generation_feedback": candidate.get("generation_feedback") or {},
            }
            definition = FactorDefinition(
                factor_id=uuid4().hex,
                name=candidate.get("name") or candidate["candidate_hash"][:12],
                rpn=rpn,
                hypothesis=candidate.get("hypothesis", ""),
                status=status,
                metrics=metrics,
                candidate_hash=candidate["candidate_hash"],
            )
            accepted.append(self._factors.save(definition))
            progress("promotion", 75 + int((index + 1) * 20 / max(len(evaluated), 1)))
        return {
            "factor_count": len(accepted),
            "factors": accepted,
            "data_version": snapshot.data_version,
            "data_snapshot_id": snapshot.data_snapshot_id,
            "content_signal_count": len(signals),
            "search_rounds": search_rounds,
            "correlation_duplicates": correlation_duplicates,
        }
