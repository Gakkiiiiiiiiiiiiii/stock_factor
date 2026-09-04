"""Discovery-only orchestration for candidate evaluation.

This module deliberately stops at discovery evidence.  Final OOS loading,
authorization, and sealing live in the OOS/artifact use cases.
"""

from __future__ import annotations

from typing import Any

from stock_factor.engine.fitness import evaluate_factor, evaluate_factor_range
from stock_factor.engine.lookback import max_lookback_from_rpn
from stock_factor.engine.purged_walkforward import run_purged_walkforward
from stock_factor.engine.research_split import build_research_split
from stock_factor.engine.vm import StackVM


def evaluate_discovery(
    *,
    service: Any,
    candidates: list[dict],
    panel: dict,
    config: Any,
    request: dict,
    budget: int,
    horizon: int,
    round_limit: int,
    candidates_per_round: int,
    progress: Any,
) -> tuple[list[dict], list[dict]]:
    """Evaluate the bounded discovery search and return evidence plus rounds."""
    evaluated: list[dict] = []
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
            if split is None:
                preliminary = evaluate_factor(
                    values, panel["close"], horizon=horizon, eval_window=request.get("eval_window")
                )
                walkforward = {"passed": False, "reason": "INSUFFICIENT_RESEARCH_HISTORY"}
            else:
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
            discovery_values = values[:, : split.discovery_end] if split else values
            discovery_close = panel["close"][:, : split.discovery_end] if split else panel["close"]
            diagnostics, exposure, capacity = service._diagnostics(values, panel)
            item = {
                "candidate": {**candidate, "generation_round": generation_round},
                "values": values,
                "preliminary": preliminary,
                "walkforward": walkforward,
                "final_oos": None,
                "diagnostics": diagnostics,
                "exposure": exposure,
                "capacity": capacity,
                "recent_alpha": service._recent_alpha(discovery_values, discovery_close, horizon),
                "split": split,
                "freeze": None,
            }
            evaluated.append(item)
            round_evaluated.append(item)
            progress("evaluate", 20 + int(len(evaluated) * 55 / max(budget, 1)))

        round_feedback = service._feedback(round_evaluated)
        search_rounds.append(
            {
                "round": generation_round,
                "candidate_count": len(pending),
                "evaluated_count": len(round_evaluated),
                "feedback": round_feedback,
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
        model_proposed = service._model_candidates(round_feedback, previous) if request.get("use_model") else []
        proposed = model_proposed or [
            service._mutate(item["candidate"], round_feedback, generation_round + 1) for item in round_evaluated
        ]
        pending = service._canonical_candidates([item for item in proposed if item], budget - len(evaluated))[
            :candidates_per_round
        ]
        pending = [item for item in pending if item["candidate_hash"] not in seen]
        seen.update(item["candidate_hash"] for item in pending)
        if not pending:
            break
    return evaluated, search_rounds
