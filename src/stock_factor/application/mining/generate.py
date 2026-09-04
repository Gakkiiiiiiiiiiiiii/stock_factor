"""Candidate generation responsibilities for factor mining.

This module owns proposal selection and deterministic mutation.  The service
module supplies the orchestration and persists only screened candidates.
"""

from __future__ import annotations

import json
from typing import Any

from stock_factor.engine.vocab import is_valid_token


def model_candidates(model: Any, feedback: dict | None = None, previous: list[dict] | None = None) -> list[dict]:
    if model is None:
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
        parsed = json.loads(model.complete(prompt, system="You are a quantitative factor researcher."))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else parsed
    return [
        item
        for item in candidates
        if isinstance(item, dict) and item.get("rpn") and all(is_valid_token(str(token)) for token in item["rpn"])
    ]


def generate_candidates(request: dict, *, seeds: Any, model: Any = None) -> list[dict]:
    supplied = list(request.get("candidates") or [])
    if supplied:
        return supplied
    if request.get("use_model") and model is not None:
        valid = model_candidates(model)
        if valid:
            return valid
    return seeds.load()


def mutate_candidate(candidate: dict, feedback: dict, generation_round: int) -> dict | None:
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


__all__ = ["generate_candidates", "model_candidates", "mutate_candidate"]
