"""Deterministic candidate screening and deduplication."""

from __future__ import annotations

import hashlib

import numpy as np

from stock_factor.engine.vocab import is_valid_token


def canonical_candidates(candidates: list[dict], budget: int) -> list[dict]:
    """Apply a deterministic candidate budget before any evaluation."""
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


def correlation_deduplicate(evaluated: list[dict], threshold: float = 0.995) -> tuple[list[dict], list[dict]]:
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


def feedback(evaluated: list[dict]) -> dict:
    if not evaluated:
        return {"reason": "EMPTY_ROUND", "quality": 0.0}
    quality = max(float(item["preliminary"].get("fitness") or 0.0) for item in evaluated)
    duplicates = len({item["candidate"]["candidate_hash"] for item in evaluated}) != len(evaluated)
    return {"reason": "LOW_FITNESS" if quality <= 0 else "IMPROVING", "quality": quality, "duplicates": duplicates}


__all__ = ["canonical_candidates", "correlation_deduplicate", "feedback"]
