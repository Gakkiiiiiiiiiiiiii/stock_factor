"""Finalist Selection（收尾文档 §18 / §25）。

Discovery 完成后、Final OOS 之前选择唯一（或少量 pre-registered）finalist。
排序只允许使用 Discovery 证据（gate / fitness / walkforward），
保证“只修改 OOS Dataset 不会改变 Finalist”（§25 / §79）。
"""
from __future__ import annotations

MAX_FINALIST_COUNT = 3
DEFAULT_FINALIST_COUNT = 1


def select_finalists(
    evaluated: list[dict],
    discovery_gates: dict[str, dict],
    finalist_count: int = DEFAULT_FINALIST_COUNT,
) -> list[dict]:
    if finalist_count < 1:
        finalist_count = DEFAULT_FINALIST_COUNT
    if finalist_count > MAX_FINALIST_COUNT:
        # §18：cohort 最多 3 个，且必须 pre-registered（在实验配置中固定）。
        finalist_count = MAX_FINALIST_COUNT

    def sort_key(item: dict):
        candidate_hash = item["candidate"]["candidate_hash"]
        gate = discovery_gates.get(candidate_hash) or {}
        fitness = item["preliminary"].get("fitness")
        fitness = float(fitness) if fitness is not None else float("-inf")
        window_pass_ratio = float((item["walkforward"] or {}).get("window_pass_ratio", 0.0))
        return (bool(gate.get("passed")), fitness, window_pass_ratio, candidate_hash)

    ordered = sorted(evaluated, key=sort_key, reverse=True)
    finalists = ordered[:finalist_count]
    for rank, item in enumerate(finalists, start=1):
        item["selection_rank"] = rank
    return finalists


__all__ = ["select_finalists", "MAX_FINALIST_COUNT", "DEFAULT_FINALIST_COUNT"]
