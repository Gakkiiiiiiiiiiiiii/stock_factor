"""Audit the research process before an OOS result can be promoted."""

from __future__ import annotations


def audit_final_oos(
    *,
    split: object | None,
    final_oos: dict | None,
    data_snapshot_id: str | None,
) -> dict:
    """Return an immutable, deterministic OOS audit record.

    The evaluator supplies performance; this function only verifies that the
    final holdout exists, follows the discovery period, and is tied to a
    reproducible data snapshot.
    """
    violations: list[str] = []
    warnings: list[str] = []
    if split is None:
        violations.append("RESEARCH_SPLIT_MISSING")
    else:
        discovery_end = getattr(split, "discovery_end", None)
        final_start = getattr(split, "final_oos_start", None)
        final_end = getattr(split, "final_oos_end", None)
        if final_start is None or final_end is None or final_start >= final_end:
            violations.append("FINAL_OOS_WINDOW_INVALID")
        if discovery_end is not None and final_start is not None and final_start < discovery_end:
            violations.append("FINAL_OOS_OVERLAPS_DISCOVERY")
    if not (final_oos or {}).get("passed"):
        warnings.append("FINAL_OOS_METRICS_NOT_PASSED")
    if not data_snapshot_id:
        violations.append("DATA_SNAPSHOT_MISSING")
    return {
        "audit_status": "PASSED" if not violations else "FAILED",
        "violations": violations,
        "warnings": warnings,
        "audit_version": "final_oos_audit_v1",
        "data_snapshot_id": data_snapshot_id,
    }
