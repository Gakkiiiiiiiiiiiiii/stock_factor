"""Validation gates for formal market snapshot references."""

from __future__ import annotations

from datetime import date, datetime, timezone

from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef


def _day(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def validate_formal_market_snapshot(
    ref: FormalMarketDatasetRef,
    *,
    requested_start: str | date | datetime,
    requested_end: str | date | datetime,
    as_of: datetime | None = None,
    expected_manifest_hash: str | None = None,
    expected_ref_hash: str | None = None,
) -> FormalMarketDatasetRef:
    """Fail closed when a formal snapshot cannot satisfy a command cutoff."""
    if not isinstance(ref, FormalMarketDatasetRef):
        raise TypeError("formal market snapshot ref is required")
    start, end = _day(requested_start), _day(requested_end)
    if start > end or start < _day(ref.start) or end > _day(ref.end):
        raise ValueError("requested market range is outside the formal snapshot")
    if expected_manifest_hash is not None and ref.manifest_hash != expected_manifest_hash:
        raise ValueError("market manifest hash mismatch")
    if expected_ref_hash is not None and ref.ref_hash != expected_ref_hash:
        raise ValueError("market snapshot ref hash mismatch")
    if as_of is not None:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        if ref.available_from > as_of.astimezone(timezone.utc):
            raise ValueError("market snapshot is not available at the requested cutoff")
    return ref


__all__ = ["validate_formal_market_snapshot"]
