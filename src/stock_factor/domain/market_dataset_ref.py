"""Immutable identity for a formal Quant market snapshot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime


def _instant(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("available_from must be timezone-aware")
    return parsed.astimezone(UTC)


def _day(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("market range datetime must be timezone-aware")
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid market range date: {value!r}") from exc


@dataclass(frozen=True)
class FormalMarketDatasetRef:
    """Content-addressed, quality-sealed market data identity."""

    market_snapshot_id: str
    manifest_hash: str
    calendar_version: str
    universe_version: str
    corporate_action_version: str
    tradability_version: str
    available_from: datetime | str
    start: date | datetime | str
    end: date | datetime | str
    quality_seal: str = "PASS"
    contract: str = "market-snapshot.v1"
    ref_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if self.contract != "market-snapshot.v1":
            raise ValueError("formal market ref requires contract market-snapshot.v1")
        for name in (
            "market_snapshot_id",
            "manifest_hash",
            "calendar_version",
            "universe_version",
            "corporate_action_version",
            "tradability_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"formal market ref requires non-empty {name}")
        if self.quality_seal != "PASS":
            raise ValueError("formal market ref requires quality_seal=PASS")
        available = _instant(self.available_from)
        start = _day(self.start)
        end = _day(self.end)
        if start > end:
            raise ValueError("formal market ref range is reversed")
        material = {
            "contract": self.contract,
            "market_snapshot_id": self.market_snapshot_id,
            "manifest_hash": self.manifest_hash,
            "calendar_version": self.calendar_version,
            "universe_version": self.universe_version,
            "corporate_action_version": self.corporate_action_version,
            "tradability_version": self.tradability_version,
            "quality_seal": self.quality_seal,
            "available_from": available.isoformat(),
            "start": start,
            "end": end,
        }
        object.__setattr__(self, "available_from", available)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(
            self,
            "ref_hash",
            hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        )

    @classmethod
    def from_payload(cls, payload: dict) -> "FormalMarketDatasetRef":
        supplied_hash = payload.get("ref_hash")
        ref = cls(**{key: value for key, value in payload.items() if key != "ref_hash"})
        if supplied_hash is not None and supplied_hash != ref.ref_hash:
            raise ValueError("formal market ref hash mismatch")
        return ref


__all__ = ["FormalMarketDatasetRef"]
