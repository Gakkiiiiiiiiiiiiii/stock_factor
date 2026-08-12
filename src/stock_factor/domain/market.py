from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MarketDataSnapshot:
    symbols: list[str]
    dates: list[str]
    bars: dict[str, Any]
    data_version: str
    data_snapshot_id: str
    source: str
