from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef


@dataclass(frozen=True)
class MarketDataSnapshot:
    symbols: list[str]
    dates: list[str]
    bars: dict[str, Any]
    data_version: str
    data_snapshot_id: str
    source: str
    formal_market_ref: FormalMarketDatasetRef | None = None
    formal_eligible: bool = False

    def __post_init__(self) -> None:
        if self.formal_market_ref is not None:
            if self.formal_market_ref.market_snapshot_id != self.data_snapshot_id:
                raise ValueError("market snapshot identity does not match formal market ref")
            object.__setattr__(self, "formal_eligible", True)
