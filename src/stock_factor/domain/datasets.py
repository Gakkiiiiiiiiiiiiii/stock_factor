"""Research Dataset Ref（详细修改方案 P0-3）。

Discovery 与 Final OOS 必须是有独立物理身份的数据集对象，
不能只依赖 split.final_oos_start/end 这样的区间参数。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from stock_factor.domain.content_signal_v5 import FormalContentRef
from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef


def _dataset_hash(material: dict) -> str:
    return hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class DiscoveryDatasetRef:
    """候选搜索数据集身份。"""

    market_snapshot_id: str
    source_data_version: str
    universe_snapshot_id: str
    feature_schema_version: str
    start: str
    end: str
    warmup_start: str | None = None
    formal_market_ref: FormalMarketDatasetRef | None = None
    formal_content_ref: FormalContentRef | None = None
    dataset_hash: str = field(init=False, default="")
    market_ref_hash: str | None = field(init=False, default=None)
    content_ref_hash: str | None = field(init=False, default=None)
    formal_eligible: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.formal_market_ref is not None and self.formal_market_ref.market_snapshot_id != self.market_snapshot_id:
            raise ValueError("dataset market_snapshot_id does not match formal market ref")
        if self.formal_content_ref is not None and not self.formal_content_ref.content_snapshot_id:
            raise ValueError("dataset requires a non-empty formal content snapshot identity")
        object.__setattr__(self, "market_ref_hash", self.formal_market_ref.ref_hash if self.formal_market_ref else None)
        object.__setattr__(
            self, "content_ref_hash", self.formal_content_ref.ref_hash if self.formal_content_ref else None
        )
        object.__setattr__(
            self, "formal_eligible", self.formal_market_ref is not None and self.formal_content_ref is not None
        )
        market_identity = asdict(self.formal_market_ref) if self.formal_market_ref else None
        if market_identity is not None:
            market_identity["available_from"] = self.formal_market_ref.available_from.isoformat()
        content_identity = self.formal_content_ref.model_dump(mode="json") if self.formal_content_ref else None
        material = {
            "scope": "DISCOVERY",
            **{
                k: v
                for k, v in asdict(self).items()
                if k
                not in {
                    "dataset_hash",
                    "market_ref_hash",
                    "content_ref_hash",
                    "formal_eligible",
                    "formal_market_ref",
                    "formal_content_ref",
                }
            },
            "market_ref": market_identity,
            "content_ref": content_identity,
            "market_ref_hash": self.market_ref_hash,
            "content_ref_hash": self.content_ref_hash,
        }
        object.__setattr__(self, "dataset_hash", _dataset_hash(material))


@dataclass(frozen=True)
class FinalOosDatasetRef:
    """Final OOS 数据集身份：与 Discovery 物理身份必须不同。"""

    market_snapshot_id: str
    source_data_version: str
    universe_snapshot_id: str
    feature_schema_version: str
    start: str
    end: str
    warmup_start: str | None = None
    formal_market_ref: FormalMarketDatasetRef | None = None
    formal_content_ref: FormalContentRef | None = None
    dataset_hash: str = field(init=False, default="")
    market_ref_hash: str | None = field(init=False, default=None)
    content_ref_hash: str | None = field(init=False, default=None)
    formal_eligible: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.formal_market_ref is not None and self.formal_market_ref.market_snapshot_id != self.market_snapshot_id:
            raise ValueError("dataset market_snapshot_id does not match formal market ref")
        if self.formal_content_ref is not None and not self.formal_content_ref.content_snapshot_id:
            raise ValueError("dataset requires a non-empty formal content snapshot identity")
        object.__setattr__(self, "market_ref_hash", self.formal_market_ref.ref_hash if self.formal_market_ref else None)
        object.__setattr__(
            self, "content_ref_hash", self.formal_content_ref.ref_hash if self.formal_content_ref else None
        )
        object.__setattr__(
            self, "formal_eligible", self.formal_market_ref is not None and self.formal_content_ref is not None
        )
        market_identity = asdict(self.formal_market_ref) if self.formal_market_ref else None
        if market_identity is not None:
            market_identity["available_from"] = self.formal_market_ref.available_from.isoformat()
        content_identity = self.formal_content_ref.model_dump(mode="json") if self.formal_content_ref else None
        material = {
            "scope": "FINAL_OOS",
            **{
                k: v
                for k, v in asdict(self).items()
                if k
                not in {
                    "dataset_hash",
                    "market_ref_hash",
                    "content_ref_hash",
                    "formal_eligible",
                    "formal_market_ref",
                    "formal_content_ref",
                }
            },
            "market_ref": market_identity,
            "content_ref": content_identity,
            "market_ref_hash": self.market_ref_hash,
            "content_ref_hash": self.content_ref_hash,
        }
        object.__setattr__(self, "dataset_hash", _dataset_hash(material))

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.formal_market_ref is not None:
            payload["formal_market_ref"] = asdict(self.formal_market_ref)
            payload["formal_market_ref"]["available_from"] = self.formal_market_ref.available_from.isoformat()
        if self.formal_content_ref is not None:
            payload["formal_content_ref"] = self.formal_content_ref.model_dump(mode="json")
        return payload


def assert_disjoint(discovery: DiscoveryDatasetRef, final_oos: FinalOosDatasetRef) -> None:
    """P0-3：Final OOS 数据 ≠ Candidate Search 数据（物理身份层面）。"""
    if discovery.dataset_hash == final_oos.dataset_hash:
        raise ValueError("Discovery 与 Final OOS 数据集身份相同，违反 P0-3 隔离要求")


__all__ = ["DiscoveryDatasetRef", "FinalOosDatasetRef", "assert_disjoint"]
