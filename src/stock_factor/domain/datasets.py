"""Research Dataset Ref（详细修改方案 P0-3）。

Discovery 与 Final OOS 必须是有独立物理身份的数据集对象，
不能只依赖 split.final_oos_start/end 这样的区间参数。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


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
    dataset_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        material = {"scope": "DISCOVERY", **{k: v for k, v in asdict(self).items() if k != "dataset_hash"}}
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
    dataset_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        material = {"scope": "FINAL_OOS", **{k: v for k, v in asdict(self).items() if k != "dataset_hash"}}
        object.__setattr__(self, "dataset_hash", _dataset_hash(material))


def assert_disjoint(discovery: DiscoveryDatasetRef, final_oos: FinalOosDatasetRef) -> None:
    """P0-3：Final OOS 数据 ≠ Candidate Search 数据（物理身份层面）。"""
    if discovery.dataset_hash == final_oos.dataset_hash:
        raise ValueError("Discovery 与 Final OOS 数据集身份相同，违反 P0-3 隔离要求")


__all__ = ["DiscoveryDatasetRef", "FinalOosDatasetRef", "assert_disjoint"]
