from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FactorDefinition:
    factor_id: str
    name: str
    rpn: list[str]
    hypothesis: str = ""
    status: str = "CANDIDATE"
    version: int = 1
    metrics: dict[str, Any] = field(default_factory=dict)
    candidate_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FactorJob:
    job_id: str
    status: str = "PENDING"
    stage: str = "queued"
    progress: int = 0
    retry_count: int = 0
    max_retries: int = 3
    request: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
