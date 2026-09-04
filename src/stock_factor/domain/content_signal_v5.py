"""Strict, point-in-time content-factor-signal.v5.1 models."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("content timestamps must be timezone-aware")
    return value.astimezone(UTC)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FormalContentQuery(_StrictModel):
    contract: Literal["content-factor-signal.v5.1"]
    checksum: str
    content_snapshot_id: str
    business_as_of: datetime
    knowledge_as_of: datetime
    availability_as_of: datetime
    pit_mode: Literal["PUBLIC_STRICT"]
    signal_policy_version: str
    min_support: StrictInt = Field(ge=1)
    producer_commit: str
    allow_proxy: bool = False

    _timestamps = field_validator("business_as_of", "knowledge_as_of", "availability_as_of")(_utc)

    @field_validator("checksum", "content_snapshot_id", "signal_policy_version", "producer_commit")
    @classmethod
    def _required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content query identity fields cannot be empty")
        return value


class ContentSignalV5(_StrictModel):
    signal_id: str
    symbol: str
    subject_key: str
    occurrence_at: datetime
    business_as_of: datetime
    knowledge_projection_at: datetime
    source_available_at: datetime
    available_from: datetime
    source_availability_quality: Literal["VERIFIED", "PROXY", "UNKNOWN"]
    lifecycle_as_of: datetime
    sentiment: str
    knowledge_kind: str
    truth_status: str
    source_video_id: str
    support_count: StrictInt = Field(ge=0)
    producer_commit: str
    content_attention_score: float | None = None
    cross_video_consensus: float | None = None
    cross_video_disagreement: float | None = None

    _timestamps = field_validator(
        "occurrence_at",
        "business_as_of",
        "knowledge_projection_at",
        "source_available_at",
        "available_from",
        "lifecycle_as_of",
    )(_utc)

    @field_validator(
        "signal_id",
        "symbol",
        "subject_key",
        "sentiment",
        "knowledge_kind",
        "truth_status",
        "source_video_id",
        "producer_commit",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content signal identity/semantic fields cannot be empty")
        return value


class FormalContentRef(_StrictModel):
    contract: Literal["content-factor-signal.v5.1"]
    checksum: str
    content_snapshot_id: str
    business_as_of: datetime
    knowledge_as_of: datetime
    availability_as_of: datetime
    pit_mode: Literal["PUBLIC_STRICT"]
    signal_policy_version: str
    min_support: StrictInt = Field(ge=1)
    producer_commit: str
    allow_proxy: bool = False
    manifest_hash: str
    ref_hash: str

    _timestamps = field_validator("business_as_of", "knowledge_as_of", "availability_as_of")(_utc)

    @field_validator(
        "checksum", "content_snapshot_id", "signal_policy_version", "producer_commit", "manifest_hash", "ref_hash"
    )
    @classmethod
    def _required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content ref identity fields cannot be empty")
        return value

    @classmethod
    def from_query(cls, query: FormalContentQuery, manifest_hash: str) -> "FormalContentRef":
        material = {**query.model_dump(mode="json"), "manifest_hash": manifest_hash}
        ref_hash = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(**material, ref_hash=ref_hash)


__all__ = ["ContentSignalV5", "FormalContentQuery", "FormalContentRef"]
