"""Formal HTTP boundary for content-factor-signal.v5.1."""

from __future__ import annotations

import os
from uuid import uuid4

import httpx

from stock_factor.adapters.http.providers.market import _trace_headers
from stock_factor.application.content_signal_eligibility import evaluate_content_eligibility
from stock_factor.domain.content_signal_v5 import ContentSignalV5, FormalContentQuery, FormalContentRef


class FormalContentSignalProviderV5:
    def __init__(self, base_url: str | None = None, timeout: float = 30) -> None:
        self._url = (base_url or os.getenv("CONTENT_SERVICE_URL", "http://localhost:8100")).rstrip("/")
        self._timeout = timeout
        self._required_checksum = os.getenv("FACTOR_REQUIRED_CONTENT_CHECKSUM") or None

    def load_signals(
        self,
        symbols: list[str],
        start: str,
        end: str,
        *,
        query: FormalContentQuery | None = None,
        expected_ref: FormalContentRef | None = None,
    ) -> list[dict]:
        if query is None:
            raise ValueError("formal content provider requires a v5.1 query")
        if expected_ref is None:
            raise ValueError("formal content provider requires expected_ref")
        if self._required_checksum is not None and query.checksum != self._required_checksum:
            raise ValueError("content query checksum does not match configured capability")
        payload = {
            "request_id": uuid4().hex,
            "symbols": symbols,
            "start": start,
            "end": end,
            **query.model_dump(mode="json"),
        }
        response = httpx.post(
            f"{self._url}/internal/v2/factor-signals/query",
            json=payload,
            headers=_trace_headers(),
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or body.get("contract_version") != query.contract:
            raise ValueError("unsupported content signal contract")
        if set(body) != {"contract_version", "data"}:
            raise ValueError("content signal response has unexpected top-level fields")
        data = body.get("data", body)
        if not isinstance(data, dict):
            raise ValueError("content signal response payload must be an object")
        for key in (
            "contract",
            "checksum",
            "content_snapshot_id",
            "business_as_of",
            "knowledge_as_of",
            "availability_as_of",
            "pit_mode",
            "signal_policy_version",
            "min_support",
            "producer_commit",
            "allow_proxy",
            "manifest_hash",
            "items",
        ):
            if key not in data:
                raise ValueError(f"content signal response misses {key}")
        unknown = set(data) - {
            "contract",
            "checksum",
            "content_snapshot_id",
            "business_as_of",
            "knowledge_as_of",
            "availability_as_of",
            "pit_mode",
            "signal_policy_version",
            "min_support",
            "producer_commit",
            "allow_proxy",
            "manifest_hash",
            "items",
        }
        if unknown:
            raise ValueError(f"content signal response has unexpected fields: {sorted(unknown)}")
        response_query = FormalContentQuery(
            contract=data["contract"],
            checksum=data["checksum"],
            content_snapshot_id=data["content_snapshot_id"],
            business_as_of=data["business_as_of"],
            knowledge_as_of=data["knowledge_as_of"],
            availability_as_of=data["availability_as_of"],
            pit_mode=data["pit_mode"],
            signal_policy_version=data["signal_policy_version"],
            min_support=data["min_support"],
            producer_commit=data["producer_commit"],
            allow_proxy=data["allow_proxy"],
        )
        if response_query != query:
            raise ValueError("content signal response identity or cutoff mismatch")
        ref = FormalContentRef.from_query(response_query, data["manifest_hash"])
        if ref != expected_ref:
            raise ValueError("content signal response ref mismatch")
        items = data.get("items")
        if not isinstance(items, list):
            raise ValueError("content signal response requires items")
        parsed = [ContentSignalV5.model_validate(item) for item in items]
        eligibility = evaluate_content_eligibility(parsed, query, allow_proxy=query.allow_proxy)
        if eligibility.rejected:
            raise ValueError({"code": "CONTENT_NOT_ELIGIBLE", "rejected": list(eligibility.rejected)})
        return [item.model_dump(mode="json") for item in eligibility.accepted]


__all__ = ["FormalContentSignalProviderV5"]
