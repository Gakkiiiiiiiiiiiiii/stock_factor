"""Exploratory legacy content provider and explicit provider router."""

from __future__ import annotations

import os

import httpx

from stock_factor.adapters.http.providers.market import _trace_headers


class ExploratoryContentSignalProvider:
    """Deprecated v1-v3 compatibility reader; never formal-eligible."""

    def __init__(self, base_url: str | None = None, timeout: float = 30) -> None:
        self._url = (base_url or os.getenv("CONTENT_SERVICE_URL", "http://localhost:8100")).rstrip("/")
        self._timeout = timeout

    def load_signals(self, symbols: list[str], start: str, end: str) -> list[dict]:
        response = httpx.post(
            f"{self._url}/internal/v1/factor-signals",
            json={"symbols": symbols, "start": start, "end": end, "minimum_support_status": "SOURCE_SUPPORTED"},
            headers=_trace_headers(),
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("contract_version") not in {
            None,
            "content-factor-signal.v1",
            "content-factor-signal.v2",
            "content-factor-signal.v3",
        }:
            raise ValueError(f"unsupported content signal contract: {body['contract_version']}")
        return [{**item, "formal_eligible": False} for item in body.get("items", [])]


LegacyContentSignalProvider = ExploratoryContentSignalProvider


class ContentSignalProviderRouter:
    """Use strict v5.1 only when a formal query/ref is explicitly supplied."""

    def __init__(self, formal=None, exploratory=None) -> None:
        from stock_factor.adapters.http.content_v5_provider import FormalContentSignalProviderV5

        self._formal = formal or FormalContentSignalProviderV5()
        self._exploratory = exploratory or ExploratoryContentSignalProvider()

    def load_signals(self, symbols, start, end, *, query=None, expected_ref=None):
        if query is not None or expected_ref is not None:
            return self._formal.load_signals(symbols, start, end, query=query, expected_ref=expected_ref)
        return self._exploratory.load_signals(symbols, start, end)


__all__ = ["ContentSignalProviderRouter", "ExploratoryContentSignalProvider", "LegacyContentSignalProvider"]
