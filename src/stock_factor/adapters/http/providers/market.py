from __future__ import annotations

import os
from uuid import uuid4

import httpx

from stock_factor.domain.market import MarketDataSnapshot
from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef


def _trace_headers() -> dict[str, str]:
    # §32：统一 Trace Headers，factor 出站调用透传 trace 与调用方标识。
    return {"X-Trace-Id": uuid4().hex, "X-Caller-Service": "stock_factor"}


def _decode_market_response(
    body: dict,
    *,
    formal: bool,
    expected_ref: FormalMarketDatasetRef | None = None,
    requested_start: str | None = None,
    requested_end: str | None = None,
) -> MarketDataSnapshot:
    expected = "market-snapshot.v1" if formal else {None, "market-data.v1"}
    if formal and body.get("contract_version") != expected:
        raise ValueError(f"unsupported formal market contract: {body.get('contract_version')}")
    if not formal and body.get("contract_version") not in expected:
        raise ValueError(f"unsupported exploratory market contract: {body.get('contract_version')}")
    payload = body.get("data", body)
    if not isinstance(payload, dict):
        raise ValueError("market data response payload must be an object")
    required = {"symbols", "dates", "bars", "data_version", "data_snapshot_id"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"market data response misses required fields: {sorted(missing)}")
    ref = None
    if formal:
        ref_payload = payload.get("market_snapshot_ref")
        if not isinstance(ref_payload, dict):
            raise ValueError("formal market response requires market_snapshot_ref")
        ref = FormalMarketDatasetRef.from_payload(ref_payload)
        if ref.market_snapshot_id != payload["data_snapshot_id"]:
            raise ValueError("market snapshot id does not match formal ref")
        if expected_ref is None:
            raise ValueError("formal market request requires a bound market snapshot ref")
        if ref.ref_hash != expected_ref.ref_hash or ref.manifest_hash != expected_ref.manifest_hash:
            raise ValueError("market snapshot response does not match requested ref")
        if ref.market_snapshot_id != expected_ref.market_snapshot_id:
            raise ValueError("market snapshot response identity does not match requested ref")
        if requested_start is not None and requested_end is not None:
            from stock_factor.application.market_snapshot_validator import validate_formal_market_snapshot

            validate_formal_market_snapshot(
                ref, requested_start=requested_start, requested_end=requested_end, as_of=ref.available_from
            )
    return MarketDataSnapshot(
        symbols=payload["symbols"],
        dates=payload["dates"],
        bars=payload["bars"],
        data_version=payload["data_version"],
        data_snapshot_id=payload["data_snapshot_id"],
        source=payload.get("source", "quant"),
        formal_market_ref=ref,
        formal_eligible=formal,
    )


class HttpMarketDataProvider:
    """Formal market-snapshot.v1 consumer with no legacy fallback."""

    def __init__(self, base_url: str | None = None, timeout: float = 30) -> None:
        self._url = (base_url or os.getenv("MARKET_DATA_SERVICE_URL", "http://quant:8011")).rstrip("/")
        self._timeout = timeout

    def get_daily_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        adjust: str = "qfq",
        *,
        formal_market_ref: FormalMarketDatasetRef | None = None,
        expected_ref: FormalMarketDatasetRef | None = None,
        as_of=None,
    ) -> MarketDataSnapshot:
        bound_ref = formal_market_ref or expected_ref
        if bound_ref is None:
            raise ValueError("formal market provider requires formal_market_ref")
        payload = {"symbols": symbols, "start": start, "end": end, "adjust": adjust}
        payload.update(
            {
                "market_snapshot_id": bound_ref.market_snapshot_id,
                "manifest_hash": bound_ref.manifest_hash,
                "ref_hash": bound_ref.ref_hash,
            }
        )
        response = httpx.post(
            f"{self._url}/api/v1/market/bars/batch", json=payload, headers=_trace_headers(), timeout=self._timeout
        )
        response.raise_for_status()
        snapshot = _decode_market_response(
            response.json(), formal=True, expected_ref=bound_ref, requested_start=start, requested_end=end
        )
        if as_of is not None and snapshot.formal_market_ref.available_from > as_of:
            raise ValueError("market snapshot is not available at the requested cutoff")
        return snapshot


class ExploratoryMarketDataProvider:
    """Legacy/current market endpoint compatibility; never formal-eligible."""

    def __init__(self, base_url: str | None = None, timeout: float = 30) -> None:
        self._url = (base_url or os.getenv("MARKET_DATA_SERVICE_URL", "http://quant:8011")).rstrip("/")
        self._timeout = timeout

    def get_daily_bars(self, symbols: list[str], start: str, end: str, adjust: str = "qfq") -> MarketDataSnapshot:
        payload = {"symbols": symbols, "start": start, "end": end, "adjust": adjust}
        try:
            response = httpx.post(
                f"{self._url}/api/v1/market/bars/batch", json=payload, headers=_trace_headers(), timeout=self._timeout
            )
            if response.status_code == 404:
                raise httpx.HTTPStatusError("legacy fallback", request=response.request, response=response)
        except httpx.HTTPError:
            response = httpx.post(
                f"{self._url}/v1/bars/batch", json=payload, headers=_trace_headers(), timeout=self._timeout
            )
        response.raise_for_status()
        return _decode_market_response(response.json(), formal=False)


LegacyMarketDataProvider = ExploratoryMarketDataProvider


class MarketDataProviderRouter:
    """Route explicitly bound formal reads to the strict provider.

    Unbound reads remain exploratory for local development compatibility; a
    formal ref can never fall through to that provider.
    """

    def __init__(self, formal=None, exploratory=None) -> None:
        from stock_factor.adapters.http.providers.market import HttpMarketDataProvider

        self._formal = formal or HttpMarketDataProvider()
        self._exploratory = exploratory or ExploratoryMarketDataProvider()

    def get_daily_bars(
        self, symbols, start, end, adjust="qfq", *, formal_market_ref=None, expected_ref=None, as_of=None
    ):
        if formal_market_ref is not None or expected_ref is not None:
            return self._formal.get_daily_bars(
                symbols,
                start,
                end,
                adjust,
                formal_market_ref=formal_market_ref,
                expected_ref=expected_ref,
                as_of=as_of,
            )
        return self._exploratory.get_daily_bars(symbols, start, end, adjust)
