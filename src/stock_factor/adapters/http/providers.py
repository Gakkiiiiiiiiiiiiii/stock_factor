from __future__ import annotations

import os

import httpx

from stock_factor.domain.market import MarketDataSnapshot


class HttpMarketDataProvider:
    def __init__(self, base_url: str | None = None, timeout: float = 30) -> None:
        self._url = (base_url or os.getenv("MARKET_DATA_SERVICE_URL", "http://localhost:8001")).rstrip("/")
        self._timeout = timeout

    def get_daily_bars(self, symbols: list[str], start: str, end: str, adjust: str = "qfq") -> MarketDataSnapshot:
        response = httpx.post(f"{self._url}/v1/bars/batch", json={"symbols": symbols, "start": start, "end": end, "adjust": adjust}, timeout=self._timeout)
        response.raise_for_status()
        payload = response.json().get("data", response.json())
        return MarketDataSnapshot(symbols=payload["symbols"], dates=payload["dates"], bars=payload["bars"], data_version=payload["data_version"], data_snapshot_id=payload["data_snapshot_id"], source=payload.get("source", "market-data-service"))


class HttpContentSignalProvider:
    def __init__(self, base_url: str | None = None, timeout: float = 30) -> None:
        self._url = (base_url or os.getenv("CONTENT_SERVICE_URL", "http://localhost:8100")).rstrip("/")
        self._timeout = timeout

    def load_signals(self, symbols: list[str], start: str, end: str) -> list[dict]:
        response = httpx.post(f"{self._url}/internal/v1/factor-signals", json={"symbols": symbols, "start": start, "end": end, "minimum_support_status": "SOURCE_SUPPORTED"}, timeout=self._timeout)
        response.raise_for_status()
        return response.json().get("items", [])


class HttpModelClient:
    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self._url = (url or os.getenv("FACTOR_MODEL_URL", "")).rstrip("/")
        self._model = model or os.getenv("FACTOR_MODEL_NAME", "")

    def complete(self, prompt: str, system: str | None = None, temperature: float | None = None) -> str:
        if not self._url:
            raise RuntimeError("FACTOR_MODEL_URL is not configured")
        response = httpx.post(self._url, json={"model": self._model, "prompt": prompt, "system": system, "temperature": temperature}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("text") or payload.get("output") or payload.get("content") or "")
