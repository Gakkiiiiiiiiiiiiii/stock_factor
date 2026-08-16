"""Factor Paper API 的 Quant proxy（收尾文档 §26 / §27）。

Paper Trading Authority 属于 quant trading.v1；Factor 的兼容端点内部转发到
Quant，不再维护本地账户状态作为生产权威。
"""
from __future__ import annotations

import os
from typing import Any

import httpx


class QuantPaperUnavailableError(RuntimeError):
    """Quant Paper Authority 不可用。"""


class QuantPaperClient:
    def __init__(self, base_url: str | None = None, timeout: float = 15.0) -> None:
        self._url = (
            base_url or os.getenv("QUANT_SERVICE_URL") or os.getenv("MARKET_DATA_SERVICE_URL", "")
        ).rstrip("/")
        self._timeout = timeout
        self._account_id: str | None = None

    def configured(self) -> bool:
        return bool(self._url)

    # ------------------------------------------------------------- internal
    def _request(self, method: str, path: str, json: dict | None = None) -> dict[str, Any]:
        try:
            response = httpx.request(
                method, f"{self._url}{path}", json=json, timeout=self._timeout,
                headers={"X-Caller-Service": "stock_factor"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise QuantPaperUnavailableError(f"quant trading.v1 unavailable: {exc}") from exc
        body = response.json()
        return body.get("data", body)

    def _ensure_account(self) -> str:
        if self._account_id:
            return self._account_id
        account = self._request("POST", "/api/v1/paper/accounts", {"name": "factor-proxy", "initial_cash": 1_000_000})
        self._account_id = str(account["account_id"])
        return self._account_id

    # ------------------------------------------------------------------ API
    def generate_orders(self, scores: list[dict], as_of: str, snapshot_id: str, top_k: int) -> dict[str, Any]:
        account_id = self._ensure_account()
        return self._request(
            "POST",
            "/api/v1/paper/orders/generate",
            {"account_id": account_id, "as_of": as_of, "scores": scores, "top_k": top_k},
        )

    def run(self, as_of: str, snapshot_id: str, market_prices: dict[str, dict]) -> dict[str, Any]:
        account_id = self._ensure_account()
        return self._request(
            "POST",
            "/api/v1/paper/run",
            {"account_id": account_id, "as_of": as_of, "market_prices": market_prices},
        )

    def state(self) -> dict[str, Any]:
        account_id = self._ensure_account()
        return self._request("GET", f"/api/v1/paper/accounts/{account_id}")

    def equity(self) -> list[dict]:
        account_id = self._ensure_account()
        payload = self._request("GET", f"/api/v1/paper/accounts/{account_id}/equity")
        return payload if isinstance(payload, list) else payload.get("items", [])

    def replay(self) -> dict[str, Any]:
        account_id = self._ensure_account()
        trades = self._request("GET", f"/api/v1/paper/accounts/{account_id}/trades")
        return {
            "authority": "quant/trading.v1",
            "account_id": account_id,
            "trades": trades if isinstance(trades, list) else trades.get("items", []),
        }


__all__ = ["QuantPaperClient", "QuantPaperUnavailableError"]
