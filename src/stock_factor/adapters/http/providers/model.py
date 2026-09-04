"""HTTP model provider boundary."""

from __future__ import annotations

import os

import httpx


class HttpModelClient:
    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self._url = (url or os.getenv("FACTOR_MODEL_URL", "")).rstrip("/")
        self._model = model or os.getenv("FACTOR_MODEL_NAME", "")

    def complete(self, prompt: str, system: str | None = None, temperature: float | None = None) -> str:
        if not self._url:
            raise RuntimeError("FACTOR_MODEL_URL is not configured")
        response = httpx.post(
            self._url,
            json={"model": self._model, "prompt": prompt, "system": system, "temperature": temperature},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("text") or payload.get("output") or payload.get("content") or "")


__all__ = ["HttpModelClient"]
