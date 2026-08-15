from __future__ import annotations

from pathlib import Path

import yaml

from stock_factor.engine.vocab import is_valid_token


class Alpha191SeedLibrary:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path(__file__).resolve().parents[3] / "config" / "factor_seed_alpha191.yaml"

    def load(self) -> list[dict]:
        payload = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        seeds = payload.get("seeds") if isinstance(payload, dict) else []
        return [
            {
                "name": str(item["name"]),
                "hypothesis": str(item.get("hypothesis") or ""),
                "rpn": [str(token) for token in item["rpn"]],
            }
            for item in seeds
            if isinstance(item, dict)
            and item.get("name")
            and item.get("rpn")
            and all(is_valid_token(str(token)) for token in item["rpn"])
        ]
