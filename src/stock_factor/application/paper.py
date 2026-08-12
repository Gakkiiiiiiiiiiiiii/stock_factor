from __future__ import annotations

from typing import Protocol


class PaperRepository(Protocol):
    def state(self, account_id: str = "default") -> dict: ...
    def freeze(self, orders: list[dict], snapshot_id: str, account_id: str = "default") -> dict: ...
    def append_equity(self, as_of: str, equity: float, cash: float, snapshot_id: str, account_id: str = "default") -> None: ...
    def equity(self, account_id: str = "default") -> list[dict]: ...


class PaperTradingService:
    """Freezes T-1 target orders and keeps an idempotent paper account state."""

    def __init__(self, repository: PaperRepository) -> None:
        self._repository = repository

    def generate_orders(self, scores: list[dict], as_of: str, snapshot_id: str, top_k: int = 10) -> dict:
        ranked = sorted(scores, key=lambda item: float(item.get("score", 0)), reverse=True)[:top_k]
        orders = [
            {
                "order_id": f"{as_of}:{item['symbol']}",
                "symbol": item["symbol"],
                "side": "BUY",
                "target_weight": round(1 / max(len(ranked), 1), 8),
                "signal_as_of": as_of,
                "execute_on": item.get("next_trading_day"),
                "status": "FROZEN",
            }
            for item in ranked
        ]
        return self._repository.freeze(orders, snapshot_id)

    def run(self, as_of: str, snapshot_id: str) -> dict:
        state = self._repository.state()
        equity = float(state["cash"])
        self._repository.append_equity(as_of, equity, float(state["cash"]), snapshot_id)
        return {"as_of": as_of, "equity": equity, "cash": state["cash"], "data_snapshot_id": snapshot_id, "frozen_order_count": len(state["frozen_orders"])}

    def state(self) -> dict:
        return self._repository.state()

    def equity(self) -> list[dict]:
        return self._repository.equity()
