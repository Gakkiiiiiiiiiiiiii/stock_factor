from __future__ import annotations

from datetime import date
from math import floor
from typing import Protocol

from stock_factor.ports.trading_calendar import TradingCalendar, WeekdayTradingCalendar


class PaperRepository(Protocol):
    def state(self, account_id: str = "default") -> dict: ...
    def freeze(self, orders: list[dict], snapshot_id: str, account_id: str = "default") -> dict: ...
    def append_equity(
        self, as_of: str, equity: float, cash: float, snapshot_id: str, account_id: str = "default"
    ) -> None: ...
    def update_state(
        self,
        *,
        cash: float,
        positions: dict,
        frozen_orders: list[dict],
        fills: list[dict],
        risk_events: list[dict],
        snapshot_id: str,
        account_id: str = "default",
    ) -> dict: ...
    def equity(self, account_id: str = "default") -> list[dict]: ...


class PaperTradingService:
    """T-1 order planning plus deterministic, cost-aware paper execution."""

    def __init__(self, repository: PaperRepository, trading_calendar: TradingCalendar | None = None) -> None:
        self._repository = repository
        self._calendar = trading_calendar or WeekdayTradingCalendar()

    @staticmethod
    def _exchange(symbol: str) -> str:
        return (
            "HK"
            if symbol.endswith(".HK")
            else "US"
            if symbol.rsplit(".", 1)[-1].isalpha() and not symbol.endswith((".SH", ".SZ"))
            else "CN"
        )

    def _execution_day(self, signal_day: str, symbol: str) -> str:
        return self._calendar.next_trading_day(self._exchange(symbol), date.fromisoformat(signal_day[:10])).isoformat()

    def generate_orders(self, scores: list[dict], as_of: str, snapshot_id: str, top_k: int = 10) -> dict:
        ranked = sorted(scores, key=lambda item: float(item.get("score", 0)), reverse=True)[:top_k]
        state = self._repository.state()
        targets = {str(item["symbol"]) for item in ranked}
        # Rebalancing is a delta plan: exits are frozen before buys so their
        # proceeds are available to the same execution cycle.
        exits = [
            {
                "order_id": f"{as_of}:{symbol}:SELL",
                "symbol": symbol,
                "side": "SELL",
                "target_weight": 0.0,
                "signal_as_of": as_of,
                "execute_on": self._execution_day(as_of, symbol),
                "status": "FROZEN",
                "order_type": "MARKET_OPEN",
                "lot_size": 100,
                "max_participation": 0.1,
                "fee_bps": 5.0,
                "minimum_commission": 5.0,
                "stamp_duty_bps": 5.0,
                "slippage_bps": 10.0,
                "factor_ids": [],
                "data_version": None,
            }
            for symbol, position in state["positions"].items()
            if symbol not in targets and int(position.get("quantity") or 0) > 0
        ]
        entries = [
            {
                "order_id": f"{as_of}:{item['symbol']}:BUY",
                "symbol": item["symbol"],
                "side": "BUY",
                "target_weight": round(1 / max(len(ranked), 1), 8),
                "signal_as_of": as_of,
                "execute_on": self._execution_day(as_of, str(item["symbol"])),
                "status": "FROZEN",
                "order_type": "MARKET_OPEN",
                "lot_size": int(item.get("lot_size") or 100),
                "max_participation": float(item.get("max_participation") or 0.1),
                "fee_bps": float(item.get("fee_bps") or 5.0),
                "minimum_commission": float(item.get("minimum_commission") or 5.0),
                "stamp_duty_bps": float(item.get("stamp_duty_bps") or 5.0),
                "slippage_bps": float(item.get("slippage_bps") or 10.0),
                "factor_ids": list(item.get("factor_ids") or []),
                "data_version": item.get("data_version"),
            }
            for item in ranked
        ]
        return self._repository.freeze([*exits, *entries], snapshot_id)

    def run(self, as_of: str, snapshot_id: str, market_prices: dict[str, dict] | None = None) -> dict:
        state = self._repository.state()
        prices = market_prices or {}
        cash = float(state["cash"])
        positions = dict(state["positions"])
        for position in positions.values():
            for lot in position.get("lots") or []:
                if str(lot.get("buy_date") or "") < as_of:
                    lot["available_quantity"] = int(lot.get("quantity") or 0)
        frozen = list(state["frozen_orders"])
        fills: list[dict] = []
        risks: list[dict] = []
        marked_equity = cash + sum(
            float(position.get("quantity", 0)) * self._mark_price(symbol, position, prices)
            for symbol, position in positions.items()
        )
        next_orders: list[dict] = []
        for order in frozen:
            if order.get("execute_on") and str(order["execute_on"]) > as_of:
                next_orders.append(order)
                continue
            quote = prices.get(order["symbol"])
            if not quote:
                next_orders.append({**order, "status": "PENDING_MARKET_DATA"})
                risks.append({"as_of": as_of, "symbol": order["symbol"], "code": "MARKET_DATA_MISSING"})
                continue
            if not quote.get("tradable", True) or quote.get("suspended", False):
                next_orders.append({**order, "status": "BLOCKED", "reason": "NOT_TRADABLE"})
                risks.append({"as_of": as_of, "symbol": order["symbol"], "code": "NOT_TRADABLE"})
                continue
            raw_price = float(quote.get("open") or quote.get("price") or 0.0)
            if raw_price <= 0:
                next_orders.append({**order, "status": "PENDING_MARKET_DATA"})
                continue
            upper, lower = quote.get("upper_limit"), quote.get("lower_limit")
            side = str(order.get("side") or "BUY").upper()
            blocked_at_limit = (side == "BUY" and upper is not None and raw_price >= float(upper)) or (
                side == "SELL" and lower is not None and raw_price <= float(lower)
            )
            if blocked_at_limit:
                next_orders.append({**order, "status": "BLOCKED", "reason": "PRICE_LIMIT"})
                risks.append({"as_of": as_of, "symbol": order["symbol"], "code": "PRICE_LIMIT"})
                continue
            slip = float(order.get("slippage_bps") or 0.0) / 10_000
            fee = float(order.get("fee_bps") or 0.0) / 10_000
            execution_price = raw_price * (1 + slip if side == "BUY" else 1 - slip)
            existing = positions.get(order["symbol"], {})
            desired_value = marked_equity * float(order.get("target_weight") or 0.0)
            current_qty = int(existing.get("quantity") or 0)
            desired_qty = floor(desired_value / execution_price / max(int(order.get("lot_size") or 100), 1)) * max(
                int(order.get("lot_size") or 100), 1
            )
            quantity = desired_qty - current_qty if side == "BUY" else -current_qty
            if quantity < 0:
                lots = list(existing.get("lots") or [])
                if lots:
                    sellable = sum(
                        int(lot.get("available_quantity", lot.get("quantity", 0)) or 0)
                        for lot in lots
                        if str(lot.get("buy_date") or "") < as_of
                    )
                    quantity = -min(abs(quantity), sellable)
            if quantity == 0:
                fills.append({**order, "status": "NO_CHANGE", "filled_quantity": 0, "as_of": as_of})
                continue
            volume = quote.get("volume") or quote.get("tradable_volume")
            if volume is not None:
                lot_size = max(int(order.get("lot_size") or 100), 1)
                max_fill = floor(float(volume) * float(order.get("max_participation") or 0.1) / lot_size) * lot_size
                quantity = min(quantity, max_fill) if quantity > 0 else -min(abs(quantity), max_fill)
            if quantity > 0:
                affordable = floor(
                    cash / (execution_price * (1 + fee)) / max(int(order.get("lot_size") or 100), 1)
                ) * max(int(order.get("lot_size") or 100), 1)
                quantity = min(quantity, affordable)
            else:
                quantity = -min(abs(quantity), current_qty)
            if quantity == 0:
                next_orders.append({**order, "status": "REJECTED", "reason": "INSUFFICIENT_CASH"})
                continue
            notional = abs(quantity) * execution_price
            commission = max(notional * fee, float(order.get("minimum_commission") or 0.0))
            stamp_duty = notional * float(order.get("stamp_duty_bps") or 0.0) / 10_000 if quantity < 0 else 0.0
            costs = commission + stamp_duty
            cash += -notional - costs if quantity > 0 else notional - costs
            new_qty = current_qty + quantity
            if new_qty:
                avg_cost = float(existing.get("avg_cost") or execution_price)
                if quantity > 0:
                    avg_cost = (avg_cost * current_qty + execution_price * quantity + costs) / new_qty
                realized = float(existing.get("realized_pnl") or 0.0)
                if quantity < 0:
                    consumed_cost = self._consume_lots(lots, abs(quantity))
                    realized += execution_price * abs(quantity) - consumed_cost - costs
                positions[order["symbol"]] = {
                    "quantity": new_qty,
                    "avg_cost": avg_cost,
                    "last_price": raw_price,
                    "realized_pnl": realized,
                    "lots": (
                        [
                            *list(existing.get("lots") or []),
                            {
                                "buy_date": as_of,
                                "quantity": quantity,
                                "available_quantity": 0,
                                "cost_price": execution_price,
                                "remaining_cost": execution_price * quantity + costs,
                            },
                        ]
                        if quantity > 0
                        else [lot for lot in lots if int(lot.get("quantity") or 0) > 0]
                    ),
                }
            else:
                positions.pop(order["symbol"], None)
            remaining_quantity = max(abs(desired_qty - current_qty) - abs(quantity), 0)
            if remaining_quantity:
                next_orders.append({**order, "status": "PARTIALLY_FILLED", "remaining_quantity": remaining_quantity})
            fills.append(
                {
                    **order,
                    "status": "FILLED",
                    "filled_quantity": quantity,
                    "remaining_quantity": remaining_quantity,
                    "execution_price": round(execution_price, 8),
                    "fees": round(costs, 8),
                    "commission": round(commission, 8),
                    "stamp_duty": round(stamp_duty, 8),
                    "as_of": as_of,
                }
            )
        updated = self._repository.update_state(
            cash=cash,
            positions=positions,
            frozen_orders=next_orders,
            fills=fills,
            risk_events=risks,
            snapshot_id=snapshot_id,
        )
        equity = cash + sum(
            float(position.get("quantity", 0)) * self._mark_price(symbol, position, prices)
            for symbol, position in positions.items()
        )
        self._repository.append_equity(as_of, equity, cash, snapshot_id)
        return {
            "as_of": as_of,
            "equity": round(equity, 8),
            "cash": round(cash, 8),
            "data_snapshot_id": snapshot_id,
            "filled_order_count": sum(item["status"] == "FILLED" for item in fills),
            "pending_order_count": len(updated["frozen_orders"]),
            "risk_events": risks,
        }

    @staticmethod
    def _mark_price(symbol: str, position: dict, prices: dict[str, dict]) -> float:
        quote = prices.get(symbol) or {}
        return float(
            quote.get("close") or quote.get("price") or position.get("last_price") or position.get("avg_cost") or 0.0
        )

    def state(self) -> dict:
        return self._repository.state()

    def equity(self) -> list[dict]:
        return self._repository.equity()

    @staticmethod
    def _consume_lots(lots: list[dict], quantity: int) -> float:
        """Consume FIFO lots exactly once and return their remaining cost basis."""
        remaining, consumed_cost = quantity, 0.0
        for lot in sorted(lots, key=lambda value: str(value.get("buy_date") or "")):
            if remaining <= 0:
                break
            available = int(lot.get("available_quantity", lot.get("quantity", 0)) or 0)
            used = min(remaining, available)
            if not used:
                continue
            total_quantity = max(int(lot.get("quantity") or 0), 1)
            lot_cost = float(lot.get("remaining_cost") or float(lot.get("cost_price") or 0.0) * total_quantity)
            proportional_cost = lot_cost * used / total_quantity
            consumed_cost += proportional_cost
            lot["quantity"] = total_quantity - used
            lot["available_quantity"] = max(available - used, 0)
            lot["remaining_cost"] = max(lot_cost - proportional_cost, 0.0)
            remaining -= used
        return consumed_cost
