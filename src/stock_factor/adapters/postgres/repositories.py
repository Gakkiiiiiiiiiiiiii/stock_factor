from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.orm import sessionmaker

from stock_factor.adapters.postgres.models import (
    FactorCandidateRow,
    FactorJobRow,
    FactorLifecycleEventRow,
    FactorPromotionDecisionRow,
    FactorRow,
    FactorVersionRow,
    PaperCashLedgerRow,
    PaperEquityRow,
    PaperFillRow,
    PaperOrderRow,
    PaperPositionLotRow,
    PaperRunRow,
    PaperStateRow,
)
from stock_factor.domain.factor import FactorDefinition, FactorJob


class PostgresFactorRepository:
    def __init__(self, sessions: sessionmaker) -> None:
        self._sessions = sessions

    @staticmethod
    def _payload(row: FactorRow) -> dict:
        return {
            name: getattr(row, name)
            for name in ("factor_id", "name", "rpn", "hypothesis", "status", "version", "metrics", "candidate_hash")
        }

    def list_active(self, limit: int = 20) -> list[dict]:
        with self._sessions() as session:
            rows = session.scalars(
                select(FactorRow).where(FactorRow.status == "ACTIVE").order_by(FactorRow.updated_at.desc()).limit(limit)
            ).all()
            return [self._payload(row) for row in rows]

    def get(self, factor_id: str) -> dict | None:
        with self._sessions() as session:
            row = session.get(FactorRow, factor_id)
            return self._payload(row) if row else None

    def save(self, factor: FactorDefinition) -> dict:
        with self._sessions.begin() as session:
            row = session.scalar(select(FactorRow).where(FactorRow.candidate_hash == factor.candidate_hash))
            prior_status = row.status if row else None
            if row is None:
                row = FactorRow(**factor.to_dict())
                session.add(row)
            else:
                row.metrics, row.status = factor.metrics, factor.status
            session.flush()
            canonical = " ".join(factor.rpn)
            candidate = session.scalar(
                select(FactorCandidateRow).where(FactorCandidateRow.candidate_hash == factor.candidate_hash)
            )
            if candidate is None:
                candidate = FactorCandidateRow(
                    candidate_id=uuid4().hex,
                    candidate_hash=factor.candidate_hash,
                    mining_job_id=(factor.metrics or {}).get("mining_job_id"),
                    parent_candidate_id=(factor.metrics or {}).get("parent_candidate_id"),
                    generation_round=int((factor.metrics or {}).get("generation_round") or 1),
                    generation_strategy=(factor.metrics or {}).get("generation_strategy") or "seed",
                    hypothesis=factor.hypothesis,
                    formula=factor.rpn,
                    canonical_formula=canonical,
                    feedback=(factor.metrics or {}).get("generation_feedback") or {},
                )
                session.add(candidate)
            version = session.scalar(
                select(FactorVersionRow).where(
                    FactorVersionRow.factor_id == row.factor_id, FactorVersionRow.version == row.version
                )
            )
            if version is None:
                session.add(
                    FactorVersionRow(
                        factor_version_id=uuid4().hex,
                        factor_id=row.factor_id,
                        version=row.version,
                        formula=row.rpn,
                        canonical_formula=canonical,
                        research_config=(row.metrics or {}).get("research_config") or {},
                    )
                )
            promotion = (factor.metrics or {}).get("promotion_gate") or {}
            session.add(
                FactorPromotionDecisionRow(
                    decision_id=uuid4().hex,
                    factor_id=row.factor_id,
                    factor_version=row.version,
                    mining_job_id=(factor.metrics or {}).get("mining_job_id"),
                    data_snapshot_id=(factor.metrics or {}).get("data_snapshot_id"),
                    passed=bool(promotion.get("passed")),
                    failed_rules=list(promotion.get("failed_rules") or []),
                    metrics_snapshot=promotion,
                )
            )
            if prior_status != row.status:
                session.add(
                    FactorLifecycleEventRow(
                        event_id=uuid4().hex,
                        factor_id=row.factor_id,
                        factor_version=row.version,
                        from_status=prior_status,
                        to_status=row.status,
                        reason="mining_promotion_gate",
                        metrics_snapshot=promotion,
                        data_snapshot_id=(factor.metrics or {}).get("data_snapshot_id"),
                    )
                )
            return self._payload(row)


class PostgresFactorJobRepository:
    def __init__(self, sessions: sessionmaker) -> None:
        self._sessions = sessions

    @staticmethod
    def _domain(row: FactorJobRow) -> FactorJob:
        return FactorJob(
            **{
                name: getattr(row, name)
                for name in (
                    "job_id",
                    "status",
                    "stage",
                    "progress",
                    "retry_count",
                    "max_retries",
                    "request",
                    "result",
                    "error",
                )
            }
        )

    def create(self, job: FactorJob) -> FactorJob:
        with self._sessions.begin() as session:
            session.add(FactorJobRow(**job.to_dict()))
        return job

    def get(self, job_id: str) -> FactorJob | None:
        with self._sessions() as session:
            row = session.get(FactorJobRow, job_id)
            return self._domain(row) if row else None

    def cancel(self, job_id: str) -> FactorJob | None:
        with self._sessions.begin() as session:
            row = session.get(FactorJobRow, job_id)
            if row and row.status == "PENDING":
                row.status = "CANCELLED"
            session.flush()
            return self._domain(row) if row else None

    def claim_pending(self, worker_id: str, lease_seconds: int) -> FactorJob | None:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.scalar(
                select(FactorJobRow)
                .where(
                    FactorJobRow.retry_count < FactorJobRow.max_retries,
                    or_(
                        FactorJobRow.status == "PENDING",
                        (FactorJobRow.status == "RUNNING") & (FactorJobRow.lease_expires_at < now),
                    ),
                )
                .order_by(FactorJobRow.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if row is None:
                return None
            row.status, row.lease_owner = "RUNNING", worker_id
            row.lease_expires_at = now + timedelta(seconds=lease_seconds)
            session.flush()
            return self._domain(row)

    def progress(self, job_id: str, stage: str, progress: int) -> None:
        with self._sessions.begin() as session:
            row = session.get(FactorJobRow, job_id)
            if row:
                row.stage, row.progress = stage, min(max(progress, 0), 100)

    def succeed(self, job_id: str, result: dict[str, Any]) -> None:
        with self._sessions.begin() as session:
            row = session.get(FactorJobRow, job_id)
            if row:
                row.status, row.stage, row.progress, row.result = "SUCCEEDED", "completed", 100, result
                row.lease_owner = row.lease_expires_at = None

    def fail(self, job_id: str, stage: str, error: str) -> None:
        with self._sessions.begin() as session:
            row = session.get(FactorJobRow, job_id)
            if row:
                row.retry_count += 1
                row.status = "FAILED" if row.retry_count >= row.max_retries else "PENDING"
                row.stage, row.error = stage, error[:4000]
                row.lease_owner = row.lease_expires_at = None


class PostgresPaperRepository:
    def __init__(self, sessions: sessionmaker) -> None:
        self._sessions = sessions

    def state(self, account_id: str = "default") -> dict:
        with self._sessions.begin() as session:
            row = session.get(PaperStateRow, account_id)
            if row is None:
                row = PaperStateRow(account_id=account_id)
                session.add(row)
                session.flush()
            return {
                "account_id": row.account_id,
                "cash": row.cash,
                "positions": row.positions or {},
                "frozen_orders": row.frozen_orders or [],
                "order_history": row.order_history or [],
                "fill_history": row.fill_history or [],
                "risk_events": row.risk_events or [],
                "data_snapshot_id": row.data_snapshot_id,
            }

    def freeze(self, orders: list[dict], snapshot_id: str, account_id: str = "default") -> dict:
        with self._sessions.begin() as session:
            row = session.get(PaperStateRow, account_id) or PaperStateRow(account_id=account_id)
            session.add(row)
            row.frozen_orders, row.data_snapshot_id = orders, snapshot_id
            row.order_history = [*(row.order_history or []), *orders]
            for order in orders:
                existing = session.get(PaperOrderRow, str(order["order_id"]))
                values = {
                    "run_id": None,
                    "symbol": str(order["symbol"]),
                    "side": str(order["side"]),
                    "requested_quantity": int(order.get("requested_quantity") or 0),
                    "filled_quantity": 0,
                    "remaining_quantity": int(order.get("requested_quantity") or 0),
                    "target_weight": order.get("target_weight"),
                    "status": str(order.get("status") or "FROZEN"),
                    "blocked_reason": order.get("reason"),
                    "execute_on": self._as_date(order.get("execute_on")),
                }
                if existing is None:
                    session.add(PaperOrderRow(order_id=str(order["order_id"]), **values))
                else:
                    for name, value in values.items():
                        setattr(existing, name, value)
            return {"account_id": account_id, "orders": orders, "data_snapshot_id": snapshot_id}

    def update_state(
        self,
        *,
        cash: float,
        positions: dict[str, Any],
        frozen_orders: list[dict],
        fills: list[dict],
        risk_events: list[dict],
        snapshot_id: str,
        account_id: str = "default",
    ) -> dict:
        with self._sessions.begin() as session:
            row = session.get(PaperStateRow, account_id) or PaperStateRow(account_id=account_id)
            session.add(row)
            row.cash = cash
            row.positions = positions
            row.frozen_orders = frozen_orders
            row.fill_history = [*(row.fill_history or []), *fills]
            row.risk_events = [*(row.risk_events or []), *risk_events]
            row.data_snapshot_id = snapshot_id
            self._persist_execution(session, account_id, cash, positions, fills, snapshot_id)
            session.flush()
            return {
                "account_id": row.account_id,
                "cash": row.cash,
                "positions": row.positions or {},
                "frozen_orders": row.frozen_orders or [],
                "order_history": row.order_history or [],
                "fill_history": row.fill_history or [],
                "risk_events": row.risk_events or [],
                "data_snapshot_id": row.data_snapshot_id,
            }

    @staticmethod
    def _as_date(value: object) -> date | None:
        if not value:
            return None
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])

    def _persist_execution(
        self, session, account_id: str, cash: float, positions: dict, fills: list[dict], snapshot_id: str
    ) -> None:
        if not fills:
            return
        trade_date = self._as_date(fills[-1].get("as_of")) or datetime.now(UTC).date()
        signal_snapshot_id = str(fills[-1].get("signal_snapshot_id") or snapshot_id)
        key = f"{account_id}:{trade_date.isoformat()}:{signal_snapshot_id}:{snapshot_id}"
        run = session.scalar(select(PaperRunRow).where(PaperRunRow.idempotency_key == key))
        if run is None:
            run = PaperRunRow(
                run_id=uuid4().hex,
                account_id=account_id,
                trade_date=trade_date,
                signal_snapshot_id=signal_snapshot_id,
                data_snapshot_id=snapshot_id,
                idempotency_key=key,
            )
            session.add(run)
        for fill in fills:
            if fill.get("status") != "FILLED":
                continue
            fill_id = uuid4().hex
            if session.scalar(
                select(PaperFillRow).where(
                    PaperFillRow.order_id == str(fill.get("order_id")),
                    PaperFillRow.executed_at == datetime.fromisoformat(str(fill["as_of"]) + "T00:00:00+00:00"),
                )
            ):
                continue
            qty, price = int(fill["filled_quantity"]), float(fill["execution_price"])
            costs = float(fill.get("fees") or 0.0)
            net_cash = -(abs(qty) * price + costs) if qty > 0 else abs(qty) * price - costs
            session.add(
                PaperFillRow(
                    fill_id=fill_id,
                    order_id=str(fill["order_id"]),
                    quantity=qty,
                    price=price,
                    gross_amount=abs(qty) * price,
                    commission=float(fill.get("commission") or 0.0),
                    stamp_duty=float(fill.get("stamp_duty") or 0.0),
                    slippage=0.0,
                    net_cash_change=net_cash,
                    executed_at=datetime.fromisoformat(str(fill["as_of"]) + "T00:00:00+00:00"),
                )
            )
            order = session.get(PaperOrderRow, str(fill["order_id"]))
            if order:
                order.run_id, order.filled_quantity, order.remaining_quantity = (
                    run.run_id,
                    abs(qty),
                    int(fill.get("remaining_quantity") or 0),
                )
                order.status = "FILLED" if not order.remaining_quantity else "PARTIALLY_FILLED"
            for event_type, amount in (
                ("BUY" if qty > 0 else "SELL", -abs(qty) * price if qty > 0 else abs(qty) * price),
                ("COMMISSION", -float(fill.get("commission") or 0.0)),
                ("STAMP_DUTY", -float(fill.get("stamp_duty") or 0.0)),
            ):
                if amount:
                    session.add(
                        PaperCashLedgerRow(
                            entry_id=uuid4().hex,
                            run_id=run.run_id,
                            event_type=event_type,
                            amount=amount,
                            balance_after=cash,
                            reference_id=fill_id,
                        )
                    )
        # The JSON state is a projection only.  Lots are independently
        # persisted so a replay can reproduce sellability and cost basis.
        active_symbols = set(positions)
        for lot in session.scalars(
            select(PaperPositionLotRow).where(PaperPositionLotRow.account_id == account_id)
        ):
            if lot.symbol not in active_symbols:
                lot.quantity, lot.available_quantity = 0, 0
                lot.remaining_cost, lot.closed_at = 0.0, datetime.now(UTC)
        for symbol, position in positions.items():
            for item in position.get("lots") or []:
                lot_id = str(item.get("lot_id") or uuid4().hex)
                item["lot_id"] = lot_id
                lot = session.get(PaperPositionLotRow, lot_id)
                values = {
                    "account_id": account_id,
                    "symbol": symbol,
                    "buy_date": self._as_date(item.get("buy_date")) or trade_date,
                    "quantity": int(item.get("quantity") or 0),
                    "available_quantity": int(item.get("available_quantity") or 0),
                    "cost_price": float(item.get("cost_price") or 0.0),
                    "remaining_cost": float(
                        item.get("remaining_cost")
                        or float(item.get("quantity") or 0) * float(item.get("cost_price") or 0.0)
                    ),
                }
                if lot is None:
                    session.add(PaperPositionLotRow(lot_id=lot_id, **values))
                else:
                    for name, value in values.items():
                        setattr(lot, name, value)

    def append_equity(
        self, as_of: str, equity: float, cash: float, snapshot_id: str, account_id: str = "default"
    ) -> None:
        with self._sessions.begin() as session:
            session.add(
                PaperEquityRow(
                    account_id=account_id, as_of=as_of, equity=equity, cash=cash, data_snapshot_id=snapshot_id
                )
            )

    def equity(self, account_id: str = "default") -> list[dict]:
        with self._sessions() as session:
            rows = session.scalars(
                select(PaperEquityRow).where(PaperEquityRow.account_id == account_id).order_by(PaperEquityRow.id)
            ).all()
            return [
                {"as_of": row.as_of, "equity": row.equity, "cash": row.cash, "data_snapshot_id": row.data_snapshot_id}
                for row in rows
            ]
