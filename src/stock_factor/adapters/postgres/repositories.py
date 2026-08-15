from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import sessionmaker

from stock_factor.adapters.postgres.models import FactorJobRow, FactorRow, PaperEquityRow, PaperStateRow
from stock_factor.domain.factor import FactorDefinition, FactorJob


class PostgresFactorRepository:
    def __init__(self, sessions: sessionmaker) -> None:
        self._sessions = sessions

    @staticmethod
    def _payload(row: FactorRow) -> dict:
        return {name: getattr(row, name) for name in ("factor_id", "name", "rpn", "hypothesis", "status", "version", "metrics", "candidate_hash")}

    def list_active(self, limit: int = 20) -> list[dict]:
        with self._sessions() as session:
            rows = session.scalars(select(FactorRow).where(FactorRow.status == "ACTIVE").order_by(FactorRow.updated_at.desc()).limit(limit)).all()
            return [self._payload(row) for row in rows]

    def get(self, factor_id: str) -> dict | None:
        with self._sessions() as session:
            row = session.get(FactorRow, factor_id)
            return self._payload(row) if row else None

    def save(self, factor: FactorDefinition) -> dict:
        with self._sessions.begin() as session:
            row = session.scalar(select(FactorRow).where(FactorRow.candidate_hash == factor.candidate_hash))
            if row is None:
                row = FactorRow(**factor.to_dict())
                session.add(row)
            else:
                row.metrics, row.status = factor.metrics, factor.status
            session.flush()
            return self._payload(row)


class PostgresFactorJobRepository:
    def __init__(self, sessions: sessionmaker) -> None:
        self._sessions = sessions

    @staticmethod
    def _domain(row: FactorJobRow) -> FactorJob:
        return FactorJob(**{name: getattr(row, name) for name in ("job_id", "status", "stage", "progress", "retry_count", "max_retries", "request", "result", "error")})

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
            row = session.scalar(select(FactorJobRow).where(FactorJobRow.retry_count < FactorJobRow.max_retries, or_(FactorJobRow.status == "PENDING", (FactorJobRow.status == "RUNNING") & (FactorJobRow.lease_expires_at < now))).order_by(FactorJobRow.created_at).limit(1).with_for_update(skip_locked=True))
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
            return {"account_id": row.account_id, "cash": row.cash, "positions": row.positions or {}, "frozen_orders": row.frozen_orders or [], "order_history": row.order_history or [], "fill_history": row.fill_history or [], "risk_events": row.risk_events or [], "data_snapshot_id": row.data_snapshot_id}

    def freeze(self, orders: list[dict], snapshot_id: str, account_id: str = "default") -> dict:
        with self._sessions.begin() as session:
            row = session.get(PaperStateRow, account_id) or PaperStateRow(account_id=account_id)
            session.add(row)
            row.frozen_orders, row.data_snapshot_id = orders, snapshot_id
            row.order_history = [*(row.order_history or []), *orders]
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
            session.flush()
            return self.state(account_id)

    def append_equity(self, as_of: str, equity: float, cash: float, snapshot_id: str, account_id: str = "default") -> None:
        with self._sessions.begin() as session:
            session.add(PaperEquityRow(account_id=account_id, as_of=as_of, equity=equity, cash=cash, data_snapshot_id=snapshot_id))

    def equity(self, account_id: str = "default") -> list[dict]:
        with self._sessions() as session:
            rows = session.scalars(select(PaperEquityRow).where(PaperEquityRow.account_id == account_id).order_by(PaperEquityRow.id)).all()
            return [{"as_of": row.as_of, "equity": row.equity, "cash": row.cash, "data_snapshot_id": row.data_snapshot_id} for row in rows]
