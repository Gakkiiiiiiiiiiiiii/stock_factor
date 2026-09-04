"""Final OOS 数据库级一次性授权（详细修改方案 P0-2）。

应用层检查 experiment.status == OOS_AUTHORIZED 已不足以阻止并发双重消费；
授权记录落库，AUTHORIZED -> CONSUMED 在同一事务内完成（SELECT ... FOR UPDATE）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

AUTHORIZED = "AUTHORIZED"
CONSUMED = "CONSUMED"
INVALIDATED = "INVALIDATED"


class OosAuthorizationError(RuntimeError):
    """Final OOS 授权异常基类。"""


class OosAuthorizationMissingError(OosAuthorizationError):
    """实验没有授权记录却尝试消费。"""


class OosAuthorizationConsumedError(OosAuthorizationError):
    """授权已被消费：Final OOS 只能一次性使用。"""


class OosAuthorizationInvalidatedError(OosAuthorizationError):
    """授权已失效。"""


class OosAuthorizationStore(Protocol):
    def authorize(self, experiment_id: str, final_oos_snapshot_id: str = "", candidate_set_hash: str = "") -> dict: ...

    def consume(self, experiment_id: str) -> dict: ...

    def invalidate(self, experiment_id: str, reason: str = "") -> None: ...

    def get(self, experiment_id: str) -> dict | None: ...


class InMemoryOosAuthorizationStore:
    """单元测试 / 无持久化运行场景（线程锁模拟事务原子性）。"""

    def __init__(self) -> None:
        import threading

        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def authorize(self, experiment_id: str, final_oos_snapshot_id: str = "", candidate_set_hash: str = "") -> dict:
        with self._lock:
            existing = self._records.get(experiment_id)
            if existing is not None:
                if (
                    existing["final_oos_snapshot_id"] != final_oos_snapshot_id
                    or existing["candidate_set_hash"] != candidate_set_hash
                ):
                    raise OosAuthorizationError("authorization identity drift")
                if existing["status"] == CONSUMED:
                    raise OosAuthorizationConsumedError(f"experiment {experiment_id} 的 Final OOS 授权已被消费")
                if existing["status"] == INVALIDATED:
                    raise OosAuthorizationInvalidatedError(f"experiment {experiment_id} 的 Final OOS 授权已失效")
                return dict(existing)
            record = {
                "authorization_id": f"oosa-{experiment_id}",
                "experiment_id": experiment_id,
                "final_oos_snapshot_id": final_oos_snapshot_id,
                "candidate_set_hash": candidate_set_hash,
                "status": AUTHORIZED,
                "authorized_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "consumed_at": None,
                "invalidated_at": None,
            }
            self._records[experiment_id] = record
            return dict(record)

    def consume(self, experiment_id: str) -> dict:
        with self._lock:
            record = self._records.get(experiment_id)
            if record is None:
                raise OosAuthorizationMissingError(f"experiment {experiment_id} 没有 Final OOS 授权记录")
            if record["status"] == CONSUMED:
                raise OosAuthorizationConsumedError(f"experiment {experiment_id} 的 Final OOS 授权已被消费")
            if record["status"] == INVALIDATED:
                raise OosAuthorizationInvalidatedError(f"experiment {experiment_id} 的 Final OOS 授权已失效")
            record["status"] = CONSUMED
            record["consumed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return dict(record)

    def invalidate(self, experiment_id: str, reason: str = "") -> None:
        with self._lock:
            record = self._records.get(experiment_id)
            if record is not None and record["status"] == AUTHORIZED:
                record["status"] = INVALIDATED
                record["invalidated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
                record["invalidation_reason"] = reason

    def get(self, experiment_id: str) -> dict | None:
        with self._lock:
            record = self._records.get(experiment_id)
            return dict(record) if record else None


class PostgresOosAuthorizationStore:
    """数据库事务级一次性授权（P0-2）。

    consume() 在同一事务内 SELECT ... FOR UPDATE + AUTHORIZED -> CONSUMED，
    并发 worker 只有一个能成功消费。
    """

    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def authorize(self, experiment_id: str, final_oos_snapshot_id: str = "", candidate_set_hash: str = "") -> dict:
        from sqlalchemy import select

        from stock_factor.adapters.postgres.models import OosAuthorizationRow

        with self._sessions.begin() as session:
            row = session.scalar(select(OosAuthorizationRow).where(OosAuthorizationRow.experiment_id == experiment_id))
            if row is not None:
                if row.status == AUTHORIZED:
                    if (
                        row.final_oos_snapshot_id != final_oos_snapshot_id
                        or row.candidate_set_hash != candidate_set_hash
                    ):
                        raise OosAuthorizationError("authorization identity drift")
                    return _payload(row)
                if row.status == CONSUMED:
                    raise OosAuthorizationConsumedError(f"experiment {experiment_id} 的 Final OOS 授权已被消费")
                raise OosAuthorizationInvalidatedError(f"experiment {experiment_id} 的 Final OOS 授权已失效")
            row = OosAuthorizationRow(
                experiment_id=experiment_id,
                final_oos_snapshot_id=final_oos_snapshot_id,
                candidate_set_hash=candidate_set_hash,
                status=AUTHORIZED,
            )
            session.add(row)
            session.flush()
            return _payload(row)

    def consume(self, experiment_id: str) -> dict:
        from sqlalchemy import select, update

        from stock_factor.adapters.postgres.models import OosAuthorizationRow

        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            # Postgres：行锁；SQLite 忽略 FOR UPDATE，真正的原子性由下方条件 UPDATE 保证。
            session.scalar(
                select(OosAuthorizationRow).where(OosAuthorizationRow.experiment_id == experiment_id).with_for_update()
            )
            # 条件 UPDATE：单语句原子，并发下只有一个执行者能把 AUTHORIZED 改为 CONSUMED。
            result = session.execute(
                update(OosAuthorizationRow)
                .where(OosAuthorizationRow.experiment_id == experiment_id, OosAuthorizationRow.status == AUTHORIZED)
                .values(status=CONSUMED, consumed_at=now)
            )
            if result.rowcount == 0:
                row = session.scalar(
                    select(OosAuthorizationRow).where(OosAuthorizationRow.experiment_id == experiment_id)
                )
                if row is None:
                    raise OosAuthorizationMissingError(f"experiment {experiment_id} 没有 Final OOS 授权记录")
                if row.status == CONSUMED:
                    raise OosAuthorizationConsumedError(f"experiment {experiment_id} 的 Final OOS 授权已被消费")
                raise OosAuthorizationInvalidatedError(f"experiment {experiment_id} 的 Final OOS 授权已失效")
            session.flush()
            row = session.scalar(select(OosAuthorizationRow).where(OosAuthorizationRow.experiment_id == experiment_id))
            return _payload(row)

    def invalidate(self, experiment_id: str, reason: str = "") -> None:
        from sqlalchemy import select

        from stock_factor.adapters.postgres.models import OosAuthorizationRow

        with self._sessions.begin() as session:
            row = session.scalar(select(OosAuthorizationRow).where(OosAuthorizationRow.experiment_id == experiment_id))
            if row is not None and row.status == AUTHORIZED:
                row.status = INVALIDATED
                row.invalidated_at = datetime.now(UTC)

    def get(self, experiment_id: str) -> dict | None:
        from sqlalchemy import select

        from stock_factor.adapters.postgres.models import OosAuthorizationRow

        with self._sessions() as session:
            row = session.scalar(select(OosAuthorizationRow).where(OosAuthorizationRow.experiment_id == experiment_id))
            return _payload(row) if row else None


def _payload(row) -> dict:
    return {
        "authorization_id": row.authorization_id,
        "experiment_id": row.experiment_id,
        "final_oos_snapshot_id": row.final_oos_snapshot_id,
        "candidate_set_hash": row.candidate_set_hash,
        "status": row.status,
        "authorized_at": row.authorized_at.isoformat(timespec="seconds") if row.authorized_at else None,
        "consumed_at": row.consumed_at.isoformat(timespec="seconds") if row.consumed_at else None,
        "invalidated_at": row.invalidated_at.isoformat(timespec="seconds") if row.invalidated_at else None,
    }


__all__ = [
    "AUTHORIZED",
    "CONSUMED",
    "INVALIDATED",
    "OosAuthorizationError",
    "OosAuthorizationMissingError",
    "OosAuthorizationConsumedError",
    "OosAuthorizationInvalidatedError",
    "OosAuthorizationStore",
    "InMemoryOosAuthorizationStore",
    "PostgresOosAuthorizationStore",
]
