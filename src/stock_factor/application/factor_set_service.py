"""FactorSet 服务（详细修改方案 §11）。

发布 / 查询正式 FactorSet 版本；同内容寻址的 FactorSet 幂等复用，
新版本发布时旧版本自动 SUPERSEDED。
"""
from __future__ import annotations

from typing import Protocol

from stock_factor.domain.factor_set import FactorSet, factor_set_from_factors


class FactorSetStore(Protocol):
    def save(self, factor_set: FactorSet) -> None: ...

    def get(self, factor_set_id: str) -> FactorSet | None: ...

    def current(self) -> FactorSet | None: ...

    def history(self) -> list[FactorSet]: ...


class InMemoryFactorSetStore:
    def __init__(self) -> None:
        self._sets: dict[str, FactorSet] = {}
        self._order: list[str] = []

    def save(self, factor_set: FactorSet) -> None:
        if factor_set.factor_set_id in self._sets:
            return
        for existing_id in self._order:
            existing = self._sets[existing_id]
            if existing.status == "ACTIVE":
                object.__setattr__(existing, "status", "SUPERSEDED")
        self._sets[factor_set.factor_set_id] = factor_set
        self._order.append(factor_set.factor_set_id)

    def get(self, factor_set_id: str) -> FactorSet | None:
        return self._sets.get(factor_set_id)

    def current(self) -> FactorSet | None:
        for factor_set_id in reversed(self._order):
            if self._sets[factor_set_id].status == "ACTIVE":
                return self._sets[factor_set_id]
        return None

    def history(self) -> list[FactorSet]:
        return [self._sets[factor_set_id] for factor_set_id in self._order]


class FactorSetService:
    def __init__(self, store: FactorSetStore) -> None:
        self._store = store

    def publish_from_factors(self, factors: list[dict], research_experiment_ids: tuple[str, ...] = ()) -> FactorSet:
        factor_set = factor_set_from_factors(factors, research_experiment_ids=research_experiment_ids)
        self._store.save(factor_set)
        return factor_set

    def publish(self, factor_set: FactorSet) -> FactorSet:
        self._store.save(factor_set)
        return factor_set

    def get(self, factor_set_id: str) -> FactorSet | None:
        return self._store.get(factor_set_id)

    def current(self) -> FactorSet | None:
        return self._store.current()


class PostgresFactorSetStore:
    """factor_sets / factor_set_members 持久化（§17）。"""

    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def save(self, factor_set: FactorSet) -> None:
        from stock_factor.adapters.postgres.models import FactorSetMemberRow, FactorSetRow

        with self._sessions.begin() as session:
            from sqlalchemy import select

            existing = session.get(FactorSetRow, factor_set.factor_set_id)
            if existing is not None:
                return
            for row in session.scalars(select(FactorSetRow).where(FactorSetRow.status == "ACTIVE")):
                row.status = "SUPERSEDED"
            session.add(
                FactorSetRow(
                    factor_set_id=factor_set.factor_set_id,
                    factor_set_version=factor_set.factor_set_version,
                    research_experiment_ids=list(factor_set.research_experiment_ids),
                    promotion_policy_version=factor_set.promotion_policy_version,
                    valid_from=factor_set.valid_from,
                    valid_to=factor_set.valid_to,
                    status=factor_set.status,
                    code_sha=factor_set.code_sha,
                    config_hash=factor_set.config_hash,
                )
            )
            for index, factor_id in enumerate(factor_set.factor_ids):
                session.add(
                    FactorSetMemberRow(
                        factor_set_id=factor_set.factor_set_id,
                        factor_id=factor_id,
                        factor_version=int(factor_set.factor_versions[index]),
                        weight=float(factor_set.weights[index]),
                    )
                )

    def get(self, factor_set_id: str) -> FactorSet | None:
        from sqlalchemy import select

        from stock_factor.adapters.postgres.models import FactorSetMemberRow, FactorSetRow

        with self._sessions() as session:
            row = session.get(FactorSetRow, factor_set_id)
            if row is None:
                return None
            members = session.scalars(
                select(FactorSetMemberRow).where(FactorSetMemberRow.factor_set_id == factor_set_id)
            ).all()
            return self._to_domain(row, members)

    def current(self) -> FactorSet | None:
        from sqlalchemy import select

        from stock_factor.adapters.postgres.models import FactorSetMemberRow, FactorSetRow

        with self._sessions() as session:
            row = session.scalar(select(FactorSetRow).where(FactorSetRow.status == "ACTIVE").order_by(FactorSetRow.created_at.desc()))
            if row is None:
                return None
            members = session.scalars(
                select(FactorSetMemberRow).where(FactorSetMemberRow.factor_set_id == row.factor_set_id)
            ).all()
            return self._to_domain(row, members)

    def history(self) -> list[FactorSet]:
        from sqlalchemy import select

        from stock_factor.adapters.postgres.models import FactorSetMemberRow, FactorSetRow

        result: list[FactorSet] = []
        with self._sessions() as session:
            for row in session.scalars(select(FactorSetRow).order_by(FactorSetRow.created_at)):
                members = session.scalars(
                    select(FactorSetMemberRow).where(FactorSetMemberRow.factor_set_id == row.factor_set_id)
                ).all()
                result.append(self._to_domain(row, members))
        return result

    @staticmethod
    def _to_domain(row, members) -> FactorSet:
        ordered = sorted(members, key=lambda member: member.factor_id)
        factor_set = FactorSet(
            factor_ids=tuple(member.factor_id for member in ordered),
            factor_versions=tuple(member.factor_version for member in ordered),
            weights=tuple(member.weight for member in ordered),
            research_experiment_ids=tuple(row.research_experiment_ids or ()),
            promotion_policy_version=row.promotion_policy_version,
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            status=row.status,
            code_sha=row.code_sha,
            config_hash=row.config_hash,
        )
        return factor_set


__all__ = ["FactorSetStore", "InMemoryFactorSetStore", "PostgresFactorSetStore", "FactorSetService"]
