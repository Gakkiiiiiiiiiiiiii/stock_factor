"""FactorSet 服务（详细修改方案 §11）。

发布 / 查询正式 FactorSet 版本；同内容寻址的 FactorSet 幂等复用，
新版本发布时旧版本自动 SUPERSEDED。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from stock_factor.domain.factor_set import FactorSet, factor_set_from_factors
from stock_factor.domain.tradability_artifact import TradabilityAssessmentArtifact


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

    def publish_from_factors(
        self,
        factors: list[dict],
        research_experiment_ids: tuple[str, ...] = (),
        *,
        artifact_id: str | None = None,
        artifact_service=None,
    ) -> FactorSet:
        if artifact_id is not None:
            if artifact_service is None:
                raise ValueError("formal FactorSet promotion requires artifact_service")
            return self.publish_formal(artifact_id, artifact_service)
        factor_set = factor_set_from_factors(factors, research_experiment_ids=research_experiment_ids)
        self._store.save(factor_set)
        return factor_set

    def publish_formal(self, artifact_id: str, artifact_service) -> FactorSet:
        """Publish only from a stored, verified, sealed V2 artifact."""
        artifact = artifact_service.require_verified_sealed(artifact_id)
        if artifact.promotion_decision.get("passed") is not True:
            raise ValueError("sealed artifact promotion decision did not pass")
        if artifact.final_oos_evidence.get("status") not in {"SEALED", "PASSED"}:
            raise ValueError("sealed artifact does not contain sealed Final OOS evidence")
        gate_result = artifact.tradability_assessment.get("gate_result")
        if not isinstance(gate_result, Mapping) or gate_result.get("passed") is not True:
            raise ValueError("sealed artifact tradability gate did not pass")
        if artifact.tradability_assessment.get("formal_eligible") is not True:
            raise ValueError("sealed artifact tradability evidence is not formal eligible")
        if not artifact.tradability_assessment.get("artifact_id"):
            raise ValueError("formal FactorSet promotion requires a sealed tradability artifact")
        if artifact.tradability_assessment.get("artifact_id") and not TradabilityAssessmentArtifact.verify_payload(
            artifact.tradability_assessment
        ):
            raise ValueError("sealed artifact tradability evidence integrity failed")
        evidence = artifact.factor_set
        if not isinstance(evidence, Mapping):
            raise ValueError("sealed artifact does not contain FactorSet evidence")
        factors = evidence.get("factors")
        if not isinstance(factors, Sequence) or isinstance(factors, (str, bytes)) or not factors:
            raise ValueError("sealed artifact FactorSet evidence is incomplete")
        if any(not isinstance(item, Mapping) or not item.get("factor_id") for item in factors):
            raise ValueError("sealed artifact FactorSet factors are invalid")
        if len({str(item["factor_id"]) for item in factors}) != len(factors):
            raise ValueError("sealed artifact FactorSet factors contain duplicates")
        tradability_factor = artifact.tradability_assessment.get("factor_artifact_id")
        if tradability_factor and str(tradability_factor) not in {str(item["factor_id"]) for item in factors}:
            raise ValueError("sealed artifact tradability factor binding mismatch")
        tradability_market = artifact.tradability_assessment.get("market_snapshot_id")
        market_snapshot = artifact.market_ref.get("market_snapshot_id")
        if tradability_market and market_snapshot and tradability_market != market_snapshot:
            raise ValueError("sealed artifact tradability market binding mismatch")
        if artifact.tradability_assessment.get("artifact_id"):
            if not artifact.tradability_assessment.get("policy_version") or not artifact.tradability_assessment.get(
                "policy_hash"
            ):
                raise ValueError("sealed artifact tradability policy identity is incomplete")
            calibration = artifact.tradability_assessment.get("execution_cost_calibration")
            if not isinstance(calibration, Mapping) or not calibration.get("checksum"):
                raise ValueError("sealed artifact tradability calibration identity is incomplete")
        factor_set = factor_set_from_factors(
            factors,
            research_experiment_ids=(artifact.experiment_id,),
            promotion_policy_version=artifact.promotion_policy_version,
            research_artifact_ids=(artifact.artifact_id,),
            formal_eligible=True,
        )
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
                    research_artifact_ids=list(factor_set.research_artifact_ids),
                    formal_eligible=factor_set.formal_eligible,
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
            row = session.scalar(
                select(FactorSetRow).where(FactorSetRow.status == "ACTIVE").order_by(FactorSetRow.created_at.desc())
            )
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
            research_artifact_ids=tuple(getattr(row, "research_artifact_ids", None) or ()),
            formal_eligible=bool(getattr(row, "formal_eligible", False)),
        )
        return factor_set


__all__ = ["FactorSetStore", "InMemoryFactorSetStore", "PostgresFactorSetStore", "FactorSetService"]
