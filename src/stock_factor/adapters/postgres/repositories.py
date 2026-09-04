from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.orm import sessionmaker

from stock_factor.adapters.postgres.models import (
    FactorCandidateFreezeRow,
    FactorCandidateRow,
    FactorFinalOosEvaluationRow,
    FactorFinalOosRow,
    FactorJobRow,
    FactorLifecycleEventRow,
    FactorOosAuditRow,
    FactorPromotionDecisionRow,
    FactorRow,
    FactorStatisticalTestRow,
    FactorVersionRow,
)
from stock_factor.domain.factor import FactorDefinition, FactorJob
from stock_factor.engine.oos_seal import CandidateFreeze


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
                session.flush()
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
            statistics = (factor.metrics or {}).get("statistics") or {}
            final_oos = (factor.metrics or {}).get("final_oos") or {}
            oos_audit = (factor.metrics or {}).get("oos_audit") or {}
            if not session.scalar(
                select(FactorStatisticalTestRow).where(
                    FactorStatisticalTestRow.candidate_id == candidate.candidate_id,
                    FactorStatisticalTestRow.experiment_id == str((factor.metrics or {}).get("data_snapshot_id") or ""),
                )
            ):
                session.add(
                    FactorStatisticalTestRow(
                        candidate_id=candidate.candidate_id,
                        experiment_id=str((factor.metrics or {}).get("data_snapshot_id") or "default"),
                        raw_p_value=statistics.get("raw_p_value"),
                        adjusted_p_value=statistics.get("adjusted_p_value"),
                        q_value=statistics.get("q_value"),
                        pbo=statistics.get("pbo"),
                        effective_trials=int(statistics.get("effective_trials") or 0),
                        passed_multiple_testing=bool(statistics.get("passed_multiple_testing")),
                        passed_pbo=bool(statistics.get("passed_pbo")),
                        method=str(statistics.get("method") or "multiple_testing_v1"),
                        data_snapshot_id=(factor.metrics or {}).get("data_snapshot_id"),
                        discovery_snapshot_id=(factor.metrics or {}).get("discovery_snapshot_id"),
                        final_oos_snapshot_id=(factor.metrics or {}).get("final_oos_snapshot_id"),
                    )
                )
                session.add(
                    FactorFinalOosRow(
                        factor_id=row.factor_id,
                        factor_version=row.version,
                        metrics=final_oos,
                        data_snapshot_id=(factor.metrics or {}).get("data_snapshot_id"),
                        discovery_snapshot_id=(factor.metrics or {}).get("discovery_snapshot_id"),
                        final_oos_snapshot_id=(factor.metrics or {}).get("final_oos_snapshot_id"),
                    )
                )
                session.add(
                    FactorOosAuditRow(
                        factor_id=row.factor_id,
                        factor_version=row.version,
                        audit_status=str(oos_audit.get("audit_status") or "FAILED"),
                        violations=list(oos_audit.get("violations") or []),
                        warnings=list(oos_audit.get("warnings") or []),
                        audit_version=str(oos_audit.get("audit_version") or "final_oos_audit_v1"),
                    )
                )
            session.add(
                FactorPromotionDecisionRow(
                    decision_id=uuid4().hex,
                    factor_id=row.factor_id,
                    factor_version=row.version,
                    mining_job_id=(factor.metrics or {}).get("mining_job_id"),
                    data_snapshot_id=(factor.metrics or {}).get("data_snapshot_id"),
                    passed=bool(promotion.get("passed")),
                    failed_rules=list(promotion.get("reject_reasons") or promotion.get("failed_rules") or []),
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
            # §33：同一 Idempotency-Key 的重复提交直接返回已有任务。
            if job.idempotency_key:
                existing = session.scalar(
                    select(FactorJobRow).where(FactorJobRow.idempotency_key == job.idempotency_key)
                )
                if existing is not None:
                    return self._domain(existing)
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


class PostgresCandidateSealStore:
    """Candidate Freeze / Final OOS 评估记录的 PostgreSQL 持久化（§13.3/§13.4）。"""

    def __init__(self, sessions: sessionmaker) -> None:
        self._sessions = sessions

    def save_freeze(self, freeze: CandidateFreeze) -> None:
        extra = {
            key: getattr(freeze, key)
            for key in (
                "experiment_id",
                "discovery_config_hash",
                "selection_policy_version",
                "selection_rank",
                "research_code_version",
                "selected_at",
                # P0-4 完整 freeze 证据
                "factor_code_hash",
                "universe_snapshot_id",
                "feature_normalization_version",
                "selection_config_hash",
                "discovery_metrics_hash",
                "candidate_count",
                "code_sha",
                "config_hash",
            )
            if getattr(freeze, key) is not None
        }
        with self._sessions.begin() as session:
            row = session.get(FactorCandidateFreezeRow, freeze.candidate_hash)
            if row is None:
                session.add(
                    FactorCandidateFreezeRow(
                        candidate_hash=freeze.candidate_hash,
                        formula=list(freeze.formula),
                        dsl_version=freeze.dsl_version,
                        feature_set_version=freeze.feature_set_version,
                        discovery_snapshot_id=freeze.discovery_snapshot_id,
                        final_oos_snapshot_id=freeze.final_oos_snapshot_id,
                        candidate_frozen_at=freeze.candidate_frozen_at,
                        extra=extra,
                    )
                )

    def get_freeze(self, candidate_hash: str) -> CandidateFreeze | None:
        with self._sessions() as session:
            row = session.get(FactorCandidateFreezeRow, candidate_hash)
            if row is None:
                return None
            extra = dict(row.extra or {})
            return CandidateFreeze(
                candidate_hash=row.candidate_hash,
                formula=list(row.formula or []),
                dsl_version=row.dsl_version,
                feature_set_version=row.feature_set_version,
                discovery_snapshot_id=row.discovery_snapshot_id,
                final_oos_snapshot_id=row.final_oos_snapshot_id,
                candidate_frozen_at=row.candidate_frozen_at,
                experiment_id=extra.get("experiment_id"),
                discovery_config_hash=extra.get("discovery_config_hash"),
                selection_policy_version=extra.get("selection_policy_version"),
                selection_rank=extra.get("selection_rank"),
                research_code_version=extra.get("research_code_version"),
                selected_at=extra.get("selected_at"),
                factor_code_hash=extra.get("factor_code_hash"),
                universe_snapshot_id=extra.get("universe_snapshot_id"),
                feature_normalization_version=extra.get("feature_normalization_version"),
                selection_config_hash=extra.get("selection_config_hash"),
                discovery_metrics_hash=extra.get("discovery_metrics_hash"),
                candidate_count=extra.get("candidate_count"),
                code_sha=extra.get("code_sha"),
                config_hash=extra.get("config_hash"),
            )

    def record_evaluation(self, candidate_hash: str, discovery_snapshot_id: str, metrics: dict) -> None:
        with self._sessions.begin() as session:
            if session.scalar(
                select(FactorFinalOosEvaluationRow).where(
                    FactorFinalOosEvaluationRow.candidate_hash == candidate_hash,
                    FactorFinalOosEvaluationRow.discovery_snapshot_id == discovery_snapshot_id,
                )
            ):
                return
            freeze = session.get(FactorCandidateFreezeRow, candidate_hash)
            session.add(
                FactorFinalOosEvaluationRow(
                    candidate_hash=candidate_hash,
                    discovery_snapshot_id=discovery_snapshot_id,
                    final_oos_snapshot_id=freeze.final_oos_snapshot_id if freeze else "",
                    metrics=metrics,
                )
            )

    def get_evaluation(self, candidate_hash: str, discovery_snapshot_id: str) -> dict | None:
        with self._sessions() as session:
            row = session.scalar(
                select(FactorFinalOosEvaluationRow).where(
                    FactorFinalOosEvaluationRow.candidate_hash == candidate_hash,
                    FactorFinalOosEvaluationRow.discovery_snapshot_id == discovery_snapshot_id,
                )
            )
            return dict(row.metrics) if row else None

    def invalidate_oos_window(self, candidate_hash: str, reason: str) -> None:
        with self._sessions.begin() as session:
            row = session.get(FactorCandidateFreezeRow, candidate_hash)
            if row is not None:
                row.oos_window_status = "INVALIDATED"
                row.oos_invalidation_reason = reason[:120]

    def oos_window_status(self, candidate_hash: str) -> str:
        with self._sessions() as session:
            row = session.get(FactorCandidateFreezeRow, candidate_hash)
            return row.oos_window_status if row else "SEALED"
