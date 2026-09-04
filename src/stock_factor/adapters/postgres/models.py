from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class FactorRow(Base):
    __tablename__ = "factor_definition"
    factor_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    rpn: Mapped[list[str]] = mapped_column(JSON)
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    candidate_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class FactorJobRow(Base):
    __tablename__ = "factor_job"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", index=True)
    stage: Mapped[str] = mapped_column(String(40), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    request: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    # §33 Idempotency-Key：避免重试产生重复挖掘任务
    idempotency_key: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# Research V2 is intentionally normalised.  ``FactorRow.metrics`` remains a
# read-model for the API, while these records are the replayable authority.
class FactorCandidateRow(Base):
    __tablename__ = "factor_candidate"
    candidate_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    candidate_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    mining_job_id: Mapped[str | None] = mapped_column(String(64), index=True)
    parent_candidate_id: Mapped[str | None] = mapped_column(String(64), index=True)
    generation_round: Mapped[int] = mapped_column(Integer, default=1)
    generation_strategy: Mapped[str | None] = mapped_column(String(80))
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    formula: Mapped[list[str]] = mapped_column(JSON, default=list)
    canonical_formula: Mapped[str] = mapped_column(Text)
    feedback: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FactorVersionRow(Base):
    __tablename__ = "factor_version"
    factor_version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    factor_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer)
    formula: Mapped[list[str]] = mapped_column(JSON, default=list)
    canonical_formula: Mapped[str] = mapped_column(Text)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    research_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FactorStatisticalTestRow(Base):
    __tablename__ = "factor_statistical_test"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_id: Mapped[str] = mapped_column(String(64), index=True)
    experiment_id: Mapped[str] = mapped_column(String(64), index=True)
    raw_p_value: Mapped[float | None] = mapped_column(Float)
    adjusted_p_value: Mapped[float | None] = mapped_column(Float)
    q_value: Mapped[float | None] = mapped_column(Float)
    pbo: Mapped[float | None] = mapped_column(Float)
    effective_trials: Mapped[int] = mapped_column(Integer, default=0)
    passed_multiple_testing: Mapped[bool] = mapped_column(default=False)
    passed_pbo: Mapped[bool] = mapped_column(default=False)
    method: Mapped[str] = mapped_column(String(80), default="multiple_testing_v1")
    data_snapshot_id: Mapped[str | None] = mapped_column(String(128))
    # §86：Experiment 快照引用（统计验证只在 Discovery 窗口）
    discovery_snapshot_id: Mapped[str | None] = mapped_column(String(128))
    final_oos_snapshot_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FactorFinalOosRow(Base):
    __tablename__ = "factor_final_oos"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    factor_id: Mapped[str] = mapped_column(String(64), index=True)
    factor_version: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    data_snapshot_id: Mapped[str | None] = mapped_column(String(128))
    discovery_snapshot_id: Mapped[str | None] = mapped_column(String(128))
    final_oos_snapshot_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FactorCandidateFreezeRow(Base):
    """§13.3 Candidate Freeze：进入 Final OOS 前的不可变冻结记录。"""

    __tablename__ = "factor_candidate_freeze"
    candidate_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    formula: Mapped[list[str]] = mapped_column(JSON, default=list)
    dsl_version: Mapped[str] = mapped_column(String(40), default="factor-dsl.v1")
    feature_set_version: Mapped[str] = mapped_column(String(80), default="")
    discovery_snapshot_id: Mapped[str] = mapped_column(String(128))
    final_oos_snapshot_id: Mapped[str] = mapped_column(String(128))
    candidate_frozen_at: Mapped[str] = mapped_column(String(64))
    # SEALED / INVALIDATED（OOS 结果被反馈进搜索后置为 INVALIDATED，§13.4）
    oos_window_status: Mapped[str] = mapped_column(String(20), default="SEALED", index=True)
    oos_invalidation_reason: Mapped[str | None] = mapped_column(String(128))
    # 详细修改方案 P0-4：完整 freeze 证据（factor_code_hash/discovery_metrics_hash/candidate_count 等）。
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class OosAuthorizationRow(Base):
    """详细修改方案 P0-2：Final OOS 数据库级一次性授权。

    AUTHORIZED -> CONSUMED 必须在同一事务内完成（SELECT ... FOR UPDATE），
    保证并发 worker 不可能双重消费。
    """

    __tablename__ = "oos_authorizations"
    authorization_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: f"oosa-{uuid4().hex[:12]}"
    )
    experiment_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    final_oos_snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    candidate_set_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="AUTHORIZED", index=True)
    dataset_ref_hash: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    market_snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    content_ref_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    active_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    authorized_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class OosEvaluationRunRow(Base):
    __tablename__ = "oos_evaluation_runs"
    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    authorization_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("oos_authorizations.authorization_id"), nullable=False, index=True
    )
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False)
    fencing_token: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="STARTED", index=True)
    lease_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(128), nullable=False, default="final-oos-v1")
    cohort_artifact_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OosCandidateCheckpointRow(Base):
    __tablename__ = "oos_candidate_checkpoints"
    checkpoint_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("oos_evaluation_runs.run_id"), nullable=False, index=True
    )
    candidate_id: Mapped[str] = mapped_column(String(128), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    result_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    __table_args__ = (UniqueConstraint("run_id", "candidate_id", name="uq_oos_checkpoint_run_candidate"),)


class OosCohortArtifactRow(Base):
    __tablename__ = "oos_cohort_artifacts"
    artifact_hash: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("oos_evaluation_runs.run_id"), nullable=False, unique=True
    )
    authorization_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("oos_authorizations.authorization_id"), nullable=False, index=True
    )
    artifact: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ResearchArtifactRow(Base):
    """Immutable content-addressed ResearchArtifactV2 evidence."""

    __tablename__ = "research_artifacts_v2"
    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    contract_version: Mapped[str] = mapped_column(String(80), nullable=False)
    artifact_status: Mapped[str] = mapped_column(String(20), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FactorSetRow(Base):
    """详细修改方案 §11/§17：FactorSet 正式版本身份。"""

    __tablename__ = "factor_sets"
    factor_set_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    factor_set_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    research_experiment_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    promotion_policy_version: Mapped[str] = mapped_column(String(80), default="promotion_gate_v2")
    valid_from: Mapped[str | None] = mapped_column(String(64), nullable=True)
    valid_to: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE", index=True)
    code_sha: Mapped[str | None] = mapped_column(String(128), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    research_artifact_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    formal_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FactorSetMemberRow(Base):
    """详细修改方案 §17：FactorSet 成员（factor + version + weight）。"""

    __tablename__ = "factor_set_members"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    factor_set_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    factor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    factor_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class FactorFinalOosEvaluationRow(Base):
    """§13.4 一次性 Final OOS 评估记录。"""

    __tablename__ = "factor_final_oos_evaluation"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_hash: Mapped[str] = mapped_column(String(64), index=True)
    discovery_snapshot_id: Mapped[str] = mapped_column(String(128), index=True)
    final_oos_snapshot_id: Mapped[str] = mapped_column(String(128))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FactorOosAuditRow(Base):
    __tablename__ = "factor_oos_audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    factor_id: Mapped[str] = mapped_column(String(64), index=True)
    factor_version: Mapped[int] = mapped_column(Integer)
    audit_status: Mapped[str] = mapped_column(String(20), index=True)
    violations: Mapped[list[str]] = mapped_column(JSON, default=list)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
    audit_version: Mapped[str] = mapped_column(String(80), default="final_oos_audit_v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FactorPromotionDecisionRow(Base):
    __tablename__ = "factor_promotion_decision"
    decision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    factor_id: Mapped[str] = mapped_column(String(64), index=True)
    factor_version: Mapped[int] = mapped_column(Integer)
    mining_job_id: Mapped[str | None] = mapped_column(String(64), index=True)
    data_snapshot_id: Mapped[str | None] = mapped_column(String(128))
    passed: Mapped[bool] = mapped_column(default=False)
    failed_rules: Mapped[list[str]] = mapped_column(JSON, default=list)
    metrics_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    gate_version: Mapped[str] = mapped_column(String(80), default="v2")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FactorLifecycleEventRow(Base):
    __tablename__ = "factor_lifecycle_event"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    factor_id: Mapped[str] = mapped_column(String(64), index=True)
    factor_version: Mapped[int] = mapped_column(Integer)
    from_status: Mapped[str | None] = mapped_column(String(30))
    to_status: Mapped[str] = mapped_column(String(30))
    reason: Mapped[str] = mapped_column(Text)
    metrics_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    data_snapshot_id: Mapped[str | None] = mapped_column(String(128))
    actor: Mapped[str] = mapped_column(String(80), default="factor-miner")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
