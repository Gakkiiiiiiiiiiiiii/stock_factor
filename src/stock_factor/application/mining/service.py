from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import yaml

from stock_factor.application.artifacts.assemble import assemble_research_artifact
from stock_factor.application.experiments.statistics import cohort_statistics as build_cohort_statistics
from stock_factor.application.experiments.statistics import discovery_gates as build_discovery_gates
from stock_factor.application.finalist_selection import select_finalists
from stock_factor.application.market_snapshot_validator import validate_formal_market_snapshot
from stock_factor.application.mining.discovery import evaluate_discovery
from stock_factor.application.mining.generate import generate_candidates, model_candidates, mutate_candidate
from stock_factor.application.mining.screen import canonical_candidates, correlation_deduplicate, feedback
from stock_factor.application.oos.evaluate import (
    FinalOosAuthorizationError,
    FinalOosDataProvider,
    FinalOosEvaluationService,
    InMemoryCandidateSealStore,
)
from stock_factor.application.oos.schedule import schedule_final_oos
from stock_factor.application.panel import build_feature_panel
from stock_factor.application.promotion.gate import evaluate_candidate_promotion
from stock_factor.application.readiness import ReadinessService
from stock_factor.application.seed_library import Alpha191SeedLibrary
from stock_factor.application.tradability_assessment import TradabilityAssessmentService
from stock_factor.config.schema import CONFIG_ROOT
from stock_factor.domain.content_signal_v5 import FormalContentQuery, FormalContentRef
from stock_factor.domain.experiment import (
    COMPLETED,
    DISCOVERY_COMPLETED,
    DISCOVERY_RUNNING,
    FINALIST_SELECTED,
    FROZEN,
    OOS_INVALIDATED,
    SELECTION_POLICY_VERSION,
    ResearchExperiment,
)
from stock_factor.domain.factor import FactorDefinition
from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef
from stock_factor.domain.tradability_artifact import ExecutionCostCalibrationRef
from stock_factor.engine.diagnostics import (
    compute_capacity_proxy,
    compute_factor_autocorrelation,
    compute_ic_decay,
    compute_turnover,
)
from stock_factor.engine.exposure import compute_factor_exposures
from stock_factor.engine.fitness import evaluate_factor_range
from stock_factor.engine.oos_audit import audit_final_oos
from stock_factor.engine.oos_seal import (
    DSL_VERSION,
    CandidateFreeze,
    OosWindowInvalidatedError,
    derive_snapshot_refs,
    feature_set_version,
)
from stock_factor.ports.oos_run_repository import OosCheckpointConflict, OosIdentityError
from stock_factor.ports.providers import (
    ContentSignalProvider,
    FactorRepository,
    MarketDataProvider,
    ModelClient,
)
from stock_factor.research_config import get_research_config


class ResearchIntegrityError(ValueError):
    """Raised before persisting a candidate whose identity no longer matches its formula."""


class FactorMiningService:
    def __init__(
        self,
        market: MarketDataProvider,
        content: ContentSignalProvider,
        factors: FactorRepository,
        model: ModelClient | None = None,
        seeds: Alpha191SeedLibrary | None = None,
        final_oos_service: FinalOosEvaluationService | None = None,
        expected_execution_cost_calibration: ExecutionCostCalibrationRef | dict | None = None,
        research_artifact_service=None,
        readiness_service: ReadinessService | None = None,
    ) -> None:
        self._market, self._content, self._factors, self._model = market, content, factors, model
        self._seeds = seeds or Alpha191SeedLibrary()
        # Final OOS 必须独立于候选搜索（集成文档 §13/§78；收尾文档 §14-§25）。
        self._oos_service = final_oos_service or FinalOosEvaluationService(InMemoryCandidateSealStore())
        self._oos_data = FinalOosDataProvider()
        self._tradability = TradabilityAssessmentService()
        self._expected_execution_cost_calibration = expected_execution_cost_calibration
        self._research_artifact_service = research_artifact_service
        self._readiness = readiness_service

    @staticmethod
    def _artifact_contract_checksums() -> dict[str, str]:
        root = CONFIG_ROOT.parent
        manifest_path = root / "contracts" / "platform-manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        entries = {entry.get("name"): entry for entry in manifest.get("contracts", [])}
        required = {"research-artifact.v2", "factor.v1", "market-snapshot.v1", "content-factor-signal.v5.1"}
        if not required.issubset(entries):
            raise ValueError("formal artifact contract inventory is incomplete")
        checksums: dict[str, str] = {}
        for name in required:
            entry = entries[name]
            path = root / str(entry["schema"])
            if not path.is_file():
                raise ValueError(f"formal artifact contract schema is missing: {name}")
            actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != entry.get("checksum"):
                raise ValueError(f"formal artifact contract checksum mismatch: {name}")
            checksums[name] = actual
        return checksums

    @staticmethod
    def _artifact_dependency_lock_hash() -> str:
        configured = os.getenv("FACTOR_DEPENDENCY_LOCK_HASH", "").strip()
        if not configured:
            raise ValueError("formal artifact requires FACTOR_DEPENDENCY_LOCK_HASH")
        return configured

    def _candidates(self, request: dict) -> list[dict]:
        return generate_candidates(request, seeds=self._seeds, model=self._model)

    def _model_candidates(self, feedback: dict | None = None, previous: list[dict] | None = None) -> list[dict]:
        return model_candidates(self._model, feedback, previous)

    @staticmethod
    def _feedback(evaluated: list[dict]) -> dict:
        return feedback(evaluated)

    @staticmethod
    def _recent_alpha(values: np.ndarray, close: np.ndarray, horizon: int) -> dict:
        # 只能在 Discovery 窗口内计算（§13.2：Recent Alpha 不得引用 Final OOS）；
        # 调用方必须在传入前截断面板。
        start = max(0, values.shape[1] - max(20, horizon * 4))
        metrics = evaluate_factor_range(values, close, start, values.shape[1] - horizon, horizon=horizon)
        return {
            "recent_rank_ic": metrics.get("rank_ic"),
            "recent_icir": metrics.get("icir"),
            "recent_topk_excess": metrics.get("research_excess_return_proxy"),
            "recent_hit_rate": metrics.get("positive_window_ratio"),
            "recent_coverage": metrics.get("coverage"),
            "recent_decay": compute_ic_decay(values[:, start:], close[:, start:], max_horizon=min(5, horizon)),
            "recent_turnover": compute_turnover(values[:, start:], top_k=max(5, int(values.shape[0] * 0.01))),
            "passed": bool(metrics.get("passed")) and float(metrics.get("rank_ic") or 0.0) > 0,
        }

    @staticmethod
    def _canonical_candidates(candidates: list[dict], budget: int) -> list[dict]:
        return canonical_candidates(candidates, budget)

    @staticmethod
    def _diagnostics(values: np.ndarray, panel: dict) -> tuple[dict, dict, dict]:
        volume = np.asarray(panel.get("volume"), dtype=float)
        latest = values[:, -1] if values.ndim == 2 and values.shape[1] else np.asarray([])
        liquidity = volume[:, -1] if volume.ndim == 2 and volume.shape[1] else None
        diagnostics = {
            "autocorrelation_lag_1": compute_factor_autocorrelation(values),
            "ic_decay": compute_ic_decay(values, np.asarray(panel["close"], dtype=float), max_horizon=5),
            "turnover": compute_turnover(values, top_k=max(5, int(values.shape[0] * 0.01))),
        }
        exposure = compute_factor_exposures(latest, liquidity=liquidity)
        capacity = compute_capacity_proxy(volume)
        return diagnostics, exposure, capacity

    @staticmethod
    def _mutate(candidate: dict, feedback: dict, generation_round: int) -> dict | None:
        return mutate_candidate(candidate, feedback, generation_round)

    @staticmethod
    def _correlation_deduplicate(evaluated: list[dict], threshold: float = 0.995) -> tuple[list[dict], list[dict]]:
        return correlation_deduplicate(evaluated, threshold)

    def run(self, request: dict, progress=lambda stage, value: None) -> dict:
        symbols = request.get("symbols") or []
        if not symbols:
            raise ValueError("symbols must not be empty")
        end = request.get("end") or datetime.now(UTC).date().isoformat()
        start = (
            request.get("start")
            or (datetime.fromisoformat(end) - timedelta(days=int(request.get("days") or 365) * 2)).date().isoformat()
        )
        progress("data", 10)
        research_mode = str(request.get("research_mode", "EXPLORATORY")).upper()
        if research_mode not in {"FORMAL", "EXPLORATORY"}:
            raise ValueError("research_mode must be FORMAL or EXPLORATORY")
        formal_ref = None
        formal_content_query = None
        formal_content_ref = None
        if research_mode == "FORMAL":
            ref_payload = request.get("formal_market_ref")
            if not isinstance(ref_payload, dict):
                raise ValueError("FORMAL research requires a complete formal_market_ref")
            formal_ref = FormalMarketDatasetRef.from_payload(ref_payload)
            query_payload = request.get("formal_content_query")
            content_payload = request.get("formal_content_ref")
            if not isinstance(query_payload, dict) or not isinstance(content_payload, dict):
                raise ValueError("FORMAL research requires formal_content_query and formal_content_ref")
            formal_content_query = FormalContentQuery.model_validate(query_payload)
            formal_content_ref = FormalContentRef.model_validate(content_payload)
            expected_content_ref = FormalContentRef.from_query(formal_content_query, formal_content_ref.manifest_hash)
            if formal_content_ref != expected_content_ref:
                raise ValueError("formal content ref does not match query")
            validate_formal_market_snapshot(
                formal_ref,
                requested_start=start,
                requested_end=end,
                as_of=formal_content_query.availability_as_of,
            )
            snapshot = self._market.get_daily_bars(
                symbols,
                start,
                end,
                "qfq",
                formal_market_ref=formal_ref,
                as_of=formal_content_query.availability_as_of,
            )
            if snapshot.formal_market_ref is None or snapshot.formal_market_ref.ref_hash != formal_ref.ref_hash:
                raise ValueError("formal market provider returned a different market snapshot ref")
            calibration_payload = request.get("execution_cost_calibration")
            expected_calibration_payload = self._expected_execution_cost_calibration
            if not isinstance(calibration_payload, dict) or expected_calibration_payload is None:
                raise ValueError("FORMAL research requires registered execution cost calibration and expected identity")
            calibration = ExecutionCostCalibrationRef.from_payload(calibration_payload)
            expected_calibration = (
                expected_calibration_payload
                if isinstance(expected_calibration_payload, ExecutionCostCalibrationRef)
                else ExecutionCostCalibrationRef.from_payload(expected_calibration_payload)
            )
            if calibration != expected_calibration:
                raise ValueError("execution cost calibration does not match registered expected identity")
            signals = self._content.load_signals(
                symbols,
                start,
                end,
                query=formal_content_query,
                expected_ref=formal_content_ref,
            )
        else:
            snapshot = self._market.get_daily_bars(symbols, start, end, "qfq")
            signals = self._content.load_signals(symbols, start, end)
        panel = build_feature_panel(snapshot, signals)
        panel_feature_version = feature_set_version(list(panel.keys()))
        config = get_research_config()
        budget = max(1, min(int(request.get("candidate_budget") or 50), 200))
        candidates = self._canonical_candidates(self._candidates(request), budget)
        horizon = int(request.get("horizon") or config.evaluation.horizon_days)
        explicit_candidates = bool(request.get("candidates") or request.get("use_model"))
        round_limit = max(1, min(int(request.get("rounds") or (1 if explicit_candidates else 3)), 10))
        candidates_per_round = max(1, int(request.get("candidates_per_round") or len(candidates)))
        evaluated, search_rounds = evaluate_discovery(
            service=self,
            candidates=candidates,
            panel=panel,
            config=config,
            request=request,
            budget=budget,
            horizon=horizon,
            round_limit=round_limit,
            candidates_per_round=candidates_per_round,
            progress=progress,
        )

        evaluated, correlation_duplicates = self._correlation_deduplicate(evaluated)
        # §13.2：FDR / PBO / DSR 队列统计验证只能使用 Discovery 窗口的 rank-IC。
        cohort_statistics = build_cohort_statistics(evaluated, panel["close"], horizon)
        # 收尾文档 §15/§17：Discovery Gate 只使用 discovery 证据，签名无 final_oos。
        discovery_gates = build_discovery_gates(evaluated, cohort_statistics)
        # §19：ResearchExperiment 状态机贯穿整个研究流程。
        finalist_count = max(1, min(int(request.get("finalist_count") or 1), 3))
        experiment_config_hash = hashlib.sha256(
            json.dumps(
                {
                    "symbols": symbols,
                    "start": start,
                    "end": end,
                    "horizon": horizon,
                    "candidate_budget": budget,
                    "round_limit": round_limit,
                    "finalist_count": finalist_count,
                    "selection_policy_version": SELECTION_POLICY_VERSION,
                    "market_ref_hash": formal_ref.ref_hash if formal_ref else None,
                    "content_ref_hash": formal_content_ref.ref_hash if formal_content_ref else None,
                    "content_manifest": formal_content_ref.model_dump(mode="json") if formal_content_ref else None,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        market_ref_hash = snapshot.formal_market_ref.ref_hash if snapshot.formal_market_ref else None
        content_ref_hash = formal_content_ref.ref_hash if formal_content_ref else None
        experiment_id = request.get("experiment_id") or (
            "exp-" + hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()[:16]
        )
        experiment = ResearchExperiment(
            symbols=symbols,
            experiment_id=experiment_id,
            market_ref_hash=market_ref_hash,
            content_ref_hash=content_ref_hash,
            content_manifest=(formal_content_ref.model_dump(mode="json") if formal_content_ref else None),
            config_hash=experiment_config_hash,
            candidate_budget=budget,
            round_limit=round_limit,
            finalist_count=finalist_count,
            readiness_evidence=(request.get("readiness_evidence") if research_mode == "FORMAL" else None),
            readiness_evidence_hash=(request.get("readiness_evidence_hash") if research_mode == "FORMAL" else None),
            readiness_frozen_at=(request.get("readiness_frozen_at") if research_mode == "FORMAL" else None),
            readiness_threshold_version=(
                request.get("readiness_threshold_version") if research_mode == "FORMAL" else None
            ),
        )
        experiment.transition(DISCOVERY_RUNNING)
        experiment.transition(DISCOVERY_COMPLETED)
        # §18：Finalist Selection 在 Final OOS 之前，默认唯一 Primary Candidate。
        finalists = select_finalists(evaluated, discovery_gates, finalist_count) if evaluated else []
        finalist_hashes = {item["candidate"]["candidate_hash"] for item in finalists}
        experiment.transition(FINALIST_SELECTED)
        selected_at = datetime.now(UTC).isoformat(timespec="seconds")

        refs = None
        sealed_cohort_artifact_hash = None
        if finalists and finalists[0]["split"] is not None:
            primary_split = finalists[0]["split"]
            refs = derive_snapshot_refs(
                snapshot.data_snapshot_id,
                (primary_split.discovery_start, primary_split.discovery_end),
                (primary_split.final_oos_start, primary_split.final_oos_end),
            )
            experiment.discovery_snapshot_id = refs.discovery_snapshot_id
            experiment.final_oos_snapshot_id = refs.final_oos_snapshot_id

        if finalists:
            # §20：Candidate Freeze（扩展字段）——全部 finalist 在 OOS 前冻结。
            frozen_items: list[dict] = []
            for item in finalists:
                if item["split"] is None:
                    continue
                item_refs = refs or derive_snapshot_refs(
                    snapshot.data_snapshot_id,
                    (item["split"].discovery_start, item["split"].discovery_end),
                    (item["split"].final_oos_start, item["split"].final_oos_end),
                )
                item["freeze"] = CandidateFreeze(
                    candidate_hash=item["candidate"]["candidate_hash"],
                    formula=list(item["candidate"]["rpn"]),
                    dsl_version=DSL_VERSION,
                    feature_set_version=panel_feature_version,
                    discovery_snapshot_id=item_refs.discovery_snapshot_id,
                    final_oos_snapshot_id=item_refs.final_oos_snapshot_id,
                    experiment_id=experiment.experiment_id,
                    discovery_config_hash=experiment_config_hash,
                    selection_policy_version=SELECTION_POLICY_VERSION,
                    selection_rank=item.get("selection_rank"),
                    research_code_version=experiment.code_version,
                    selected_at=selected_at,
                )
                self._oos_service.freeze_candidate(item["freeze"])
                frozen_items.append(item)

        if finalists and frozen_items:
            experiment.transition(FROZEN)
            # §23：FROZEN → OOS_AUTHORIZED 后才允许加载 Final OOS 数据。
            if research_mode == "FORMAL" and self._readiness is not None:
                # Revalidate at the last safe point before authorization; no
                # Final OOS data is loaded until this immutable evidence still
                # matches the admission decision.
                self._readiness.revalidate_oos(request, request.get("readiness_evidence_hash"))
            experiment.authorize_oos()
            try:
                # P0-2/P0 F-01：schedule owns the single freeze→authorize→cohort path.
                final_dataset_ref, cohort_evidence = schedule_final_oos(
                    oos_service=self._oos_service,
                    oos_data=self._oos_data,
                    experiment=experiment,
                    frozen_items=frozen_items,
                    snapshot=snapshot,
                    panel=panel,
                    start=start,
                    content_ref=formal_content_ref,
                    formal_ref=formal_ref,
                    research_mode=research_mode,
                    feature_set_version_value=panel_feature_version,
                    experiment_config_hash=experiment_config_hash,
                    horizon=horizon,
                )
                cohort_results = cohort_evidence.results
                sealed_cohort_artifact_hash = cohort_evidence.cohort_artifact_hash
                for item, metrics in zip(frozen_items, cohort_results):
                    item["final_oos"] = metrics
            except OosWindowInvalidatedError:
                # §24：OOS 区间失效 => 本次实验 OOS_INVALIDATED。
                for item in frozen_items:
                    item["final_oos"] = {"passed": False, "reason": "OOS_WINDOW_INVALIDATED"}
                experiment.transition(OOS_INVALIDATED)
            except FinalOosAuthorizationError:
                # 授权缺失/已消费是 terminal；不得继续走候选接受/晋升路径。
                for item in frozen_items:
                    item["final_oos"] = {"passed": False, "reason": "OOS_WINDOW_INVALIDATED"}
                if experiment.status == "OOS_AUTHORIZED":
                    experiment.transition(OOS_INVALIDATED)
            except (OosIdentityError, OosCheckpointConflict):
                for item in frozen_items:
                    item["final_oos"] = {"passed": False, "reason": "OOS_WINDOW_INVALIDATED"}
                experiment.transition(OOS_INVALIDATED)
            for item in finalists:
                if item["split"] is None:
                    item["final_oos"] = {"passed": False, "reason": "INSUFFICIENT_RESEARCH_HISTORY"}

        accepted = []
        for index, item in enumerate(evaluated):
            candidate = item["candidate"]
            rpn = list(candidate["rpn"])
            if hashlib.sha256(" ".join(rpn).encode()).hexdigest() != candidate["candidate_hash"]:
                raise ResearchIntegrityError("candidate_hash does not match the candidate RPN")
            values = item["values"]
            preliminary = item["preliminary"]
            walkforward = item["walkforward"]
            final_oos = item["final_oos"]
            diagnostics = item["diagnostics"]
            exposure = item["exposure"]
            capacity = item["capacity"]
            recent_alpha = item["recent_alpha"]
            split = item["split"]
            freeze = item.get("freeze")
            is_finalist = candidate["candidate_hash"] in finalist_hashes
            statistics = cohort_statistics[candidate["candidate_hash"]]
            discovery_gate = discovery_gates[candidate["candidate_hash"]]
            oos_audit = audit_final_oos(
                split=split,
                final_oos=final_oos,
                data_snapshot_id=snapshot.data_snapshot_id,
            )
            factor_id = uuid4().hex
            tradability_flags = {
                key: snapshot.bars[key]
                for key in (
                    "tradable",
                    "is_tradable",
                    "can_trade",
                    "halted",
                    "is_halted",
                    "suspended",
                    "is_suspended",
                    "limit",
                    "limit_hit",
                    "limit_up",
                    "limit_down",
                )
                if key in snapshot.bars
            }
            tradability = self._tradability.assess(
                factor_artifact_id=factor_id,
                market_snapshot_id=snapshot.data_snapshot_id,
                factor_values=values,
                closes=panel["close"],
                amount=panel.get("amount"),
                volume=panel.get("volume"),
                execution_cost_calibration=request.get("execution_cost_calibration"),
                expected_execution_cost_calibration=self._expected_execution_cost_calibration,
                tradability_flags=tradability_flags,
                market_snapshot_ref=formal_ref,
                formal=research_mode == "FORMAL",
                neutralization={"available": bool(exposure), "exposure": exposure},
            )
            tradability_payload = tradability.to_dict()
            promotion = evaluate_candidate_promotion(
                walkforward=walkforward,
                statistics=statistics,
                final_oos=final_oos,
                oos_audit=oos_audit,
                diagnostics=diagnostics,
                exposure=exposure,
                capacity=capacity,
                recent_alpha=recent_alpha,
                tradability_assessment=tradability_payload,
                formal_research=research_mode == "FORMAL",
                data_snapshot_id=snapshot.data_snapshot_id,
            )
            # 收尾文档 §27：Factor 状态不再直接进入 PAPER_TRADING（Paper Authority 属于 Quant）。
            # 详细修改方案 §10：PAPER_ELIGIBLE 弃用 -> OOS_PASSED（交易执行已迁往 Quant）。
            if is_finalist:
                if final_oos and final_oos.get("passed") and promotion["passed"]:
                    status = "OOS_PASSED"
                elif final_oos and final_oos.get("passed"):
                    status = "DISCOVERY_PASSED"
                else:
                    status = "OOS_FAILED"
            elif discovery_gate["passed"]:
                status = "DISCOVERY_PASSED"
            else:
                status = "DISCOVERY_CANDIDATE"
            freeze_dict = freeze.to_dict() if isinstance(freeze, CandidateFreeze) else freeze
            metrics = {
                "in_sample": preliminary,
                "walkforward": walkforward,
                "final_oos": final_oos,
                "oos_audit": oos_audit,
                "statistics": statistics,
                "diagnostics": diagnostics,
                "exposure": exposure,
                "capacity": capacity,
                "tradability_assessment": tradability_payload,
                "tradability_assessment_hash": tradability.artifact_id,
                "tradability_policy_version": tradability.policy_version,
                "tradability_policy_hash": tradability.policy_hash,
                "capacity_artifact_id": tradability.capacity_artifact.artifact_id
                if tradability.capacity_artifact
                else None,
                "recent_alpha": recent_alpha,
                "promotion_gate": promotion,
                "discovery_gate": discovery_gate,
                "is_finalist": is_finalist,
                "selection_rank": item.get("selection_rank"),
                "research_split": split.diagnostics(horizon, values.shape[1]) if split else None,
                "data_version": snapshot.data_version,
                "data_snapshot_id": snapshot.data_snapshot_id,
                # §86：Experiment 必须分别保存 discovery / final OOS 快照引用。
                "discovery_snapshot_id": freeze_dict["discovery_snapshot_id"]
                if freeze_dict
                else (refs.discovery_snapshot_id if refs else None),
                "final_oos_snapshot_id": freeze_dict["final_oos_snapshot_id"]
                if freeze_dict
                else (refs.final_oos_snapshot_id if refs else None),
                "candidate_frozen_at": freeze_dict["candidate_frozen_at"] if freeze_dict else None,
                "dsl_version": DSL_VERSION,
                "feature_set_version": panel_feature_version,
                "candidate_search_window": "discovery_only",
                "generation_round": candidate.get("generation_round", 1),
                "parent_candidate_id": candidate.get("parent_candidate_hash"),
                "generation_strategy": candidate.get("generation_strategy", "seed"),
                "generation_feedback": candidate.get("generation_feedback") or {},
                # 收尾文档 §19/§39：实验血缘。
                "research_experiment_id": experiment.experiment_id,
                "final_oos_dataset_ref_hash": experiment.final_oos_dataset_ref_hash,
                "market_ref_hash": market_ref_hash,
                "content_ref_hash": formal_content_ref.ref_hash if formal_content_ref else None,
                "content_snapshot_id": formal_content_query.content_snapshot_id if formal_content_query else None,
                "content_policy_version": formal_content_query.signal_policy_version if formal_content_query else None,
                "content_manifest": formal_content_ref.model_dump(mode="json") if formal_content_ref else None,
            }
            definition = FactorDefinition(
                factor_id=factor_id,
                name=candidate.get("name") or candidate["candidate_hash"][:12],
                rpn=rpn,
                hypothesis=candidate.get("hypothesis", ""),
                status=status,
                metrics=metrics,
                candidate_hash=candidate["candidate_hash"],
            )
            accepted.append(self._factors.save(definition))
            progress("promotion", 75 + int((index + 1) * 20 / max(len(evaluated), 1)))
        if experiment.status not in {COMPLETED, "FAILED", OOS_INVALIDATED}:
            experiment.transition(COMPLETED)
        research_artifact_id = None
        if (
            research_mode == "FORMAL"
            and self._research_artifact_service is not None
            and experiment.final_oos_dataset_ref
        ):
            if sealed_cohort_artifact_hash:
                artifact = assemble_research_artifact(
                    experiment=experiment,
                    request=request,
                    accepted=accepted,
                    frozen_items=frozen_items,
                    formal_ref=formal_ref,
                    formal_content_ref=formal_content_ref,
                    cohort_statistics=cohort_statistics,
                    cohort_evidence=cohort_evidence,
                    contract_checksums=self._artifact_contract_checksums(),
                    dependency_lock_hash=self._artifact_dependency_lock_hash(),
                    producer_commit=os.getenv("FACTOR_GIT_COMMIT", "").strip(),
                )
                research_artifact_id = self._research_artifact_service.seal(artifact).artifact_id
        return {
            "factor_count": len(accepted),
            "factors": accepted,
            "data_version": snapshot.data_version,
            "data_snapshot_id": snapshot.data_snapshot_id,
            "content_signal_count": len(signals),
            "search_rounds": search_rounds,
            "correlation_duplicates": correlation_duplicates,
            "experiment": experiment.to_dict(),
            "finalists": sorted(finalist_hashes),
            "finalist_count": finalist_count,
            "formal_eligible": research_mode == "FORMAL",
            "market_ref_hash": market_ref_hash,
            "content_ref_hash": formal_content_ref.ref_hash if formal_content_ref else None,
            "content_manifest": formal_content_ref.model_dump(mode="json") if formal_content_ref else None,
            "final_oos_dataset_ref_hash": experiment.final_oos_dataset_ref_hash,
            "research_artifact_id": research_artifact_id,
        }
