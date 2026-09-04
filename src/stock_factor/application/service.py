from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np

from stock_factor.application.market_snapshot_validator import validate_formal_market_snapshot
from stock_factor.application.mining.service import FactorMiningService
from stock_factor.application.panel import build_feature_panel
from stock_factor.application.readiness import ReadinessService
from stock_factor.config.runtime import RuntimeConfig
from stock_factor.domain.content_signal_v5 import FormalContentQuery, FormalContentRef
from stock_factor.domain.factor import FactorJob
from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef
from stock_factor.engine.alpha import compose_alpha_scores_with_evidence
from stock_factor.engine.fitness import evaluate_factor
from stock_factor.engine.vm import StackVM
from stock_factor.ports.providers import (
    ContentSignalProvider,
    FactorJobRepository,
    FactorRepository,
    MarketDataProvider,
)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class MarketDataUnavailableError(RuntimeError):
    """§90：Quant 市场数据不可用，禁止启动新 Mining。"""


class FactorApplication:
    def __init__(
        self,
        jobs: FactorJobRepository,
        factors: FactorRepository,
        mining: FactorMiningService,
        market: MarketDataProvider,
        content: ContentSignalProvider,
        paper: Any,
        runtime_config: RuntimeConfig | None = None,
        research_artifact_service=None,
        readiness_service: ReadinessService | None = None,
    ) -> None:
        self._jobs, self._factors, self._mining = jobs, factors, mining
        self._market, self._content, self._paper = market, content, paper
        self._runtime_config = runtime_config or RuntimeConfig.from_env()
        self.research_artifact_service = research_artifact_service
        self._readiness = readiness_service or ReadinessService(self._runtime_config)

    @property
    def runtime_config(self) -> RuntimeConfig:
        """The validated immutable configuration used to compose this app."""
        return self._runtime_config

    @property
    def readiness_service(self) -> ReadinessService:
        return self._readiness

    def create_mining_job(self, payload: dict, idempotency_key: str | None = None) -> dict:
        # Do not leak the generated job/experiment identity into the caller's
        # mutable request object; a second submission is a distinct job unless
        # the repository deduplicates it by idempotency key.
        payload = dict(payload)
        research_mode = str(payload.get("research_mode", "EXPLORATORY")).upper()
        if research_mode not in {"FORMAL", "EXPLORATORY"}:
            raise ValueError("research_mode must be FORMAL or EXPLORATORY")
        # Admission is deliberately before any market/content access.  A
        # failed formal admission therefore cannot create a job or trigger a
        # provider request.
        readiness = (
            self._readiness.admit_formal_mining(payload)
            if research_mode == "FORMAL"
            else self._readiness.research(model_requested=bool(payload.get("use_model")))
        )
        payload["readiness_evidence"] = readiness.to_dict()
        payload["readiness_evidence_hash"] = readiness.evidence_hash
        payload["readiness_frozen_at"] = readiness.frozen_at
        payload["readiness_threshold_version"] = readiness.threshold_version
        formal_ref = None
        formal_content_query = None
        formal_content_ref = None
        if research_mode == "FORMAL":
            ref_payload = payload.get("formal_market_ref")
            if not isinstance(ref_payload, dict):
                raise ValueError("FORMAL research requires a complete formal_market_ref")
            ref = FormalMarketDatasetRef.from_payload(ref_payload)
            formal_ref = ref
            query_payload = payload.get("formal_content_query")
            content_payload = payload.get("formal_content_ref")
            if not isinstance(query_payload, dict) or not isinstance(content_payload, dict):
                raise ValueError("FORMAL research requires formal_content_query and formal_content_ref")
            formal_content_query = FormalContentQuery.model_validate(query_payload)
            formal_content_ref = FormalContentRef.model_validate(content_payload)
            expected_content_ref = FormalContentRef.from_query(formal_content_query, formal_content_ref.manifest_hash)
            if formal_content_ref != expected_content_ref:
                raise ValueError("formal content ref does not match query")
            requested_start = payload.get("start") or ref.start
            requested_end = payload.get("end") or ref.end
            as_of = formal_content_query.availability_as_of
            validate_formal_market_snapshot(
                ref,
                requested_start=requested_start,
                requested_end=requested_end,
                as_of=as_of,
            )
            payload["formal_market_ref"] = {
                **ref_payload,
                "contract": ref.contract,
                "ref_hash": ref.ref_hash,
            }
            payload["formal_content_query"] = formal_content_query.model_dump(mode="json")
            payload["formal_content_ref"] = formal_content_ref.model_dump(mode="json")
            payload["formal_eligible"] = True
        else:
            payload["research_mode"] = "EXPLORATORY"
            payload["formal_eligible"] = False
        # §90：Quant 市场数据不可用时禁止启动新 Mining。
        symbols = list(payload.get("symbols") or [])
        if symbols:
            if formal_ref is not None:
                end = payload.get("end") or formal_ref.end
                start = payload.get("start") or formal_ref.start
                as_of = formal_content_query.availability_as_of
                snapshot = self._market.get_daily_bars(
                    symbols,
                    start,
                    end,
                    "qfq",
                    formal_market_ref=formal_ref,
                    as_of=as_of,
                )
                if snapshot.formal_market_ref is None or snapshot.formal_market_ref.ref_hash != formal_ref.ref_hash:
                    raise ValueError("formal market provider returned a different market snapshot ref")
                self._content.load_signals(
                    symbols,
                    start,
                    end,
                    query=formal_content_query,
                    expected_ref=formal_content_ref,
                )
            else:
                self._require_market_available(symbols)
        job_id = uuid4().hex
        if not payload.get("experiment_id"):
            # Job identity, rather than request content, isolates independent
            # submissions while preserving the id on a retried job record.
            payload["experiment_id"] = "exp-" + hashlib.sha256(job_id.encode()).hexdigest()[:16]
        job = FactorJob(job_id=job_id, request=payload, idempotency_key=idempotency_key)
        result = self._jobs.create(job).to_dict()
        result["formal_eligible"] = payload["formal_eligible"]
        return result

    def _require_market_available(self, symbols: list[str]) -> None:
        end = datetime.now(UTC).date().isoformat()
        start = (datetime.now(UTC).date() - timedelta(days=10)).isoformat()
        try:
            self._market.get_daily_bars(symbols[:1], start, end, "qfq")
        except Exception as exc:  # noqa: BLE001
            raise MarketDataUnavailableError(f"DATA_NOT_READY: quant market data unavailable: {exc}") from exc

    def get_mining_job(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def cancel_mining_job(self, job_id: str) -> dict | None:
        job = self._jobs.cancel(job_id)
        return {"job_id": job_id, "cancelled": job.status == "CANCELLED", "status": job.status} if job else None

    def process_next(self, worker_id: str, lease_seconds: int = 900) -> dict | None:
        job = self._jobs.claim_pending(worker_id, lease_seconds)
        if job is None:
            return None
        stage = "prepare"
        try:
            result = self._mining.run(job.request, lambda name, value: self._jobs.progress(job.job_id, name, value))
            result = _json_safe(result)
            self._jobs.succeed(job.job_id, result)
            return {"job_id": job.job_id, "status": "SUCCEEDED", **result}
        except Exception as exc:
            current = self._jobs.get(job.job_id)
            stage = current.stage if current else stage
            # §90：数据拉取阶段失败 => 显式标记 DATA_NOT_READY。
            message = (
                f"DATA_NOT_READY: {type(exc).__name__}: {exc}" if stage == "data" else f"{type(exc).__name__}: {exc}"
            )
            self._jobs.fail(job.job_id, stage, message)
            return {"job_id": job.job_id, "status": "FAILED", "stage": stage, "error": message}

    def list_factors(self, limit: int) -> list[dict]:
        return self._factors.list_active(limit)

    def get_factor(self, factor_id: str) -> dict | None:
        return self._factors.get(factor_id)

    def _panel(self, symbols: list[str], start: str | None, end: str | None):
        resolved_end = end or datetime.now(UTC).date().isoformat()
        resolved_start = start or (datetime.fromisoformat(resolved_end) - timedelta(days=730)).date().isoformat()
        snapshot = self._market.get_daily_bars(symbols, resolved_start, resolved_end, "qfq")
        signals = self._content.load_signals(symbols, resolved_start, resolved_end)
        return snapshot, build_feature_panel(snapshot, signals)

    def evaluate(
        self,
        factor_id: str | None,
        rpn: list[str],
        symbols: list[str],
        start: str | None,
        end: str | None,
        horizon: int,
    ) -> dict:
        definition = self._factors.get(factor_id) if factor_id else None
        formula = list(definition["rpn"] if definition else rpn)
        if not formula or not symbols:
            raise ValueError("rpn/factor_id and symbols are required")
        snapshot, panel = self._panel(symbols, start, end)
        values = StackVM().execute(formula, panel)
        if values is None:
            raise ValueError("invalid or non-computable factor formula")
        metrics = _json_safe(evaluate_factor(values, panel["close"], horizon=horizon))
        return {
            "factor_id": factor_id,
            "rpn": formula,
            "metrics": metrics,
            "data_version": snapshot.data_version,
            "data_snapshot_id": snapshot.data_snapshot_id,
        }

    def alpha_score(self, symbols: list[str], as_of: str | None) -> dict:
        snapshot, panel = self._panel(symbols, None, as_of)
        factors = self._factors.list_active(100)
        scores, contributions = compose_alpha_scores_with_evidence(panel, factors)
        items: list[dict] = []
        ranked_scores: list[dict] = []
        if scores is not None:
            order = sorted(
                (index for index in range(len(snapshot.symbols)) if not np.isnan(scores[index])),
                key=lambda index: float(scores[index]),
                reverse=True,
            )
            ranks = {index: position + 1 for position, index in enumerate(order)}
            factor_count = len(contributions)
            for index, symbol in enumerate(snapshot.symbols):
                score = None if np.isnan(scores[index]) else round(float(scores[index]), 8)
                items.append({"symbol": symbol, "score": score})
                evidence = [
                    {
                        "factor_id": item["factor_id"],
                        "contribution": round(float(item["column"][index]) / factor_count, 8)
                        if not np.isnan(item["column"][index])
                        else None,
                    }
                    for item in contributions
                    if not np.isnan(item["column"][index])
                ]
                # 详细修改方案 §12：coverage/confidence 避免 Agent 把缺数据的股票当高置信 Alpha。
                coverage = round(len(evidence) / factor_count, 4) if factor_count else 0.0
                confidence = round(coverage * (1.0 - 1.0 / max(len(ranks), 1)) if score is not None else 0.0, 4)
                ranked_scores.append(
                    {
                        "symbol": symbol,
                        "alpha_score": score,
                        "score": score,
                        "rank": ranks.get(index),
                        "coverage": coverage,
                        "confidence": confidence,
                        "factor_contributions": evidence,
                        "evidence": evidence,
                    }
                )
        factor_set_version = (
            "factor-set-"
            + hashlib.sha256("|".join(sorted(str(factor.get("factor_id")) for factor in factors)).encode()).hexdigest()[
                :12
            ]
        )
        factor_set_id = (
            "fs-"
            + hashlib.sha256(
                "|".join(sorted(f"{factor.get('factor_id')}:{factor.get('version', 1)}" for factor in factors)).encode()
            ).hexdigest()[:16]
        )
        return {
            "as_of": snapshot.dates[-1] if snapshot.dates else as_of,
            "factor_count": len(contributions),
            "data_version": snapshot.data_version,
            "data_snapshot_id": snapshot.data_snapshot_id,
            # §14.2 On-demand Alpha Score 契约字段
            "factor_set_id": factor_set_id,
            "factor_set_version": factor_set_version,
            "market_snapshot_id": snapshot.data_snapshot_id,
            "scores": ranked_scores,
            # 兼容旧消费方的精简结构
            "items": items,
        }

    def generate_paper_orders(self, scores: list[dict], as_of: str, snapshot_id: str, top_k: int) -> dict:
        return self._paper.generate_orders(scores, as_of, snapshot_id, top_k)

    def run_paper(self, as_of: str, snapshot_id: str, market_prices: dict[str, dict] | None = None) -> dict:
        return self._paper.run(as_of, snapshot_id, market_prices)

    def paper_state(self) -> dict:
        return self._paper.state()

    def paper_equity(self) -> list[dict]:
        return self._paper.equity()

    def paper_replay(self) -> dict:
        return self._paper.replay()
