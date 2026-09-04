from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from stock_factor.adapters.http.quant_paper_client import QuantPaperClient, QuantPaperUnavailableError
from stock_factor.api.dependencies import build_application
from stock_factor.api.readiness import create_router as create_readiness_router
from stock_factor.api.research_artifacts import create_router as create_research_artifact_router
from stock_factor.application.artifacts.seal import (
    InMemoryResearchArtifactRepository,
    ResearchArtifactService,
)
from stock_factor.application.readiness import ReadinessAdmissionError, ReadinessService
from stock_factor.application.service import FactorApplication, MarketDataUnavailableError
from stock_factor.config.runtime import RuntimeConfig

SERVICE_NAME = "stock_factor"
SERVICE_VERSION = "1.0.0"
CONTRACT_VERSIONS = [
    "factor.v1",
    "market-snapshot.v1",
    "market-data.v1",
    "content-factor-signal.v3",
    "content-factor-signal.v5.1",
]


def _mark_deprecated(response: Response) -> None:
    # §26：Factor Paper 兼容端点显式标记弃用，权威在 quant/trading.v1。
    response.headers["Deprecation"] = "true"
    response.headers["X-Deprecated-By"] = "quant/trading.v1"


class MiningJobRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    research_question: str | None = None
    hypothesis: str | None = None
    start: str | None = None
    end: str | None = None
    days: int | None = Field(default=365, ge=60)
    eval_window: int | None = Field(default=None, ge=1)
    horizon: int = Field(default=5, ge=1, le=60)
    candidates: list[dict] = Field(default_factory=list)
    use_model: bool = False
    research_mode: str = "EXPLORATORY"
    formal_market_ref: dict | None = None
    formal_content_query: dict | None = None
    formal_content_ref: dict | None = None
    execution_cost_calibration: dict | None = None


class AlphaScoreRequest(BaseModel):
    symbols: list[str] = Field(min_length=1)
    as_of: str | None = None
    # §14.2：可选因子集（当前仅支持 production 全集）
    factor_set: str | None = "production"


class FactorEvaluateRequest(BaseModel):
    factor_id: str | None = None
    rpn: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(min_length=1)
    start: str | None = None
    end: str | None = None
    horizon: int = Field(default=5, ge=1, le=60)


class PaperOrderRequest(BaseModel):
    scores: list[dict]
    as_of: str
    data_snapshot_id: str
    top_k: int = Field(default=10, ge=1)


class PaperRunRequest(BaseModel):
    as_of: str
    data_snapshot_id: str
    market_prices: dict[str, dict] = Field(default_factory=dict)


def create_app(service: FactorApplication | None = None) -> FastAPI:
    app = FastAPI(title="stock_factor", version="1.0.0")
    config = service.runtime_config if service is not None else RuntimeConfig.from_env()
    application = service or build_application(runtime_config=config)
    readiness_service = getattr(application, "readiness_service", None) or ReadinessService(config)
    artifact_service = getattr(application, "research_artifact_service", None)
    if artifact_service is None:
        artifact_service = ResearchArtifactService(InMemoryResearchArtifactRepository())
    app.include_router(create_research_artifact_router(artifact_service))
    app.include_router(create_readiness_router(readiness_service))

    @app.middleware("http")
    async def trace_headers(request: Request, call_next):
        # §32：统一 Trace Headers，全链路保持同一 trace_id。
        trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())
        response: Response = await call_next(request)
        response.headers["x-trace-id"] = trace_id
        if request.headers.get("x-decision-id"):
            response.headers["x-decision-id"] = request.headers["x-decision-id"]
        response.headers["x-caller-service"] = request.headers.get("x-caller-service", "")
        return response

    @app.get("/healthz")
    def health() -> dict:
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "contract_version": "factor.v1",
            "paper_authority": "quant",
            "required_quant_contract": config.required_quant_contract,
            "required_quant_checksum": config.required_quant_checksum,
            "required_content_contract": config.required_content_contract,
            "required_content_checksum": config.required_content_checksum,
        }

    @app.get("/health/version")
    def health_version() -> dict:
        # §106 Release Version
        return {
            "service": SERVICE_NAME,
            "service_version": SERVICE_VERSION,
            "git_commit": os.getenv("FACTOR_GIT_COMMIT", "unknown"),
            "contract_versions": CONTRACT_VERSIONS,
            "paper_authority": "quant",
            "required_quant_contract": config.required_quant_contract,
            "required_quant_checksum": config.required_quant_checksum,
            "required_content_contract": config.required_content_contract,
            "required_content_checksum": config.required_content_checksum,
        }

    @app.get("/health/research-ready")
    def research_ready() -> dict:
        """Expose frozen, non-mutating research readiness and legacy fields."""
        report = readiness_service.research()
        return {
            **report.to_dict(),
            "status": "ready" if report.ready else "not_ready",
            "runtime_profile": config.profile.value,
            "paper_authority": "quant",
            "required_quant_contract": config.required_quant_contract,
            "required_quant_checksum_configured": bool(config.required_quant_checksum),
            "required_content_contract": config.required_content_contract,
            "required_content_checksum_configured": bool(config.required_content_checksum),
            "quant_base_url_configured": bool(config.quant_base_url),
            "allow_local_paper": config.allow_local_paper,
            "formal_eligible": report.ready,
        }

    @app.post("/api/v1/mining/jobs")
    def create_mining_job(request: MiningJobRequest, http_request: Request) -> dict:
        payload = request.model_dump()
        payload["symbols"] = payload["symbols"] or ["000001.SH", "399001.SZ"]
        # §33：写接口幂等，避免重试产生重复挖掘任务。
        idempotency_key = http_request.headers.get("idempotency-key")
        try:
            data = application.create_mining_job(payload, idempotency_key=idempotency_key)
        except ReadinessAdmissionError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": exc.code,
                    "blocking_reasons": list(exc.report.blocking_reasons),
                    "readiness": exc.report.to_dict(),
                },
            ) from exc
        except MarketDataUnavailableError as exc:
            # §90：Quant 市场数据不可用 => 禁止启动新 Mining。
            raise HTTPException(status_code=503, detail={"code": "DATA_NOT_READY", "message": str(exc)}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"contract_version": "factor.v1", "data": data}

    @app.get("/api/v1/mining/jobs/{job_id}")
    def get_mining_job(job_id: str) -> dict:
        payload = application.get_mining_job(job_id)
        if payload is None:
            raise HTTPException(404, "job not found")
        return {"contract_version": "factor.v1", "data": payload}

    @app.post("/api/v1/mining/jobs/{job_id}/cancel")
    def cancel_mining_job(job_id: str) -> dict:
        payload = application.cancel_mining_job(job_id)
        if payload is None:
            raise HTTPException(404, "job not found")
        return {"contract_version": "factor.v1", "data": payload}

    @app.get("/api/v1/factors")
    def list_factors(limit: int = 20) -> dict:
        resolved = min(max(limit, 1), 100)
        return {"contract_version": "factor.v1", "items": application.list_factors(resolved), "limit": resolved}

    @app.get("/api/v1/factors/{factor_id}")
    def get_factor(factor_id: str) -> dict:
        payload = application.get_factor(factor_id)
        if payload is None:
            raise HTTPException(404, "factor not found")
        return {"contract_version": "factor.v1", "data": payload}

    @app.post("/api/v1/factors/evaluate")
    def evaluate_factor(request: FactorEvaluateRequest) -> dict:
        try:
            data = application.evaluate(
                request.factor_id, request.rpn, request.symbols, request.start, request.end, request.horizon
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"contract_version": "factor.v1", "data": data}

    @app.post("/api/v1/alpha/score")
    def score_alpha(request: AlphaScoreRequest) -> dict:
        return {"contract_version": "factor.v1", "data": application.alpha_score(request.symbols, request.as_of)}

    quant_paper = QuantPaperClient(config.quant_base_url)

    def _quant_call(method, *args):
        try:
            return method(*args)
        except QuantPaperUnavailableError as exc:
            raise HTTPException(status_code=503, detail={"code": "QUANT_UNAVAILABLE", "message": str(exc)}) from exc

    @app.post("/api/v1/paper/orders/generate")
    def generate_orders(request: PaperOrderRequest, response: Response) -> dict:
        _mark_deprecated(response)
        return {
            "contract_version": "trading.v1",
            "authority": "quant",
            "formal_eligible": True,
            "data": _quant_call(
                quant_paper.generate_orders, request.scores, request.as_of, request.data_snapshot_id, request.top_k
            ),
        }

    @app.post("/api/v1/paper/run")
    def run_paper(request: PaperRunRequest, response: Response) -> dict:
        _mark_deprecated(response)
        return {
            "contract_version": "trading.v1",
            "authority": "quant",
            "formal_eligible": True,
            "data": _quant_call(quant_paper.run, request.as_of, request.data_snapshot_id, request.market_prices),
        }

    @app.get("/api/v1/paper/state")
    def paper_state(response: Response) -> dict:
        _mark_deprecated(response)
        return {
            "contract_version": "trading.v1",
            "authority": "quant",
            "formal_eligible": True,
            "data": _quant_call(quant_paper.state),
        }

    @app.get("/api/v1/paper/equity")
    def paper_equity(response: Response) -> dict:
        _mark_deprecated(response)
        return {
            "contract_version": "trading.v1",
            "authority": "quant",
            "formal_eligible": True,
            "items": _quant_call(quant_paper.equity),
        }

    @app.get("/api/v1/paper/replay")
    def paper_replay(response: Response) -> dict:
        _mark_deprecated(response)
        return {
            "contract_version": "trading.v1",
            "authority": "quant",
            "formal_eligible": True,
            "data": _quant_call(quant_paper.replay),
        }

    return app


app = create_app()
