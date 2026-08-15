from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from stock_factor.api.dependencies import build_application
from stock_factor.application.service import FactorApplication


class MiningJobRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None
    days: int | None = Field(default=365, ge=60)
    eval_window: int | None = Field(default=None, ge=1)
    horizon: int = Field(default=5, ge=1, le=60)
    candidates: list[dict] = Field(default_factory=list)
    use_model: bool = False


class AlphaScoreRequest(BaseModel):
    symbols: list[str] = Field(min_length=1)
    as_of: str | None = None


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
    application = service or build_application()

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok", "service": "stock_factor", "contract_version": "factor.v1"}

    @app.post("/api/v1/mining/jobs")
    def create_mining_job(request: MiningJobRequest) -> dict:
        payload = request.model_dump()
        payload["symbols"] = payload["symbols"] or ["000001.SH", "399001.SZ"]
        return {"contract_version": "factor.v1", "data": application.create_mining_job(payload)}

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

    @app.post("/api/v1/paper/orders/generate")
    def generate_orders(request: PaperOrderRequest) -> dict:
        return {
            "contract_version": "factor.v1",
            "data": application.generate_paper_orders(
                request.scores, request.as_of, request.data_snapshot_id, request.top_k
            ),
        }

    @app.post("/api/v1/paper/run")
    def run_paper(request: PaperRunRequest) -> dict:
        return {
            "contract_version": "factor.v1",
            "data": application.run_paper(request.as_of, request.data_snapshot_id, request.market_prices),
        }

    @app.get("/api/v1/paper/state")
    def paper_state() -> dict:
        return {"contract_version": "factor.v1", "data": application.paper_state()}

    @app.get("/api/v1/paper/equity")
    def paper_equity() -> dict:
        return {"contract_version": "factor.v1", "items": application.paper_equity()}

    return app


app = create_app()
