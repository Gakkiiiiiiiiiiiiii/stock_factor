from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from stock_factor.application.service import FactorApplication

app = FastAPI(title="stock_factor", version="1.0.0")
service = FactorApplication()


class MiningJobRequest(BaseModel):
    rounds: int | None = Field(default=None, ge=1)
    candidates_per_round: int | None = Field(default=None, ge=1)
    symbols: list[str] = Field(default_factory=list)
    days: int | None = Field(default=None, ge=60)
    eval_window: int | None = Field(default=None, ge=1)


class AlphaScoreRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    as_of: str | None = None


class FactorEvaluateRequest(BaseModel):
    factor_id: str | None = None
    rpn: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok", "service": "stock_factor", "contract_version": "factor.v1"}


@app.post("/api/v1/mining/jobs")
def create_mining_job(request: MiningJobRequest) -> dict:
    return service.create_mining_job(request.model_dump())


@app.get("/api/v1/mining/jobs/{job_id}")
def get_mining_job(job_id: str) -> dict:
    payload = service.get_mining_job(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="job not found")
    return payload


@app.post("/api/v1/mining/jobs/{job_id}/cancel")
def cancel_mining_job(job_id: str) -> dict:
    payload = service.cancel_mining_job(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="job not found")
    return payload


@app.get("/api/v1/factors")
def list_factors(limit: int = 20) -> dict:
    return {"items": [], "limit": min(max(limit, 1), 100)}


@app.post("/api/v1/factors/evaluate")
def evaluate_factor(request: FactorEvaluateRequest) -> dict:
    return {"factor_id": request.factor_id, "rpn": request.rpn, "metrics": {}, "warning": "factor engine migration pending"}


@app.post("/api/v1/alpha/score")
def score_alpha(request: AlphaScoreRequest) -> dict:
    return {"as_of": request.as_of, "factor_version": None, "data_version": None, "items": []}
