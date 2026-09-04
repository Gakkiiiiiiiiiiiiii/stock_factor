"""HTTP routes for liveness and separated readiness states."""

from __future__ import annotations

from fastapi import APIRouter

from stock_factor.application.readiness import ReadinessService


def create_router(service: ReadinessService) -> APIRouter:
    router = APIRouter()

    @router.get("/health/live")
    def live() -> dict:
        return service.liveness()

    @router.get("/health/ml-ready")
    def ml_ready() -> dict:
        return service.ml().to_dict()

    @router.get("/health/paper-ready")
    def paper_ready() -> dict:
        return service.paper().to_dict()

    return router


__all__ = ["create_router"]
