"""HTTP boundary for immutable ResearchArtifactV2 evidence."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from stock_factor.application.artifacts.seal import ResearchArtifactError, ResearchArtifactService


def create_router(service: ResearchArtifactService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/research-artifacts", tags=["research-artifacts"])

    @router.post("")
    def seal_artifact(payload: dict) -> dict:
        try:
            artifact = service.seal(payload)
        except (ResearchArtifactError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"contract_version": "research-artifact.v2", "data": artifact.to_payload()}

    @router.get("/{artifact_id}")
    def get_artifact(artifact_id: str) -> dict:
        artifact = service.get(artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="research artifact not found")
        return {"contract_version": "research-artifact.v2", "data": artifact.to_payload()}

    @router.post("/{artifact_id}/verify")
    def verify_artifact(artifact_id: str, payload: dict | None = None) -> dict:
        payload = payload or {}
        try:
            result = service.verify(
                artifact_id,
                dependency_lock_hash=payload.get("dependency_lock_hash"),
                contract_checksums=payload.get("contract_checksums"),
            )
        except ResearchArtifactError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"contract_version": "research-artifact.v2", "data": result}

    @router.get("/{artifact_id}/verify")
    def verify_artifact_get(artifact_id: str) -> dict:
        try:
            result = service.verify(artifact_id)
        except ResearchArtifactError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"contract_version": "research-artifact.v2", "data": result}

    @router.get("/{artifact_id}/replay")
    def replay_artifact(artifact_id: str) -> dict:
        try:
            service.verify(artifact_id)
            artifact = service.get(artifact_id)
        except ResearchArtifactError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if artifact is None:
            raise HTTPException(status_code=404, detail="research artifact not found")
        return {"contract_version": "research-artifact.v2", "data": artifact.to_payload()}

    return router


__all__ = ["create_router"]
