"""Seal, verify, and retrieve immutable ResearchArtifactV2 evidence."""

from __future__ import annotations

from typing import Any

from stock_factor.domain.research_artifact import CONTRACT_VERSION, ResearchArtifactV2, canonical_json
from stock_factor.ports.research_artifact_repository import ResearchArtifactRepository


class ResearchArtifactError(ValueError):
    """Artifact validation or append-only conflict."""


class InMemoryResearchArtifactRepository:
    def __init__(self) -> None:
        self._items: dict[str, tuple[bytes, ResearchArtifactV2]] = {}

    def save(self, artifact: ResearchArtifactV2) -> ResearchArtifactV2:
        if not artifact.verify():
            raise ResearchArtifactError("research artifact hash mismatch")
        key = artifact.artifact_id
        encoded = canonical_json(artifact.to_payload(include_id=False))
        existing = self._items.get(key)
        if existing is not None:
            if existing[0] != encoded:
                raise ResearchArtifactError("artifact id already exists with different payload")
            return existing[1]
        self._items[key] = (encoded, artifact)
        return artifact

    def get(self, artifact_id: str) -> ResearchArtifactV2 | None:
        item = self._items.get(artifact_id)
        return item[1] if item else None


class ResearchArtifactService:
    def __init__(self, repository: ResearchArtifactRepository) -> None:
        self.append_only = True
        self._repository = repository

    def seal(self, artifact: ResearchArtifactV2 | dict[str, Any]) -> ResearchArtifactV2:
        item = artifact if isinstance(artifact, ResearchArtifactV2) else ResearchArtifactV2.from_payload(artifact)
        if item.artifact_status != "SEALED":
            raise ResearchArtifactError("only SEALED ResearchArtifactV2 may be persisted")
        return self._repository.save(item)

    def get(self, artifact_id: str) -> ResearchArtifactV2 | None:
        return self._repository.get(artifact_id)

    def verify(
        self,
        artifact_id: str,
        *,
        dependency_lock_hash: str | None = None,
        contract_checksums: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        artifact = self._repository.get(artifact_id)
        if artifact is None:
            raise ResearchArtifactError("research artifact not found")
        if dependency_lock_hash is not None and artifact.dependency_lock_hash != dependency_lock_hash:
            raise ResearchArtifactError("dependency lock hash mismatch")
        if contract_checksums is not None and artifact.contract_checksums != contract_checksums:
            raise ResearchArtifactError("contract checksum mismatch")
        if not artifact.verify() or artifact.artifact_status != "SEALED":
            raise ResearchArtifactError("research artifact verification failed")
        return {
            "artifact_id": artifact.artifact_id,
            "contract_version": CONTRACT_VERSION,
            "artifact_status": artifact.artifact_status,
            "verified": True,
            "dependency_lock_hash": artifact.dependency_lock_hash,
            "contract_checksums": dict(artifact.contract_checksums),
        }

    def require_verified_sealed(self, artifact_id: str) -> ResearchArtifactV2:
        self.verify(artifact_id)
        artifact = self.get(artifact_id)
        if artifact is None:
            raise ResearchArtifactError("research artifact not found")
        return artifact


__all__ = [
    "InMemoryResearchArtifactRepository",
    "ResearchArtifactError",
    "ResearchArtifactService",
]
