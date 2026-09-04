"""Persistence boundary for immutable research artifacts."""

from __future__ import annotations

from typing import Protocol

from stock_factor.domain.research_artifact import ResearchArtifactV2


class ResearchArtifactRepository(Protocol):
    def save(self, artifact: ResearchArtifactV2) -> ResearchArtifactV2: ...

    def get(self, artifact_id: str) -> ResearchArtifactV2 | None: ...


__all__ = ["ResearchArtifactRepository"]
