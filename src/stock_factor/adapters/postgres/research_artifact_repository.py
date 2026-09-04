"""Transactional PostgreSQL repository for append-only ResearchArtifactV2."""

from __future__ import annotations

from stock_factor.adapters.postgres.models import ResearchArtifactRow
from stock_factor.domain.research_artifact import ResearchArtifactV2, canonical_json
from stock_factor.ports.research_artifact_repository import ResearchArtifactRepository


class PostgresResearchArtifactRepository(ResearchArtifactRepository):
    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def save(self, artifact: ResearchArtifactV2) -> ResearchArtifactV2:
        with self._sessions.begin() as session:
            row = session.get(ResearchArtifactRow, artifact.artifact_id)
            if row is not None:
                existing = ResearchArtifactV2.from_payload(row.payload)
                if canonical_json(existing.to_payload(include_id=False)) != canonical_json(
                    artifact.to_payload(include_id=False)
                ):
                    raise ValueError("artifact id already exists with different payload")
                return existing
            session.add(
                ResearchArtifactRow(
                    artifact_id=artifact.artifact_id,
                    contract_version="research-artifact.v2",
                    artifact_status=artifact.artifact_status,
                    payload=artifact.to_payload(),
                )
            )
            session.flush()
            return artifact

    def get(self, artifact_id: str) -> ResearchArtifactV2 | None:
        with self._sessions() as session:
            row = session.get(ResearchArtifactRow, artifact_id)
            return ResearchArtifactV2.from_payload(row.payload) if row else None


__all__ = ["PostgresResearchArtifactRepository"]
