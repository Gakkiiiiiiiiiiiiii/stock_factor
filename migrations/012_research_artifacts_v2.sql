-- Immutable formal research evidence (SF-P1-02).
CREATE TABLE IF NOT EXISTS research_artifacts_v2 (
    artifact_id VARCHAR(64) PRIMARY KEY,
    contract_version VARCHAR(80) NOT NULL,
    artifact_status VARCHAR(20) NOT NULL,
    payload JSON NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
