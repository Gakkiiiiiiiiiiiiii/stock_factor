CREATE TABLE IF NOT EXISTS factor_evaluation (
    evaluation_id BIGSERIAL PRIMARY KEY,
    factor_id VARCHAR(64) NOT NULL REFERENCES factor_definition(factor_id) ON DELETE CASCADE,
    data_version VARCHAR(128) NOT NULL,
    data_snapshot_id VARCHAR(128) NOT NULL,
    method VARCHAR(64) NOT NULL,
    metrics JSONB NOT NULL,
    passed BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS factor_lifecycle (
    id BIGSERIAL PRIMARY KEY,
    factor_id VARCHAR(64) NOT NULL REFERENCES factor_definition(factor_id) ON DELETE CASCADE,
    previous_status VARCHAR(30),
    new_status VARCHAR(30) NOT NULL,
    reason TEXT NOT NULL,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
