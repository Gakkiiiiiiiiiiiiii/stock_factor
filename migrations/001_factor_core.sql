CREATE TABLE IF NOT EXISTS factor_definition (
    factor_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(160) NOT NULL,
    rpn JSONB NOT NULL,
    hypothesis TEXT NOT NULL DEFAULT '',
    status VARCHAR(30) NOT NULL DEFAULT 'CANDIDATE',
    version INTEGER NOT NULL DEFAULT 1,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    candidate_hash VARCHAR(64) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_factor_status ON factor_definition(status, updated_at);

CREATE TABLE IF NOT EXISTS factor_job (
    job_id VARCHAR(64) PRIMARY KEY,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    stage VARCHAR(40) NOT NULL DEFAULT 'queued',
    progress INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    request JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT,
    lease_owner VARCHAR(128),
    lease_expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_factor_job_claim ON factor_job(status, lease_expires_at, created_at);
