CREATE TABLE IF NOT EXISTS factor_version (
    factor_version_id VARCHAR(64) PRIMARY KEY, factor_id VARCHAR(64) NOT NULL REFERENCES factor_definition(factor_id) ON DELETE CASCADE,
    version INTEGER NOT NULL, formula JSONB NOT NULL, canonical_formula TEXT NOT NULL,
    operator_version VARCHAR(80), parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    normalization JSONB NOT NULL DEFAULT '{}'::jsonb, universe JSONB NOT NULL DEFAULT '{}'::jsonb,
    research_config JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(factor_id, version)
);
CREATE TABLE IF NOT EXISTS factor_promotion_decision (
    decision_id VARCHAR(64) PRIMARY KEY, factor_id VARCHAR(64) NOT NULL, factor_version INTEGER NOT NULL,
    mining_job_id VARCHAR(64), research_window_id VARCHAR(64), data_snapshot_id VARCHAR(128), passed BOOLEAN NOT NULL,
    failed_rules JSONB NOT NULL DEFAULT '[]'::jsonb, metrics_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    gate_version VARCHAR(80) NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS factor_lifecycle_event (
    event_id VARCHAR(64) PRIMARY KEY, factor_id VARCHAR(64) NOT NULL, factor_version INTEGER NOT NULL,
    from_status VARCHAR(30), to_status VARCHAR(30) NOT NULL, reason TEXT NOT NULL,
    metrics_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb, data_snapshot_id VARCHAR(128), actor VARCHAR(80) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
