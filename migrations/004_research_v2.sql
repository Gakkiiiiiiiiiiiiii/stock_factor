CREATE TABLE IF NOT EXISTS factor_candidate (
    candidate_id VARCHAR(64) PRIMARY KEY, candidate_hash VARCHAR(64) UNIQUE NOT NULL,
    mining_job_id VARCHAR(64), parent_candidate_id VARCHAR(64), generation_round INTEGER NOT NULL DEFAULT 1,
    generation_strategy VARCHAR(80), hypothesis TEXT NOT NULL DEFAULT '', formula JSONB NOT NULL,
    canonical_formula TEXT NOT NULL, model_name VARCHAR(120), prompt_version VARCHAR(80),
    feedback JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS factor_statistical_test (
    id BIGSERIAL PRIMARY KEY, candidate_id VARCHAR(64) NOT NULL REFERENCES factor_candidate(candidate_id) ON DELETE CASCADE,
    experiment_id VARCHAR(64) NOT NULL, raw_p_value DOUBLE PRECISION, adjusted_p_value DOUBLE PRECISION,
    q_value DOUBLE PRECISION, pbo DOUBLE PRECISION, effective_trials INTEGER NOT NULL,
    passed_multiple_testing BOOLEAN NOT NULL, passed_pbo BOOLEAN NOT NULL, method VARCHAR(80) NOT NULL,
    data_snapshot_id VARCHAR(128), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS factor_final_oos (
    id BIGSERIAL PRIMARY KEY, factor_id VARCHAR(64) NOT NULL, factor_version INTEGER NOT NULL,
    metrics JSONB NOT NULL, data_snapshot_id VARCHAR(128), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS factor_oos_audit (
    id BIGSERIAL PRIMARY KEY, factor_id VARCHAR(64) NOT NULL, factor_version INTEGER NOT NULL,
    audit_status VARCHAR(20) NOT NULL, violations JSONB NOT NULL DEFAULT '[]'::jsonb,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb, audit_version VARCHAR(80) NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
