-- 详细修改方案 §11/§17：FactorSet 正式版本化
CREATE TABLE IF NOT EXISTS factor_sets (
    factor_set_id VARCHAR(64) PRIMARY KEY,
    factor_set_version VARCHAR(64) NOT NULL,
    research_experiment_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    promotion_policy_version VARCHAR(80) NOT NULL DEFAULT 'promotion_gate_v2',
    valid_from VARCHAR(64),
    valid_to VARCHAR(64),
    status VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',
    code_sha VARCHAR(128),
    config_hash VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_factor_sets_status ON factor_sets(status);
CREATE INDEX IF NOT EXISTS idx_factor_sets_version ON factor_sets(factor_set_version);

CREATE TABLE IF NOT EXISTS factor_set_members (
    id BIGSERIAL PRIMARY KEY,
    factor_set_id VARCHAR(64) NOT NULL,
    factor_id VARCHAR(64) NOT NULL,
    factor_version INTEGER NOT NULL DEFAULT 1,
    weight DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_factor_set_members_set ON factor_set_members(factor_set_id);
