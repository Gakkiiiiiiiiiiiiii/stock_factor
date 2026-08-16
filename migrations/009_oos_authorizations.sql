-- 详细修改方案 P0-2：Final OOS 数据库级一次性授权
-- AUTHORIZED -> CONSUMED 必须在同一事务内完成（SELECT ... FOR UPDATE）。
CREATE TABLE IF NOT EXISTS oos_authorizations (
    authorization_id VARCHAR(36) PRIMARY KEY,
    experiment_id VARCHAR(64) NOT NULL UNIQUE,
    final_oos_snapshot_id VARCHAR(128) NOT NULL DEFAULT '',
    candidate_set_hash VARCHAR(64) NOT NULL DEFAULT '',
    status VARCHAR(16) NOT NULL DEFAULT 'AUTHORIZED',
    authorized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consumed_at TIMESTAMPTZ,
    invalidated_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_oos_authorizations_status ON oos_authorizations(status);

-- 详细修改方案 P0-4：Candidate Freeze 完整证据列
ALTER TABLE factor_candidate_freeze ADD COLUMN IF NOT EXISTS extra JSONB NOT NULL DEFAULT '{}'::jsonb;
