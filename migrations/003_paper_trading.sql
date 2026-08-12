CREATE TABLE IF NOT EXISTS paper_state (
    account_id VARCHAR(64) PRIMARY KEY,
    cash DOUBLE PRECISION NOT NULL DEFAULT 1000000,
    positions JSONB NOT NULL DEFAULT '{}'::jsonb,
    frozen_orders JSONB NOT NULL DEFAULT '[]'::jsonb,
    data_snapshot_id VARCHAR(128),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_equity (
    id BIGSERIAL PRIMARY KEY,
    account_id VARCHAR(64) NOT NULL,
    as_of VARCHAR(32) NOT NULL,
    equity DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    data_snapshot_id VARCHAR(128)
);

CREATE TABLE IF NOT EXISTS paper_order (
    order_id VARCHAR(128) PRIMARY KEY,
    account_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(8) NOT NULL,
    target_weight DOUBLE PRECISION NOT NULL,
    signal_as_of VARCHAR(32) NOT NULL,
    execute_on VARCHAR(32),
    status VARCHAR(20) NOT NULL,
    data_snapshot_id VARCHAR(128) NOT NULL
);
