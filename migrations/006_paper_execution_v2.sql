CREATE TABLE IF NOT EXISTS paper_run (
    run_id VARCHAR(64) PRIMARY KEY, account_id VARCHAR(64) NOT NULL, trade_date DATE NOT NULL,
    signal_snapshot_id VARCHAR(128) NOT NULL, data_snapshot_id VARCHAR(128) NOT NULL,
    fee_model_version VARCHAR(80) NOT NULL, slippage_model_version VARCHAR(80) NOT NULL,
    idempotency_key VARCHAR(255) UNIQUE NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS paper_order_v2 (
    order_id VARCHAR(128) PRIMARY KEY, run_id VARCHAR(64) REFERENCES paper_run(run_id) ON DELETE CASCADE,
    symbol VARCHAR(32) NOT NULL, side VARCHAR(8) NOT NULL, requested_quantity INTEGER NOT NULL,
    filled_quantity INTEGER NOT NULL DEFAULT 0, remaining_quantity INTEGER NOT NULL,
    target_weight DOUBLE PRECISION, status VARCHAR(20) NOT NULL, blocked_reason VARCHAR(80),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), execute_on DATE
);
CREATE TABLE IF NOT EXISTS paper_fill (
    fill_id VARCHAR(64) PRIMARY KEY, order_id VARCHAR(128) NOT NULL REFERENCES paper_order_v2(order_id) ON DELETE CASCADE,
    quantity INTEGER NOT NULL, price DOUBLE PRECISION NOT NULL, gross_amount DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION NOT NULL DEFAULT 0, stamp_duty DOUBLE PRECISION NOT NULL DEFAULT 0,
    slippage DOUBLE PRECISION NOT NULL DEFAULT 0, net_cash_change DOUBLE PRECISION NOT NULL, executed_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_cash_ledger (
    entry_id VARCHAR(64) PRIMARY KEY, run_id VARCHAR(64) REFERENCES paper_run(run_id) ON DELETE CASCADE,
    event_type VARCHAR(30) NOT NULL, amount DOUBLE PRECISION NOT NULL, balance_after DOUBLE PRECISION NOT NULL,
    reference_id VARCHAR(128), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
