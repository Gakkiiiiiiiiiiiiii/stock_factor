-- Additive runtime-authority tables omitted by the first V2 schema pass.
CREATE TABLE IF NOT EXISTS paper_position_lot (
    lot_id VARCHAR(64) PRIMARY KEY, account_id VARCHAR(64) NOT NULL, symbol VARCHAR(32) NOT NULL,
    buy_date DATE NOT NULL, quantity INTEGER NOT NULL, available_quantity INTEGER NOT NULL,
    cost_price DOUBLE PRECISION NOT NULL, remaining_cost DOUBLE PRECISION NOT NULL,
    opened_by_fill_id VARCHAR(64), closed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_paper_position_lot_account_symbol ON paper_position_lot(account_id, symbol);
