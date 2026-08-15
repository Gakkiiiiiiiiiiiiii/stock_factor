ALTER TABLE paper_cash_ledger ADD COLUMN IF NOT EXISTS sequence INTEGER NOT NULL DEFAULT 1;
ALTER TABLE paper_cash_ledger ADD COLUMN IF NOT EXISTS balance_before DOUBLE PRECISION NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS ix_paper_cash_ledger_run_sequence ON paper_cash_ledger(run_id, sequence);
