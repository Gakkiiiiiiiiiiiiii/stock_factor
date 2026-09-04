-- FactorSet formal promotion lineage (SF-P1-02).
ALTER TABLE factor_sets ADD COLUMN IF NOT EXISTS research_artifact_ids JSON NOT NULL DEFAULT '[]';
ALTER TABLE factor_sets ADD COLUMN IF NOT EXISTS formal_eligible BOOLEAN NOT NULL DEFAULT FALSE;
