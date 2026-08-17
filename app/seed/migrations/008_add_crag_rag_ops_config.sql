-- CRAG is opt-in and starts fully disabled, regardless of the (retired)
-- crag_enabled_by_default=True scaffold in RAGFeatureSettings - see
-- app.core.config.rag_features. Static model/prompt/threshold budgets
-- remain deployment settings; only the safe operational switches - on at
-- all, how much traffic is sampled into it, and whether allowlisted
-- regulatory-web correction is permitted - belong in the mutable
-- control-plane row, matching hyde_enabled/hyde_rollout_percentage (see
-- 007_add_hyde_rag_ops_config.sql).

ALTER TABLE rag_ops_config
    ADD COLUMN IF NOT EXISTS crag_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS crag_rollout_percentage SMALLINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS crag_web_enabled BOOLEAN NOT NULL DEFAULT FALSE;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'rag_ops_config_crag_rollout_percentage_check'
    ) THEN
        ALTER TABLE rag_ops_config
            ADD CONSTRAINT rag_ops_config_crag_rollout_percentage_check
            CHECK (crag_rollout_percentage BETWEEN 0 AND 100);
    END IF;
END $$;
