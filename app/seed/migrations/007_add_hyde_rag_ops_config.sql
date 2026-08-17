-- HyDE is opt-in. Static model/prompt budgets remain deployment settings
-- (see app.core.config.rag_features.RAGFeatureSettings); only the safe
-- operational switches - whether it's on at all, and how much traffic is
-- sampled into it - belong in the mutable control-plane row, matching
-- reranking's reranker_rollout_percentage (see 005_create_rag_ops_config.sql).

ALTER TABLE rag_ops_config
    ADD COLUMN IF NOT EXISTS hyde_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS hyde_rollout_percentage SMALLINT NOT NULL DEFAULT 0;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'rag_ops_config_hyde_rollout_percentage_check'
    ) THEN
        ALTER TABLE rag_ops_config
            ADD CONSTRAINT rag_ops_config_hyde_rollout_percentage_check
            CHECK (hyde_rollout_percentage BETWEEN 0 AND 100);
    END IF;
END $$;
