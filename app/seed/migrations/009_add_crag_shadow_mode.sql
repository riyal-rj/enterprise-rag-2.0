-- CRAG shadow mode: an observe-only rollout stage, ahead of a real
-- (web-eligible) treatment rollout - see
-- app.rag_services.crag.dynamic_corrective_retriever and RAGService.answer's
-- "shadow" cohort handling. Starts fully disabled, matching
-- crag_enabled/crag_web_enabled (see 008_add_crag_rag_ops_config.sql).

ALTER TABLE rag_ops_config
    ADD COLUMN IF NOT EXISTS crag_shadow_enabled BOOLEAN NOT NULL DEFAULT FALSE;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'rag_ops_config_crag_shadow_requires_crag_check'
    ) THEN
        ALTER TABLE rag_ops_config
            ADD CONSTRAINT rag_ops_config_crag_shadow_requires_crag_check
            CHECK (NOT crag_shadow_enabled OR crag_enabled);
    END IF;
END $$;
