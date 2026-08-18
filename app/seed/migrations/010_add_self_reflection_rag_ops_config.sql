-- Self-reflective answer critique/revision stage: an admin-controlled
-- rollout, same two-field shape as HyDE (enabled + rollout percentage,
-- no web/shadow-style dependent sub-flag) - see
-- app.rag_services.reflection.dynamic_self_reflection and
-- RAGService.answer's post-answer reflection block. Starts fully disabled,
-- matching every other feature flag's rollout-safety default (see
-- 007_add_hyde_rag_ops_config.sql, 008_add_crag_rag_ops_config.sql).

ALTER TABLE rag_ops_config
    ADD COLUMN IF NOT EXISTS self_reflective_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS self_reflective_rollout_percentage SMALLINT NOT NULL DEFAULT 0;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'rag_ops_config_self_reflective_rollout_percentage_check'
    ) THEN
        ALTER TABLE rag_ops_config
            ADD CONSTRAINT rag_ops_config_self_reflective_rollout_percentage_check
            CHECK (self_reflective_rollout_percentage BETWEEN 0 AND 100);
    END IF;
END $$;
