-- Self-reflection rollout maturity, closing the two gaps flagged ahead of a
-- live rollout: an observe-only shadow cohort (same role as CRAG's, see
-- 009_add_crag_shadow_mode.sql) so reflection can be measured against real
-- traffic before it's ever served, and a separate opt-in for the bounded
-- additional-retrieval step so "revise the answer" and "also allow one more
-- retrieval" can be promoted independently (a revision-only canary). Both
-- start fully disabled and require self_reflective_enabled, matching every
-- other dependent sub-flag in this table (crag_web_enabled/crag_shadow_enabled
-- in 008/009).

ALTER TABLE rag_ops_config
    ADD COLUMN IF NOT EXISTS self_reflective_shadow_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS self_reflective_retrieval_enabled BOOLEAN NOT NULL DEFAULT FALSE;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'rag_ops_config_self_reflective_shadow_requires_enabled_check'
    ) THEN
        ALTER TABLE rag_ops_config
            ADD CONSTRAINT rag_ops_config_self_reflective_shadow_requires_enabled_check
            CHECK (NOT self_reflective_shadow_enabled OR self_reflective_enabled);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'rag_ops_config_self_reflective_retrieval_requires_enabled_check'
    ) THEN
        ALTER TABLE rag_ops_config
            ADD CONSTRAINT rag_ops_config_self_reflective_retrieval_requires_enabled_check
            CHECK (NOT self_reflective_retrieval_enabled OR self_reflective_enabled);
    END IF;
END $$;
