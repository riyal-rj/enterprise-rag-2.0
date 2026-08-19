-- Text-to-SQL: admin-controlled rollout flag + proposal/approval/catalog
-- state. Same two-field-plus-invariant shape HyDE/CRAG/self-reflection used
-- (007/008/010) - sql_proposal_only starts (and, at the application layer,
-- stays - see RagRuntimeConfig.__post_init__) TRUE: this release has no
-- automatic-execution path at all, only proposal-then-approve. These new
-- tables hold app metadata (proposals/approvals/audit), not analytics data -
-- they live in this database alongside rag_ops_config, same as every other
-- feature's rollout state, not in the separate analytics database SQL
-- queries themselves run against (see app.core.config.sql_settings).

ALTER TABLE rag_ops_config
    ADD COLUMN IF NOT EXISTS sql_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS sql_rollout_percentage SMALLINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS sql_proposal_only BOOLEAN NOT NULL DEFAULT TRUE;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'rag_ops_config_sql_rollout_percentage_check'
    ) THEN
        ALTER TABLE rag_ops_config
            ADD CONSTRAINT rag_ops_config_sql_rollout_percentage_check
            CHECK (sql_rollout_percentage BETWEEN 0 AND 100);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'rag_ops_config_sql_proposal_only_check'
    ) THEN
        -- Database-level backstop matching the application-level invariant
        -- (RagRuntimeConfig.__post_init__) - automatic execution isn't
        -- supported by this release regardless of who writes this row.
        ALTER TABLE rag_ops_config
            ADD CONSTRAINT rag_ops_config_sql_proposal_only_check
            CHECK (sql_proposal_only);
    END IF;
END $$;

-- Catalog version state: a single mutable row, same singleton-row shape as
-- rag_ops_config - see app.sql.catalog.StaticSQLCatalog. Bumped by an
-- out-of-band admin/background catalog refresh, never by a live request.
CREATE TABLE IF NOT EXISTS sql_catalog_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    refreshed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    refreshed_by VARCHAR(128)
);

INSERT INTO sql_catalog_state (singleton)
VALUES (TRUE)
ON CONFLICT (singleton) DO NOTHING;

-- One row per generated-and-validated (never per raw model attempt)
-- proposal - see app.repositories.sql_proposal_repository. The state
-- machine (proposed -> approved/executing/executed/rejected/expired/failed)
-- is enforced in application code under SELECT ... FOR UPDATE; the CHECK
-- constraint here is the final backstop against any writer that bypasses
-- the application layer, same role migrations 009/010's CHECK constraints
-- play for CRAG/self-reflection state.
CREATE TABLE IF NOT EXISTS sql_query_proposals (
    id UUID PRIMARY KEY,
    username VARCHAR(128) NOT NULL,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    question TEXT NOT NULL,
    sql_text TEXT NOT NULL,
    sql_fingerprint CHAR(64) NOT NULL,
    referenced_tables JSONB NOT NULL DEFAULT '[]'::jsonb,
    assumptions JSONB NOT NULL DEFAULT '[]'::jsonb,
    catalog_version INTEGER NOT NULL CHECK (catalog_version > 0),
    policy_version VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'proposed'
        CHECK (status IN (
            'proposed', 'approved', 'executing', 'executed',
            'rejected', 'expired', 'failed'
        )),
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    approved_at TIMESTAMPTZ,
    executed_at TIMESTAMPTZ,
    row_count INTEGER,
    execution_ms DOUBLE PRECISION,
    error_code VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_sql_query_proposals_owner_status
    ON sql_query_proposals (username, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_sql_query_proposals_pending_expiry
    ON sql_query_proposals (expires_at)
    WHERE status = 'proposed';

-- Curated, human-reviewed question/SQL pairs used to steer generation - see
-- app.repositories.sql_example_repository. Distinct from every row in
-- sql_query_proposals: promotion here requires review, not just a user
-- approving a query once (see the architecture blueprint's warning against
-- auto-training on execution history).
CREATE TABLE IF NOT EXISTS sql_approved_examples (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    sql_text TEXT NOT NULL,
    sql_fingerprint CHAR(64) NOT NULL UNIQUE,
    catalog_version INTEGER NOT NULL,
    approved_by VARCHAR(128) NOT NULL,
    approved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb
);
