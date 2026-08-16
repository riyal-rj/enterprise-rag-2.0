-- Append-only audit trail for RAG Operations panel mutations: config
-- changes, emergency disable/enable, and corpus version bumps (see
-- app.repositories.rag_ops_repository.PostgresRagOpsRepository). No FK to
-- rag_ops_config (a singleton row) - this log is independent of current
-- state and must keep every historical entry even across a config reset.

CREATE TABLE IF NOT EXISTS rag_ops_audit_log (
    id         SERIAL PRIMARY KEY,
    actor      VARCHAR(32) NOT NULL,
    action     VARCHAR(32) NOT NULL,
    changes    JSONB NOT NULL DEFAULT '{}'::jsonb,
    reason     TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_ops_audit_log_created_at
    ON rag_ops_audit_log (created_at DESC);
