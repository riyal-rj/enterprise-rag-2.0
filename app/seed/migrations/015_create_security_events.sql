-- Sanitized audit trail for guardrail decisions (input/context/output/tool/
-- ingestion BLOCK and REDACT actions) - see app.guardrails and
-- app.repositories.security_events_repository. Same append-only shape as
-- rag_ops_audit_log (migration 006), but a separate table: these rows are
-- keyed by request/finding, not by an admin actor mutating global config,
-- and are written at much higher volume (every BLOCK/REDACT decision, not
-- every config change) - splitting keeps the two audit trails' retention/
-- volume characteristics independent.
--
-- "changes" is a sanitized finding summary (category/score/detector only -
-- see app.guardrails.security_events.summarize_decision) - raw scanned
-- text, matched content, and the user's question/answer are never written
-- here, matching every other guardrail audit surface's discipline.

CREATE TABLE IF NOT EXISTS security_events (
    id          BIGSERIAL PRIMARY KEY,
    actor       VARCHAR(32),
    action      VARCHAR(32) NOT NULL,
    stage       VARCHAR(16) NOT NULL,
    category    VARCHAR(32),
    mode        VARCHAR(16) NOT NULL CHECK (mode IN ('enforce', 'monitor')),
    changes     JSONB NOT NULL DEFAULT '{}'::jsonb,
    reason      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_security_events_created_at ON security_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_security_events_stage_category ON security_events (stage, category);
