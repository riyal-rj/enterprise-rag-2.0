-- Per-document ingestion lifecycle for the policy-upload quarantine
-- workflow (see app.guardrails.ingestion_security and
-- app.services.policy_ingestion_security_service). A separate CREATE
-- TABLE, not folded into 013's rag_ops_config ALTER: unrelated concern
-- (per-document lifecycle, not a global admin toggle), matching how
-- migration 006's new audit table was split from 005's new config table
-- even though both landed in the same feature.
--
-- One row per uploaded filename; a re-upload of the same filename creates
-- a NEW row (never reuses/updates a previous one) so the audit history of
-- every version stays intact even though Qdrant's own upsert-by-source
-- semantics replace the prior version's vectors.

CREATE TABLE IF NOT EXISTS document_security_state (
    id                  BIGSERIAL PRIMARY KEY,
    source              VARCHAR(512) NOT NULL,
    status              VARCHAR(24) NOT NULL DEFAULT 'pending_scan'
                            CHECK (status IN (
                                'pending_scan', 'scan_passed', 'scan_failed',
                                'approved', 'active', 'rejected'
                            )),
    uploaded_by         VARCHAR(32) NOT NULL,
    uploaded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    scan_decision       JSONB,
    scanned_at          TIMESTAMPTZ,
    approved_by         VARCHAR(32),
    approved_at         TIMESTAMPTZ,
    rejected_reason     TEXT,
    chunk_count         INTEGER
);

CREATE INDEX IF NOT EXISTS idx_document_security_state_source_uploaded_at
    ON document_security_state (source, uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_document_security_state_status
    ON document_security_state (status) WHERE status IN ('pending_scan', 'scan_passed');
