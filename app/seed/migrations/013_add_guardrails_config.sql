-- Guardrails layer control-plane fields (app.guardrails) - see
-- app.rag_services.rag_runtime_config.RagRuntimeConfig and
-- app.repositories.rag_ops_repository. guardrail_mode gates whether a
-- BLOCK/REDACT-worthy decision is actually applied ("enforce") or only
-- recorded ("monitor" - production startup refuses "monitor" while SQL/
-- CRAG-web is enabled, see app.core.config.settings).
-- safety_lockdown_enabled is a security-incident kill switch distinct from
-- emergency_disabled: it additionally forces the SQL route fully closed
-- regardless of sql_enabled (see app.guardrails.tool_guardrail), with its
-- own reason/at/by audit trio matching emergency_disabled_*'s shape.
-- Starts fully enforcing (not monitor-only), matching this layer's
-- "guardrails apply to every request" design - unlike every other feature
-- flag in this table, there is no "off by default" posture for guardrails.

ALTER TABLE rag_ops_config
    ADD COLUMN IF NOT EXISTS guardrail_mode VARCHAR(16) NOT NULL DEFAULT 'enforce',
    ADD COLUMN IF NOT EXISTS guardrail_policy_version VARCHAR(64) NOT NULL DEFAULT 'guardrails-policy-v1',
    ADD COLUMN IF NOT EXISTS safety_lockdown_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS safety_lockdown_reason TEXT,
    ADD COLUMN IF NOT EXISTS safety_lockdown_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS safety_lockdown_by VARCHAR(32);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'rag_ops_config_guardrail_mode_check'
    ) THEN
        ALTER TABLE rag_ops_config
            ADD CONSTRAINT rag_ops_config_guardrail_mode_check
            CHECK (guardrail_mode IN ('enforce', 'monitor'));
    END IF;
END $$;
