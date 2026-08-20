-- =============================================================================
-- Text-to-SQL analytics database provisioning script.
--
-- THIS SCRIPT IS NOT RUN BY THE APPLICATION, ANY MIGRATION RUNNER, OR CI.
-- It must be reviewed and run manually by a DBA against the real analytics
-- database once one is provisioned for this deployment - never against the
-- application's own database (the one DATABASE_URL/app/seed/migrations
-- points at). See the Text-to-SQL architecture blueprint, section 13.
--
-- Prerequisites this script assumes are already true:
--   1. A separate "analytics" Postgres database/instance exists, containing
--      (or about to contain) curated, read-only views under an
--      "approved_analytics" schema - never raw operational tables, and
--      never this application's own users/chat_history/conversations/
--      rag_ops_config tables, which must not be reachable from this role
--      at all.
--   2. Those views already exclude columns this deployment hasn't approved
--      for SQL access (passwords, hashes, tokens, full PII) or mark them
--      sensitive so app.sql.sql_result_policy can mask them.
--   3. app/sql/catalog_definition.json has been (or will be) updated to
--      describe exactly the tables/columns granted below - the AST policy
--      (app.sql.sql_policy) allowlists against that file, independent of
--      what this role can actually SELECT; both must agree.
--   4. The generated password below is replaced with a real secret from
--      your vault/secrets manager before running this script, and that
--      secret becomes SQL_DATABASE_URL - never committed to source control.
--
-- Placeholders to replace before running:
--   <DB_NAME>            analytics database name
--   <SECRET_FROM_VAULT>  a strong, randomly generated password
--   approved_analytics.<view_name>  one GRANT per approved view/table
--   <tenant_id column>   whatever column your approved views expose for
--                        row-level tenant scoping, if applicable
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Role: no ownership, no DDL, no superuser, no RLS bypass.
-- ---------------------------------------------------------------------------
CREATE ROLE rag_sql_reader LOGIN PASSWORD '<SECRET_FROM_VAULT>'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;

ALTER ROLE rag_sql_reader SET default_transaction_read_only = on;
ALTER ROLE rag_sql_reader SET statement_timeout = '5s';
ALTER ROLE rag_sql_reader SET lock_timeout = '500ms';
ALTER ROLE rag_sql_reader SET idle_in_transaction_session_timeout = '5s';
ALTER ROLE rag_sql_reader SET search_path = pg_catalog;

-- ---------------------------------------------------------------------------
-- 2. Database/schema grants: connect + read the approved schema only.
-- ---------------------------------------------------------------------------
REVOKE ALL ON DATABASE <DB_NAME> FROM PUBLIC;
GRANT CONNECT ON DATABASE <DB_NAME> TO rag_sql_reader;
GRANT USAGE ON SCHEMA approved_analytics TO rag_sql_reader;

-- One GRANT per approved, curated view - never a blanket
-- "ALL TABLES IN SCHEMA" grant, and never a raw operational table. Add a
-- line here, and a matching entry in app/sql/catalog_definition.json, every
-- time a new view is approved for Text-to-SQL access.
-- GRANT SELECT ON approved_analytics.<view_name> TO rag_sql_reader;

-- ---------------------------------------------------------------------------
-- 3. Row-Level Security: defense in depth, not a substitute for #2.
--
-- The application never asks the model to add tenant/scope predicates -
-- RLS enforces them server-side regardless of what SQL was generated (see
-- app.sql.sql_executor.PostgresReadOnlySQLExecutor._prepare_transaction,
-- which sets app.current_user/app.tenant_id session config on every
-- connection before a query runs). Adjust the USING expression to whatever
-- column(s) your approved views actually expose for scoping.
-- ---------------------------------------------------------------------------
-- ALTER TABLE approved_analytics.<view_name> ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE approved_analytics.<view_name> FORCE ROW LEVEL SECURITY;
--
-- CREATE POLICY sql_reader_tenant_select
-- ON approved_analytics.<view_name>
-- AS RESTRICTIVE
-- FOR SELECT
-- TO rag_sql_reader
-- USING (
--     tenant_id = NULLIF(current_setting('app.tenant_id', true), '')
-- );

-- ---------------------------------------------------------------------------
-- 4. Verification (run manually after the above, as a superuser):
--
--   SELECT rolname, rolsuper, rolcreatedb, rolcreaterole, rolbypassrls
--   FROM pg_roles WHERE rolname = 'rag_sql_reader';
--   -- rolsuper/rolcreatedb/rolcreaterole/rolbypassrls must all be false.
--
--   SET ROLE rag_sql_reader;
--   SELECT current_setting('transaction_read_only');  -- must be 'on'
--   INSERT INTO approved_analytics.<view_name> DEFAULT VALUES;  -- must fail
-- ---------------------------------------------------------------------------
