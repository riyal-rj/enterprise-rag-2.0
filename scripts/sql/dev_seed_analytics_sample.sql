-- =============================================================================
-- DEV/LOCAL TESTING ONLY - not the production provisioning script.
--
-- Creates a small, entirely synthetic "approved_analytics" schema inside the
-- SAME local docker-compose Postgres (adv_rag) that already hosts users/
-- chat_history/rag_ops_config, plus a genuinely separate, SELECT-only role
-- pointed at it - so SQL_DATABASE_URL can be a real, different credential
-- from DATABASE_URL even though this dev environment only has one Postgres
-- container. Production must use scripts/sql/provision_sql_reader_role.sql
-- against a real, physically separate analytics database instead - see that
-- script's header and the architecture blueprint, section 13.
--
-- Safe to re-run: every statement is idempotent.
--
-- Run once (uses the superuser credential in DATABASE_URL, same as
-- scripts/seed_db.py's migrations):
--   docker exec -i enterprise-rag-20-postgres-1 psql -U postgres -d adv_rag \
--     < scripts/sql/dev_seed_analytics_sample.sql
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS approved_analytics;

CREATE TABLE IF NOT EXISTS approved_analytics.branches (
    id     SERIAL PRIMARY KEY,
    name   TEXT NOT NULL,
    city   TEXT NOT NULL,
    region TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approved_analytics.customers (
    id         SERIAL PRIMARY KEY,
    full_name  TEXT NOT NULL,
    email      TEXT NOT NULL,
    ssn        TEXT NOT NULL,
    segment    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS approved_analytics.accounts (
    id           SERIAL PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES approved_analytics.customers (id),
    branch_id    INTEGER NOT NULL REFERENCES approved_analytics.branches (id),
    account_type TEXT NOT NULL,
    balance      NUMERIC(14, 2) NOT NULL,
    opened_at    TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS approved_analytics.transactions (
    id               SERIAL PRIMARY KEY,
    account_id       INTEGER NOT NULL REFERENCES approved_analytics.accounts (id),
    amount           NUMERIC(14, 2) NOT NULL,
    transaction_type TEXT NOT NULL,
    occurred_at      TIMESTAMPTZ NOT NULL
);

-- --- Sample data (fabricated - no real people, no real accounts) ----------

INSERT INTO approved_analytics.branches (id, name, city, region) VALUES
    (1, 'Meridian Trust - Downtown',   'Mumbai',    'West'),
    (2, 'Meridian Trust - Andheri',    'Mumbai',    'West'),
    (3, 'Meridian Trust - Connaught',  'Delhi',     'North'),
    (4, 'Meridian Trust - Koramangala','Bengaluru', 'South'),
    (5, 'Meridian Trust - Salt Lake',  'Kolkata',   'East')
ON CONFLICT (id) DO NOTHING;
SELECT setval('approved_analytics.branches_id_seq', 5, true);

INSERT INTO approved_analytics.customers (id, full_name, email, ssn, segment, created_at)
SELECT
    n,
    'Customer ' || n,
    'customer' || n || '@example.test',
    lpad((100000000 + n)::text, 9, '0'),
    (ARRAY['retail', 'premium', 'business'])[1 + (n % 3)],
    now() - ((36 - (n % 36)) || ' months')::interval
FROM generate_series(1, 40) AS n
ON CONFLICT (id) DO NOTHING;
SELECT setval('approved_analytics.customers_id_seq', 40, true);

INSERT INTO approved_analytics.accounts (id, customer_id, branch_id, account_type, balance, opened_at)
SELECT
    n,
    1 + (n % 40),
    1 + (n % 5),
    (ARRAY['checking', 'savings', 'fixed_deposit'])[1 + (n % 3)],
    round((1000 + random() * 500000)::numeric, 2),
    now() - ((24 - (n % 24)) || ' months')::interval
FROM generate_series(1, 60) AS n
ON CONFLICT (id) DO NOTHING;
SELECT setval('approved_analytics.accounts_id_seq', 60, true);

INSERT INTO approved_analytics.transactions (id, account_id, amount, transaction_type, occurred_at)
SELECT
    n,
    1 + (n % 60),
    round(((random() - 0.4) * 20000)::numeric, 2),
    (ARRAY['deposit', 'withdrawal', 'transfer', 'fee'])[1 + (n % 4)],
    now() - ((12 - (n % 12)) || ' months')::interval - ((n % 28) || ' days')::interval
FROM generate_series(1, 400) AS n
ON CONFLICT (id) DO NOTHING;
SELECT setval('approved_analytics.transactions_id_seq', 400, true);

-- --- Read-only role, separate credential from DATABASE_URL -----------------

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_sql_reader_dev') THEN
        CREATE ROLE rag_sql_reader_dev LOGIN PASSWORD 'dev-only-not-a-secret'
            NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;
    END IF;
END $$;

ALTER ROLE rag_sql_reader_dev SET default_transaction_read_only = on;
ALTER ROLE rag_sql_reader_dev SET statement_timeout = '5s';
ALTER ROLE rag_sql_reader_dev SET lock_timeout = '500ms';

GRANT CONNECT ON DATABASE adv_rag TO rag_sql_reader_dev;
GRANT USAGE ON SCHEMA approved_analytics TO rag_sql_reader_dev;
GRANT SELECT ON ALL TABLES IN SCHEMA approved_analytics TO rag_sql_reader_dev;
-- Every table in this schema is synthetic sample data, so a blanket grant
-- is fine here - scripts/sql/provision_sql_reader_role.sql (production)
-- deliberately grants view-by-view instead; do not copy this shortcut there.
