-- Conversation threads, grouping chat_history rows for the sidebar's
-- recency-grouped history panel. Backs app.repositories.conversation_repository.
--
-- Every existing chat_history row predates this table, so this migration
-- also backfills one synthetic conversation per user covering all of their
-- pre-existing turns, titled from their earliest question.
--
-- The migration runner (scripts/seed_db.py) has no tracking table and
-- re-executes every *.sql file on every run, so every statement here must
-- be safe to re-run indefinitely.

CREATE TABLE IF NOT EXISTS conversations (
    id         SERIAL PRIMARY KEY,
    username   VARCHAR(32) NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    title      VARCHAR(200) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_username_updated_at
    ON conversations (username, updated_at DESC);

ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS conversation_id INTEGER
    REFERENCES conversations(id) ON DELETE CASCADE;

-- Backfill, guarded by `conversation_id IS NULL` throughout: on every run
-- after the first, first_rows/spans/new_conversations are all empty, so the
-- INSERT and UPDATE below affect zero rows.
WITH first_rows AS (
    SELECT DISTINCT ON (username) username, question, created_at
    FROM chat_history WHERE conversation_id IS NULL
    ORDER BY username, created_at ASC
),
spans AS (
    SELECT username, min(created_at) AS first_at, max(created_at) AS last_at
    FROM chat_history WHERE conversation_id IS NULL
    GROUP BY username
),
new_conversations AS (
    INSERT INTO conversations (username, title, created_at, updated_at)
    SELECT f.username, left(f.question, 60), s.first_at, s.last_at
    FROM first_rows f JOIN spans s USING (username)
    RETURNING id, username
)
UPDATE chat_history SET conversation_id = nc.id
FROM new_conversations nc
WHERE chat_history.username = nc.username AND chat_history.conversation_id IS NULL;

-- Idempotent: re-asserting NOT NULL on an already-NOT-NULL column is a no-op.
ALTER TABLE chat_history ALTER COLUMN conversation_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chat_history_conversation_id_created_at
    ON chat_history (conversation_id, created_at ASC);
