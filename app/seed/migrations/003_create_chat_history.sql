-- Chat history table backing
-- app.repositories.chat_history_repository.PostgresChatHistoryRepository.
-- One row per answered question, keyed by the owning user's username (not
-- id, since app.models.auth_user.AuthenticatedUser only carries a username)
-- so each user can look back at what they previously asked.

CREATE TABLE IF NOT EXISTS chat_history (
    id         SERIAL PRIMARY KEY,
    username   VARCHAR(32) NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    sources    JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Supports "list this user's history, most recent first" without a sort.
CREATE INDEX IF NOT EXISTS idx_chat_history_username_created_at
    ON chat_history (username, created_at DESC);
