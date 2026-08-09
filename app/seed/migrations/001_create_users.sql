-- Users table backing app.repositories.user_repository.PostgresUserRepository.
-- Column set/types intentionally mirror app.models.user.User and the
-- validation bounds in app.schemas.auth.RegisterRequest (username <= 32
-- chars). password_hash stores a bcrypt hash (see
-- app.core.security.passwords.BcryptPasswordHasher), never plaintext.

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(32) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
