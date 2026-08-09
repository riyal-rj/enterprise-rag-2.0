-- Dummy users for local development/testing only.
-- Run after 001_create_users.sql. Safe to re-run (ON CONFLICT DO NOTHING).
--
-- Plaintext passwords (never stored — shown here only so you can log in
-- locally against these seeded rows):
--   admin / AdminPass123!   (is_admin = true)
--   alice / AlicePass123!
--   bob   / BobPass123!
--   carol / CarolPass123!
--
-- Hashes below were generated with bcrypt rounds=12, matching
-- app.core.security.passwords.BcryptPasswordHasher's default, so
-- POST /auth/login verifies against them as-is.

INSERT INTO users (username, password_hash, is_admin) VALUES
    ('admin', '$2b$12$xnwE6yFUVvOlb8t2ZNQzpeNW84gGoNGg/m0Re33onWY7rIiJ5MlBW', TRUE),
    ('alice', '$2b$12$jtfWlz5gTo5xAJ994RhtreXK15lSPk34n8q6pHToFML/QYYGwC9r2', FALSE),
    ('bob',   '$2b$12$DkBHjn4VcCdkx9OBwychseDGON5QGCSSGEs1fU9CcKc2hX6FXV58W', FALSE),
    ('carol', '$2b$12$pkjvtX9aEAbbRUDQ/mnWGuGCZ3K6jQacTl1PbAKw4HQgK4gyyj05C', FALSE)
ON CONFLICT (username) DO NOTHING;
