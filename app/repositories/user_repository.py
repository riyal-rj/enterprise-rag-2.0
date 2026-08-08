"""User persistence, behind a swappable interface (Repository pattern)."""

from __future__ import annotations

from typing import Protocol

import psycopg2

from app.core.db import PostgresConnectionPool
from app.core.exceptions import UserAlreadyExistsError
from app.models.user import User


class UserRepository(Protocol):
    """Persistence contract for user records."""

    def get_by_username(self, username: str) -> User | None: ...

    def create(self, username: str, password_hash: str) -> User: ...


class PostgresUserRepository:
    """Postgres-backed :class:`UserRepository`."""

    def __init__(self, pool: PostgresConnectionPool) -> None:
        self._pool = pool

    def get_by_username(self, username: str) -> User | None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, password_hash, is_admin "
                "FROM users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return User(id=row[0], username=row[1], password_hash=row[2], is_admin=bool(row[3]))

    def create(self, username: str, password_hash: str) -> User:
        """Insert a new user.

        Raises:
            UserAlreadyExistsError: if ``username`` is already taken.
        """
        with self._pool.connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO users (username, password_hash) "
                        "VALUES (%s, %s) RETURNING id, is_admin",
                        (username, password_hash),
                    )
                    row = cur.fetchone()
                    if row is None:
                        raise RuntimeError("Failed to retrieve inserted user")
                    user_id, is_admin = row
                conn.commit()
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                raise UserAlreadyExistsError(username) from None
        return User(id=user_id, username=username, password_hash=password_hash, is_admin=bool(is_admin))
