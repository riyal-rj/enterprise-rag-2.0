"""Conversation (thread) persistence, behind a swappable interface (Repository pattern)."""

from __future__ import annotations

from typing import Protocol

from app.core.db import PostgresConnectionPool
from app.models.conversation import ConversationEntry


class ConversationRepository(Protocol):
    """Persistence contract for a user's conversation threads."""

    def create(self, username: str, title: str) -> ConversationEntry: ...

    def get_owned(self, conversation_id: int, username: str) -> ConversationEntry | None: ...

    def list_by_user(
        self, username: str, search: str | None, limit: int, offset: int
    ) -> list[ConversationEntry]: ...

    def touch(self, conversation_id: int) -> None: ...


class PostgresConversationRepository:
    """Postgres-backed :class:`ConversationRepository`."""

    def __init__(self, pool: PostgresConnectionPool) -> None:
        self._pool = pool

    def create(self, username: str, title: str) -> ConversationEntry:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversations (username, title) "
                    "VALUES (%s, %s) "
                    "RETURNING id, created_at, updated_at",
                    (username, title),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("Failed to retrieve inserted conversation")
                conversation_id, created_at, updated_at = row
            conn.commit()
        return ConversationEntry(
            id=conversation_id,
            username=username,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
        )

    def get_owned(self, conversation_id: int, username: str) -> ConversationEntry | None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT title, created_at, updated_at FROM conversations "
                "WHERE id = %s AND username = %s",
                (conversation_id, username),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return ConversationEntry(
            id=conversation_id,
            username=username,
            title=row[0],
            created_at=row[1],
            updated_at=row[2],
        )

    def list_by_user(
        self, username: str, search: str | None, limit: int, offset: int
    ) -> list[ConversationEntry]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            if search:
                cur.execute(
                    "SELECT id, title, created_at, updated_at FROM conversations "
                    "WHERE username = %s AND title ILIKE %s "
                    "ORDER BY updated_at DESC "
                    "LIMIT %s OFFSET %s",
                    (username, f"%{search}%", limit, offset),
                )
            else:
                cur.execute(
                    "SELECT id, title, created_at, updated_at FROM conversations "
                    "WHERE username = %s "
                    "ORDER BY updated_at DESC "
                    "LIMIT %s OFFSET %s",
                    (username, limit, offset),
                )
            rows = cur.fetchall()
        return [
            ConversationEntry(
                id=row[0], username=username, title=row[1], created_at=row[2], updated_at=row[3]
            )
            for row in rows
        ]

    def touch(self, conversation_id: int) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversations SET updated_at = now() WHERE id = %s",
                    (conversation_id,),
                )
            conn.commit()
