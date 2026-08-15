"""Chat history persistence, behind a swappable interface (Repository pattern)."""

from __future__ import annotations

from typing import Protocol

from psycopg2.extras import Json

from app.core.db import PostgresConnectionPool
from app.models.chat_history import ChatHistoryEntry


class ChatHistoryRepository(Protocol):
    """Persistence contract for a user's past question/answer turns."""

    def create(
        self,
        username: str,
        conversation_id: int,
        question: str,
        answer: str,
        sources: list[str],
        confidence: float,
    ) -> ChatHistoryEntry: ...

    def list_by_user(self, username: str, limit: int, offset: int) -> list[ChatHistoryEntry]: ...

    def list_by_conversation(
        self, conversation_id: int, limit: int = 200
    ) -> list[ChatHistoryEntry]: ...


class PostgresChatHistoryRepository:
    """Postgres-backed :class:`ChatHistoryRepository`."""

    def __init__(self, pool: PostgresConnectionPool) -> None:
        self._pool = pool

    def create(
        self,
        username: str,
        conversation_id: int,
        question: str,
        answer: str,
        sources: list[str],
        confidence: float,
    ) -> ChatHistoryEntry:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_history "
                    "(username, conversation_id, question, answer, sources, confidence) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "RETURNING id, created_at",
                    (username, conversation_id, question, answer, Json(sources), confidence),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("Failed to retrieve inserted chat history entry")
                entry_id, created_at = row
            conn.commit()
        return ChatHistoryEntry(
            id=entry_id,
            username=username,
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            created_at=created_at,
        )

    def list_by_user(self, username: str, limit: int, offset: int) -> list[ChatHistoryEntry]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, question, answer, sources, confidence, created_at "
                "FROM chat_history WHERE username = %s "
                "ORDER BY created_at DESC "
                "LIMIT %s OFFSET %s",
                (username, limit, offset),
            )
            rows = cur.fetchall()
        return [
            ChatHistoryEntry(
                id=row[0],
                username=username,
                question=row[1],
                answer=row[2],
                sources=list(row[3]),
                confidence=row[4],
                created_at=row[5],
            )
            for row in rows
        ]

    def list_by_conversation(
        self, conversation_id: int, limit: int = 200
    ) -> list[ChatHistoryEntry]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, question, answer, sources, confidence, created_at "
                "FROM chat_history WHERE conversation_id = %s "
                "ORDER BY created_at ASC "
                "LIMIT %s",
                (conversation_id, limit),
            )
            rows = cur.fetchall()
        return [
            ChatHistoryEntry(
                id=row[0],
                username=row[1],
                question=row[2],
                answer=row[3],
                sources=list(row[4]),
                confidence=row[5],
                created_at=row[6],
            )
            for row in rows
        ]
