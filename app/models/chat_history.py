"""Internal ``ChatHistoryEntry`` domain model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ChatHistoryEntry:
    """A single persisted question/answer turn, owned by one user."""

    id: int
    username: str
    question: str
    answer: str
    sources: list[str]
    confidence: float
    created_at: datetime
