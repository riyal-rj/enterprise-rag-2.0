"""Internal ``ConversationEntry`` domain model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ConversationEntry:
    """A conversation thread, owned by one user, that chat turns are grouped under."""

    id: int
    username: str
    title: str
    created_at: datetime
    updated_at: datetime
