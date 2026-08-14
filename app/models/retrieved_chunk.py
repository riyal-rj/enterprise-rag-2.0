"""Internal ``RetrievedChunk`` domain model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk is returned by a similarity or a text search, with its score."""

    text: str
    source: str
    score: float
