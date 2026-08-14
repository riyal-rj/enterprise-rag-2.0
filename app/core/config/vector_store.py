"""Vector database (Qdrant) connection configuration."""

from __future__ import annotations

from pydantic import Field

from app.core.config.base import EnvBaseSettings


class QdrantSettings(EnvBaseSettings):
    """Connection details for the Qdrant vector store."""

    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="documents")
    qdrant_timeout_seconds: float = Field(default=30.0, gt=0)
