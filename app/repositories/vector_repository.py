"""Vector-store persistence and similarity search (Repository pattern).

Same shape as ``UserRepository``/``PostgresUserRepository``: callers depend
on the ``VectorRepository`` Protocol, and ``QdrantVectorRepository`` wraps
the Qdrant SDK behind it. The client is built via ``build_qdrant_client``
and injected rather than constructed per call - consistent with
``build_redis_client``/``build_openai_client`` - so a single pooled HTTP
client is reused instead of opening a new one on every operation.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.ingestion.document_processor import DocumentChunk
from app.models.retrieved_chunk import RetrievedChunk

# Must match the output dimensionality of LLMSettings.embedding_model
# (text-embedding-3-small). Not exposed as a setting: changing the
# embedding model requires recreating the collection anyway, so there's
# no scenario where this varies independently at runtime.
_VECTOR_SIZE = 1536


def build_qdrant_client(url: str, timeout_seconds: float) -> QdrantClient:
    return QdrantClient(url=url, timeout=int(timeout_seconds))


class VectorRepository(Protocol):
    """Persistence contract for chunk vectors."""

    def upsert_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None: ...

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]: ...

    def scroll_all_chunks(self, limit: int = 10_000) -> list[dict[str, str | int | None]]: ...

    def delete_by_source(self, source: str) -> None: ...


class QdrantVectorRepository:
    """:class:`VectorRepository` backed by Qdrant.

    The collection is ensured to exist once, at construction time, rather
    than on every ``upsert_chunks`` call - this class is built once and
    cached (see ``app.api.deps.get_vector_repository``), so re-checking
    collection existence on every write would be a wasted round-trip.
    """

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name
        self._ensure_collection()

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            RetrievedChunk(
                text=str(point.payload.get("text", "")) if point.payload else "",
                source=str(point.payload.get("source", "")) if point.payload else "",
                score=float(point.score),
                page_number=point.payload.get("page_number") if point.payload else None,
            )
            for point in results
        ]

    def scroll_all_chunks(self, limit: int = 10_000) -> list[dict[str, str | int | None]]:
        """Full-collection listing for building an out-of-process sparse index.

        Returns raw payload dicts (including the Qdrant point id) rather
        than a typed model - callers here need the id, which isn't part
        of either :class:`DocumentChunk` or :class:`RetrievedChunk`.
        """
        points, _next_page = self._client.scroll(
            collection_name=self._collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "id": str(point.id),
                "text": str(point.payload.get("text", "")) if point.payload else "",
                "source": str(point.payload.get("source", "")) if point.payload else "",
                "page_number": point.payload.get("page_number") if point.payload else None,
            }
            for point in points
        ]

    def delete_by_source(self, source: str) -> None:
        """Delete every chunk previously ingested for ``source``.

        Used to replace a policy document on re-upload: the caller deletes
        the old chunks for that filename before upserting the new ones, so
        a re-ingested document doesn't leave stale duplicates alongside it.
        """
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
        )

    def _ensure_collection(self) -> None:
        existing = {collection.name for collection in self._client.get_collections().collections}
        if self._collection_name not in existing:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
            )
