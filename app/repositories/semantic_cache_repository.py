"""Semantic (embedding-similarity) query cache persistence (Repository pattern).

Backs ``RAGService``'s paraphrase-aware cache lookup: two questions with
different wording but the same meaning should still hit the cache. A plain
key/value store (``QueryCacheService``) can only match identical strings;
this repository indexes each cached question's embedding in Qdrant so a new
question can be matched against the nearest previously-answered one by
cosine similarity instead of exact text.

Partitioned by ``(cache_namespace, top_k)`` - the same partitioning
``RAGService``'s exact-match cache key uses - so a semantic hit can never
cross retrieval mode/reranker config or return a different chunk count than
requested. Stores only a pointer (``cache_key``) into the existing
``CacheTier.RAG_ANSWER`` Redis entry, not a duplicate copy of the answer -
if that Redis entry has since expired or been cleared, the pointer is
simply stale and the caller falls back to generating a fresh answer.
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
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

# Must match the output dimensionality of LLMSettings.embedding_model
# (text-embedding-3-small) - same invariant as
# app.repositories.vector_repository._VECTOR_SIZE, since both index the
# same embedding client's output.
_VECTOR_SIZE = 1536


class SemanticQueryCache(Protocol):
    """Paraphrase-aware cache lookup: nearest previously-answered question
    by embedding similarity, within the same retrieval-mode/reranker/top_k
    partition."""

    def find_similar(
        self, *, query_embedding: list[float], cache_namespace: str, top_k: int
    ) -> str | None:
        """Return the exact-match cache key of the closest previously
        cached question in this partition, or ``None`` if nothing clears
        the configured similarity threshold."""
        ...

    def record(
        self, *, query_embedding: list[float], cache_namespace: str, top_k: int, cache_key: str
    ) -> None:
        """Index a freshly generated answer's question so future
        paraphrases can find it."""
        ...


class QdrantSemanticQueryCache:
    """:class:`SemanticQueryCache` backed by a dedicated Qdrant collection.

    Deliberately its own collection, not a payload addition to the main
    documents collection (``QdrantVectorRepository``) - different lifecycle
    (grows/shrinks with query traffic, not ingestion) and a different
    partitioning key (``cache_namespace``/``top_k``, not ``source``).
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        similarity_threshold: float,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        self._client = client
        self._collection_name = collection_name
        self._similarity_threshold = similarity_threshold
        self._ensure_collection()

    def find_similar(
        self, *, query_embedding: list[float], cache_namespace: str, top_k: int
    ) -> str | None:
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            query_filter=self._partition_filter(cache_namespace, top_k),
            limit=1,
            with_payload=True,
        ).points

        if not results:
            return None

        best = results[0]
        if best.score < self._similarity_threshold:
            return None

        payload = best.payload or {}
        cache_key = payload.get("cache_key")
        return str(cache_key) if cache_key is not None else None

    def record(
        self, *, query_embedding: list[float], cache_namespace: str, top_k: int, cache_key: str
    ) -> None:
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=query_embedding,
            payload={
                "cache_namespace": cache_namespace,
                "top_k": top_k,
                "cache_key": cache_key,
            },
        )
        self._client.upsert(collection_name=self._collection_name, points=[point])

    def _partition_filter(self, cache_namespace: str, top_k: int) -> Filter:
        return Filter(
            must=[
                FieldCondition(key="cache_namespace", match=MatchValue(value=cache_namespace)),
                FieldCondition(key="top_k", match=MatchValue(value=top_k)),
            ]
        )

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection_name):
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="cache_namespace",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="top_k",
            field_schema=PayloadSchemaType.INTEGER,
        )
