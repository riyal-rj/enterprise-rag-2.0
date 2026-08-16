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
simply stale and the caller should try the next-nearest candidate (see
``find_candidates``) before falling back to generating a fresh answer.
"""

from __future__ import annotations

import threading
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

from app.rag_services.rag_runtime_config import RagRuntimeConfigStore

# Must match the output dimensionality of LLMSettings.embedding_model
# (text-embedding-3-small) - same invariant as
# app.repositories.vector_repository._VECTOR_SIZE, since both index the
# same embedding client's output.
_VECTOR_SIZE = 1536

# How many nearest neighbors to consider per lookup. Only the single
# nearest point is usually live, but if its Redis-backed answer expired
# ahead of the next-nearest match, checking only the top-1 would treat a
# perfectly good semantic hit as a miss. Small and fixed rather than a
# setting - this is an implementation deepening of the recall guarantee,
# not something a deployment should need to tune.
_CANDIDATE_LIMIT = 5


class SemanticQueryCache(Protocol):
    """Paraphrase-aware cache lookup: previously-answered questions by
    embedding similarity, within the same retrieval-mode/reranker/top_k
    partition."""

    def find_candidates(
        self, *, query_embedding: list[float], cache_namespace: str, top_k: int
    ) -> list[str]:
        """Return exact-match cache keys of previously-cached questions in
        this partition that clear the configured similarity threshold,
        nearest-first. Empty if nothing clears it. Callers should try each
        key in order (via the exact-match cache) until one is still live,
        since the nearest embedding match's underlying answer may have
        expired."""
        ...

    def record(
        self, *, query_embedding: list[float], cache_namespace: str, top_k: int, cache_key: str
    ) -> None:
        """Index a freshly generated answer's question so future
        paraphrases can find it."""
        ...

    def clear(self) -> int:
        """Delete every indexed point. Returns the number removed.

        Called from the same admin action that clears the Redis-backed
        exact-match cache (see ``AdminController.cache_clear``) - without
        this, a Redis clear (e.g. after a corpus re-ingestion) leaves
        semantic pointers that still resolve to stale answers as soon as
        Redis repopulates under the same keys.
        """
        ...


class QdrantSemanticQueryCache:
    """:class:`SemanticQueryCache` backed by a dedicated Qdrant collection.

    Deliberately its own collection, not a payload addition to the main
    documents collection (``QdrantVectorRepository``) - different lifecycle
    (grows/shrinks with query traffic, not ingestion) and a different
    partitioning key (``cache_namespace``/``top_k``, not ``source``).

    The collection is created lazily, on the first real operation
    (``find_candidates``/``record``/``clear``), not at construction time.
    ``app.api.deps.get_rag_service`` always constructs this class regardless
    of whether semantic caching is currently enabled (so an admin can flip
    it on later without a cold-construction cost) - eagerly ensuring the
    Qdrant collection in ``__init__`` would reintroduce a hard Qdrant
    dependency for deployments that keep semantic caching disabled.

    ``similarity_threshold`` is read from a shared ``RagRuntimeConfigStore``
    (see ``app.rag_services.rag_runtime_config``) instead of an independently
    mutable instance field, so an admin's threshold change is applied via
    the exact same atomic snapshot swap ``RAGService``/``DynamicReranker``
    observe - not a fourth, separately-timed mutation.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        config_store: RagRuntimeConfigStore,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._config_store = config_store
        self._collection_ready = False
        self._collection_ready_lock = threading.Lock()

    def find_candidates(
        self, *, query_embedding: list[float], cache_namespace: str, top_k: int
    ) -> list[str]:
        self._ensure_collection_ready()
        similarity_threshold = self._config_store.current.semantic_cache_threshold
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            query_filter=self._partition_filter(cache_namespace, top_k),
            limit=_CANDIDATE_LIMIT,
            with_payload=True,
        ).points

        candidates: list[str] = []
        for point in results:
            if point.score < similarity_threshold:
                break  # results are score-sorted descending; nothing after this clears it either
            payload = point.payload or {}
            cache_key = payload.get("cache_key")
            if cache_key is not None:
                candidates.append(str(cache_key))

        # Recording a "hit"/"miss" metric here would only mean "the
        # embedding search found a clearing match", not that the exact-match
        # Redis entry it points at is confirmed still live - RAGService
        # tries each candidate and falls through on a stale one, so it's the
        # one place that knows the real, confirmed outcome. See
        # ``RAGService._check_semantic_cache``.
        return candidates

    def record(
        self, *, query_embedding: list[float], cache_namespace: str, top_k: int, cache_key: str
    ) -> None:
        self._ensure_collection_ready()
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

    def clear(self) -> int:
        self._ensure_collection_ready()
        count = self._client.count(collection_name=self._collection_name).count
        self._client.delete_collection(collection_name=self._collection_name)
        self._ensure_collection()
        return count

    def _partition_filter(self, cache_namespace: str, top_k: int) -> Filter:
        return Filter(
            must=[
                FieldCondition(key="cache_namespace", match=MatchValue(value=cache_namespace)),
                FieldCondition(key="top_k", match=MatchValue(value=top_k)),
            ]
        )

    def _ensure_collection_ready(self) -> None:
        """Thread-safe, run-once gate in front of ``_ensure_collection`` -
        concurrent requests can both reach the first real operation on a
        freshly enabled cache before either has created the collection, and
        ``_ensure_collection``'s existence-check-then-create isn't atomic
        against that race."""
        if self._collection_ready:
            return
        with self._collection_ready_lock:
            if self._collection_ready:
                return
            self._ensure_collection()
            self._collection_ready = True

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
