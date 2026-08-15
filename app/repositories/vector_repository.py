"""Vector-store persistence and similarity search (Repository pattern).

Same shape as ``UserRepository``/``PostgresUserRepository``: callers depend
on the ``VectorRepository`` Protocol, and ``QdrantVectorRepository`` wraps
the Qdrant SDK behind it. The client is built via ``build_qdrant_client``
and injected rather than constructed per call - consistent with
``build_redis_client``/``build_openai_client`` - so a single pooled HTTP
client is reused instead of opening a new one on every operation.

Each point carries two named vectors - ``dense`` (OpenAI embedding) and
``bm25`` (FastEmbed sparse, see ``app.core.llm.sparse_embedding_client``) -
computed once at ingestion, not a single bare vector. ``search_hybrid``
fuses them server-side via Qdrant's Query API (prefetch + RRF), replacing
the earlier design that scrolled the whole collection and fit a
``TfidfVectorizer`` in-process on every hybrid request.
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
    Modifier,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    Rrf,
    RrfQuery,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.ingestion.document_processor import DocumentChunk
from app.models.retrieved_chunk import RetrievedChunk

# Must match the output dimensionality of LLMSettings.embedding_model
# (text-embedding-3-small). Not exposed as a setting: changing the
# embedding model requires recreating the collection anyway, so there's
# no scenario where this varies independently at runtime.
_VECTOR_SIZE = 1536
_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "bm25"


def build_qdrant_client(url: str, timeout_seconds: float) -> QdrantClient:
    return QdrantClient(url=url, timeout=int(timeout_seconds))


class VectorRepository(Protocol):
    """Persistence contract for chunk vectors (dense + BM25 sparse)."""

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[SparseVector],
    ) -> None: ...

    def search_dense(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]: ...

    def search_sparse(self, query_sparse: SparseVector, top_k: int = 5) -> list[RetrievedChunk]: ...

    def search_hybrid(
        self,
        query_embedding: list[float],
        query_sparse: SparseVector,
        top_k: int = 5,
        candidate_top_k: int = 20,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]: ...

    def scroll_all_chunks(self, limit: int = 10_000) -> list[dict[str, str | int | None]]: ...

    def delete_by_source(self, source: str) -> None: ...


class QdrantVectorRepository:
    """:class:`VectorRepository` backed by Qdrant.

    ``collection_name`` may be a physical collection name or an alias -
    Qdrant resolves both transparently for point/search/scroll/delete
    operations (only ``create_collection`` itself always creates a
    physical collection). ``_ensure_collection`` therefore checks
    ``collection_exists`` (alias-aware) rather than a literal name match
    against ``get_collections()`` (which lists physical collections only)
    - the latter would otherwise try to recreate a collection that
    already exists under an alias, e.g. during a blue/green cutover (see
    ``scripts/migrate_qdrant_hybrid.py``).

    The collection is ensured to exist once, at construction time, rather
    than on every ``upsert_chunks`` call - this class is built once and
    cached (see ``app.api.deps.get_vector_repository``), so re-checking
    collection existence on every write would be a wasted round-trip.
    """

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name
        self._ensure_collection()

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[SparseVector],
    ) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={_DENSE_VECTOR_NAME: dense, _SPARSE_VECTOR_NAME: sparse},
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                },
            )
            for chunk, dense, sparse in zip(chunks, dense_embeddings, sparse_embeddings, strict=True)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search_dense(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            using=_DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        ).points
        return [self._to_retrieved_chunk(point) for point in results]

    def search_sparse(self, query_sparse: SparseVector, top_k: int = 5) -> list[RetrievedChunk]:
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_sparse,
            using=_SPARSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        ).points
        return [self._to_retrieved_chunk(point) for point in results]

    def search_hybrid(
        self,
        query_embedding: list[float],
        query_sparse: SparseVector,
        top_k: int = 5,
        candidate_top_k: int = 20,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """Dense + sparse fusion, server-side, via Qdrant's Query API.

        ``RrfQuery(rrf=Rrf(k=rrf_k))`` (not the parameterless
        ``FusionQuery(fusion=Fusion.RRF)``) - confirmed against the live
        Qdrant 1.17.0 server that configurable-``k`` RRF is supported, so
        ``RAGFeatureSettings.rrf_k`` stays meaningful rather than becoming
        a dead setting.
        """
        results = self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                Prefetch(query=query_embedding, using=_DENSE_VECTOR_NAME, limit=candidate_top_k),
                Prefetch(query=query_sparse, using=_SPARSE_VECTOR_NAME, limit=candidate_top_k),
            ],
            query=RrfQuery(rrf=Rrf(k=rrf_k)),
            limit=top_k,
            with_payload=True,
        ).points
        return [self._to_retrieved_chunk(point) for point in results]

    def scroll_all_chunks(self, limit: int = 10_000) -> list[dict[str, str | int | None]]:
        """Full-collection listing for the admin policy list.

        Returns raw payload dicts (including the Qdrant point id) rather
        than a typed model - callers here need the id, which isn't part
        of either :class:`DocumentChunk` or :class:`RetrievedChunk`. Used
        by ``PolicyIngestionService.list_policies()``; no longer used for
        sparse-index fitting (that whole approach is gone - see
        ``search_sparse``/``search_hybrid``).
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
        if self._client.collection_exists(self._collection_name):
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={_DENSE_VECTOR_NAME: VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE)},
            sparse_vectors_config={_SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
        )
        # IDF weighting (above) is computed server-side from this index -
        # real corpus statistics, not something baked into the FastEmbed
        # BM25 encoding itself (which is stateless per-text term weights).
        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    @staticmethod
    def _to_retrieved_chunk(point: ScoredPoint) -> RetrievedChunk:
        return RetrievedChunk(
            text=str(point.payload.get("text", "")) if point.payload else "",
            source=str(point.payload.get("source", "")) if point.payload else "",
            score=float(point.score),
            page_number=point.payload.get("page_number") if point.payload else None,
        )
