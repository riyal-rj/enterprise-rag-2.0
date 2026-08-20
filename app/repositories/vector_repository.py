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

import logging
import uuid
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchValue,
    Modifier,
    PayloadField,
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

logger = logging.getLogger(__name__)

# Must match the output dimensionality of LLMSettings.embedding_model
# (text-embedding-3-small). Not exposed as a setting: changing the
# embedding model requires recreating the collection anyway, so there's
# no scenario where this varies independently at runtime.
_VECTOR_SIZE = 1536
_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "bm25"

# Ingestion lifecycle state carried on every point's payload, mirroring
# document_security_state.status (see
# app/seed/migrations/014_create_document_security_state.sql and
# app.services.policy_ingestion_security_service). Only ACTIVE points are
# ever visible to retrieval - a freshly uploaded document is upserted as
# QUARANTINED and stays unsearchable until an admin approves it.
INGESTION_STATUS_ACTIVE = "active"
INGESTION_STATUS_QUARANTINED = "quarantined"

_ACTIVE_ONLY_FILTER = Filter(
    must=[
        FieldCondition(
            key="ingestion_status", match=MatchValue(value=INGESTION_STATUS_ACTIVE)
        )
    ]
)


def build_qdrant_client(url: str, timeout_seconds: float) -> QdrantClient:
    return QdrantClient(url=url, timeout=int(timeout_seconds))


class VectorRepository(Protocol):
    """Persistence contract for chunk vectors (dense + BM25 sparse)."""

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[SparseVector],
        ingestion_status: str = INGESTION_STATUS_ACTIVE,
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

    def scroll_all_chunks(
        self, limit: int = 10_000, ingestion_status: str | None = INGESTION_STATUS_ACTIVE
    ) -> list[dict[str, str | int | None]]: ...

    def set_ingestion_status(self, source: str, ingestion_status: str) -> None: ...

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
        ingestion_status: str = INGESTION_STATUS_ACTIVE,
    ) -> None:
        """``ingestion_status`` defaults to ``active`` so every pre-existing
        caller (``scripts/seed_db.py``, the eval harness, tests) keeps its
        original behavior; the quarantine upload path passes
        ``quarantined`` explicitly - see
        ``app.services.policy_ingestion_security_service``."""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={_DENSE_VECTOR_NAME: dense, _SPARSE_VECTOR_NAME: sparse},
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "ingestion_status": ingestion_status,
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
            query_filter=_ACTIVE_ONLY_FILTER,
        ).points
        return [self._to_retrieved_chunk(point) for point in results]

    def search_sparse(self, query_sparse: SparseVector, top_k: int = 5) -> list[RetrievedChunk]:
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_sparse,
            using=_SPARSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
            query_filter=_ACTIVE_ONLY_FILTER,
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
            query_filter=_ACTIVE_ONLY_FILTER,
        ).points
        return [self._to_retrieved_chunk(point) for point in results]

    def scroll_all_chunks(
        self, limit: int = 10_000, ingestion_status: str | None = INGESTION_STATUS_ACTIVE
    ) -> list[dict[str, str | int | None]]:
        """Full-collection listing for the admin policy list.

        Returns raw payload dicts (including the Qdrant point id) rather
        than a typed model - callers here need the id, which isn't part
        of either :class:`DocumentChunk` or :class:`RetrievedChunk`. Used
        by ``PolicyIngestionService.list_policies()``; no longer used for
        sparse-index fitting (that whole approach is gone - see
        ``search_sparse``/``search_hybrid``).

        ``ingestion_status`` defaults to ``active`` so the admin policy
        list keeps showing only live documents; pass ``None`` to include
        quarantined/pending ones too.
        """
        scroll_filter = (
            None
            if ingestion_status is None
            else Filter(
                must=[
                    FieldCondition(
                        key="ingestion_status", match=MatchValue(value=ingestion_status)
                    )
                ]
            )
        )
        points, _next_page = self._client.scroll(
            collection_name=self._collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            scroll_filter=scroll_filter,
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

    def set_ingestion_status(self, source: str, ingestion_status: str) -> None:
        """Flip every chunk of ``source`` to ``ingestion_status`` in place.

        A targeted payload update, not a re-upsert: publishing an approved
        document must not re-chunk or re-embed it (that work already
        happened at upload time - see
        ``app.services.policy_ingestion_security_service``).
        """
        self._client.set_payload(
            collection_name=self._collection_name,
            payload={"ingestion_status": ingestion_status},
            points=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )

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
            self._ensure_ingestion_status_backfilled()
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
        # Backs the active-only filter every search path applies - without
        # an index, that filter would be a full scan on every query.
        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="ingestion_status",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def _ensure_ingestion_status_backfilled(self) -> None:
        """Make an already-populated collection safe for the active-only
        search filter.

        Points written before ``ingestion_status`` existed carry no such
        payload field, so the filter every search path now applies would
        make the entire pre-existing corpus invisible - a silent, total
        retrieval outage on upgrade. Backfilling them to ``active``
        preserves exactly the behavior those documents already had (they
        were live and searchable), while keeping the filter itself strict
        rather than treating "no status" as implicitly active forever.

        Idempotent and cheap on an already-migrated collection: the
        ``is_empty`` filter matches nothing once every point has a value.
        Also (re)creates the payload index, which an older collection
        won't have.
        """
        try:
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="ingestion_status",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self._client.set_payload(
                collection_name=self._collection_name,
                payload={"ingestion_status": INGESTION_STATUS_ACTIVE},
                points=Filter(
                    must=[IsEmptyCondition(is_empty=PayloadField(key="ingestion_status"))]
                ),
            )
        except Exception:  # noqa: BLE001 - deliberate: never block startup on backfill
            # A backfill failure must not take the process down, but it
            # does mean pre-existing points stay unsearchable until it
            # succeeds, so it's logged loudly rather than swallowed.
            logger.exception(
                "vector_repository.ingestion_status_backfill_failed",
                extra={"collection": self._collection_name},
            )

    @staticmethod
    def _to_retrieved_chunk(point: ScoredPoint) -> RetrievedChunk:
        return RetrievedChunk(
            text=str(point.payload.get("text", "")) if point.payload else "",
            source=str(point.payload.get("source", "")) if point.payload else "",
            score=float(point.score),
            page_number=point.payload.get("page_number") if point.payload else None,
        )
