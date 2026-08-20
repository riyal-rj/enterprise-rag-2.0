"""Opt-in integration test against a real local Qdrant instance.

Unlike the rest of this repo's tests (all hand-rolled fakes, no
containers/CI), this hits the actual local Qdrant container to catch what
a fake client can't: wrong SDK class names, an invalid ``query_points``
payload the server rejects, etc. Skipped automatically unless Qdrant is
actually reachable at ``QDRANT_URL`` (default ``http://localhost:6333``,
matching ``QdrantSettings``) - no separate opt-in flag needed, and the
standard ``pytest tests/ -q`` run doesn't newly depend on Docker being up.

Run explicitly once the local Qdrant container (see docker-compose.yml)
is up:

    pytest tests/repositories/test_vector_repository_integration.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.ingestion.document_processor import DocumentChunk
from app.repositories.vector_repository import QdrantVectorRepository

_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _qdrant_reachable() -> bool:
    try:
        QdrantClient(url=_QDRANT_URL, timeout=2).get_collections()
        return True
    except Exception:  # noqa: BLE001 - any connection failure means "skip"
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_reachable(), reason=f"Qdrant not reachable at {_QDRANT_URL} - skipping integration test"
)


@pytest.fixture
def repository():
    client = QdrantClient(url=_QDRANT_URL, timeout=10)
    collection_name = f"test_vector_repository_integration_{uuid.uuid4().hex[:8]}"
    repo = QdrantVectorRepository(client=client, collection_name=collection_name)
    yield repo
    client.delete_collection(collection_name)


def test_collection_is_created_with_named_dense_and_sparse_vectors_and_a_source_index(
    repository: QdrantVectorRepository,
) -> None:
    info = repository._client.get_collection(repository._collection_name)  # noqa: SLF001
    assert "dense" in info.config.params.vectors
    assert info.config.params.sparse_vectors is not None
    assert "bm25" in info.config.params.sparse_vectors
    assert "source" in info.payload_schema


def test_upsert_then_dense_sparse_and_hybrid_search_all_find_the_point(
    repository: QdrantVectorRepository,
) -> None:
    chunks = [DocumentChunk(text="refund policy for wire transfers", source="a.pdf", page_number=1)]
    dense_embeddings = [[0.1] * 1536]
    sparse_embeddings = [SparseVector(indices=[10, 20], values=[1.5, 0.8])]

    repository.upsert_chunks(chunks, dense_embeddings, sparse_embeddings)

    dense_results = repository.search_dense(dense_embeddings[0], top_k=5)
    assert len(dense_results) == 1
    assert dense_results[0].source == "a.pdf"
    assert dense_results[0].page_number == 1

    sparse_results = repository.search_sparse(SparseVector(indices=[10], values=[1.0]), top_k=5)
    assert len(sparse_results) == 1
    assert sparse_results[0].source == "a.pdf"

    hybrid_results = repository.search_hybrid(
        query_embedding=dense_embeddings[0],
        query_sparse=SparseVector(indices=[10], values=[1.0]),
        top_k=5,
    )
    assert len(hybrid_results) == 1
    assert hybrid_results[0].source == "a.pdf"


def test_delete_by_source_removes_only_the_matching_points(
    repository: QdrantVectorRepository,
) -> None:
    chunks = [
        DocumentChunk(text="keep me", source="keep.pdf"),
        DocumentChunk(text="delete me", source="delete.pdf"),
    ]
    dense_embeddings = [[0.1] * 1536, [0.2] * 1536]
    sparse_embeddings = [
        SparseVector(indices=[1], values=[1.0]),
        SparseVector(indices=[2], values=[1.0]),
    ]
    repository.upsert_chunks(chunks, dense_embeddings, sparse_embeddings)

    repository.delete_by_source("delete.pdf")

    remaining = repository.scroll_all_chunks()
    assert [r["source"] for r in remaining] == ["keep.pdf"]


def test_quarantined_points_are_invisible_to_every_search_path(
    repository: QdrantVectorRepository,
) -> None:
    """The whole point of the ingestion-quarantine workflow: a freshly
    uploaded, unapproved document must not be retrievable by real chat
    traffic - see app.services.policy_ingestion_security_service."""
    chunks = [DocumentChunk(text="unapproved content", source="pending.pdf", page_number=1)]
    dense_embeddings = [[0.3] * 1536]
    sparse_embeddings = [SparseVector(indices=[30], values=[1.0])]

    repository.upsert_chunks(
        chunks, dense_embeddings, sparse_embeddings, ingestion_status="quarantined"
    )

    assert repository.search_dense(dense_embeddings[0], top_k=5) == []
    assert (
        repository.search_sparse(SparseVector(indices=[30], values=[1.0]), top_k=5) == []
    )
    assert (
        repository.search_hybrid(
            query_embedding=dense_embeddings[0],
            query_sparse=SparseVector(indices=[30], values=[1.0]),
            top_k=5,
        )
        == []
    )
    assert repository.scroll_all_chunks() == []  # default active-only listing


def test_set_ingestion_status_activates_a_quarantined_document(
    repository: QdrantVectorRepository,
) -> None:
    chunks = [DocumentChunk(text="soon to be approved", source="approve_me.pdf")]
    dense_embeddings = [[0.4] * 1536]
    sparse_embeddings = [SparseVector(indices=[40], values=[1.0])]
    repository.upsert_chunks(
        chunks, dense_embeddings, sparse_embeddings, ingestion_status="quarantined"
    )
    assert repository.search_dense(dense_embeddings[0], top_k=5) == []

    repository.set_ingestion_status("approve_me.pdf", "active")

    results = repository.search_dense(dense_embeddings[0], top_k=5)
    assert [r.source for r in results] == ["approve_me.pdf"]


def test_scroll_all_chunks_with_explicit_status_lists_only_that_status(
    repository: QdrantVectorRepository,
) -> None:
    active_chunk = [DocumentChunk(text="live", source="live.pdf")]
    quarantined_chunk = [DocumentChunk(text="pending", source="pending.pdf")]
    repository.upsert_chunks(
        active_chunk, [[0.5] * 1536], [SparseVector(indices=[50], values=[1.0])]
    )
    repository.upsert_chunks(
        quarantined_chunk,
        [[0.6] * 1536],
        [SparseVector(indices=[60], values=[1.0])],
        ingestion_status="quarantined",
    )

    active_only = repository.scroll_all_chunks(ingestion_status="active")
    quarantined_only = repository.scroll_all_chunks(ingestion_status="quarantined")
    everything = repository.scroll_all_chunks(ingestion_status=None)

    assert [r["source"] for r in active_only] == ["live.pdf"]
    assert [r["source"] for r in quarantined_only] == ["pending.pdf"]
    assert sorted(str(r["source"]) for r in everything) == ["live.pdf", "pending.pdf"]


def test_preexisting_points_with_no_ingestion_status_are_backfilled_to_active() -> None:
    """Upgrade safety: a collection populated before ingestion_status
    existed must not go dark - QdrantVectorRepository's construction must
    backfill those points to "active" rather than leaving them invisible
    to the new active-only search filter."""
    client = QdrantClient(url=_QDRANT_URL, timeout=10)
    collection_name = f"test_vector_repository_backfill_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        collection_name=collection_name,
        vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
        sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
    )
    # Simulates a point written before ingestion_status existed - no such
    # payload key at all, not merely a null value.
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": [0.7] * 1536,
                    "bm25": SparseVector(indices=[70], values=[1.0]),
                },
                payload={"text": "pre-migration content", "source": "legacy.pdf"},
            )
        ],
    )

    try:
        repository = QdrantVectorRepository(client=client, collection_name=collection_name)

        results = repository.search_dense([0.7] * 1536, top_k=5)

        assert [r.source for r in results] == ["legacy.pdf"]
    finally:
        client.delete_collection(collection_name)
