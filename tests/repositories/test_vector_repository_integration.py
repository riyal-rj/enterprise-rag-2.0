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
from qdrant_client.models import SparseVector

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
