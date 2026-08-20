from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    Prefetch,
    Rrf,
    RrfQuery,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.ingestion.document_processor import DocumentChunk
from app.models.retrieved_chunk import RetrievedChunk
from app.repositories.vector_repository import QdrantVectorRepository


@dataclass
class _FakeCollection:
    name: str


@dataclass
class _FakeCollectionsResponse:
    collections: list[_FakeCollection]


@dataclass
class _FakeScoredPoint:
    payload: dict[str, str | int] | None
    score: float


@dataclass
class _FakeQueryResponse:
    points: list[_FakeScoredPoint]


@dataclass
class _FakeRecord:
    id: str
    payload: dict[str, str | int] | None


class _FakeQdrantClient:
    def __init__(self, existing_collections: list[str] | None = None) -> None:
        self._existing = set(existing_collections or [])
        self.created_collections: list[dict[str, object]] = []
        self.created_payload_indexes: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []
        self.set_payload_calls: list[dict[str, object]] = []
        self.query_points_calls: list[dict[str, object]] = []
        self.scroll_calls: list[dict[str, object]] = []
        self.query_response: _FakeQueryResponse = _FakeQueryResponse(points=[])
        self.scroll_response: tuple[list[_FakeRecord], None] = ([], None)

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._existing

    def get_collections(self) -> _FakeCollectionsResponse:
        return _FakeCollectionsResponse(
            collections=[_FakeCollection(name=name) for name in self._existing]
        )

    def create_collection(self, **kwargs: object) -> None:
        self.created_collections.append(kwargs)
        self._existing.add(cast(str, kwargs["collection_name"]))

    def create_payload_index(self, **kwargs: object) -> None:
        self.created_payload_indexes.append(kwargs)

    def upsert(self, **kwargs: object) -> None:
        self.upsert_calls.append(kwargs)

    def query_points(self, **kwargs: object) -> _FakeQueryResponse:
        self.query_points_calls.append(kwargs)
        return self.query_response

    def scroll(self, **kwargs: object) -> tuple[list[_FakeRecord], None]:
        self.scroll_calls.append(kwargs)
        return self.scroll_response

    def delete(self, **kwargs: object) -> None:
        self.delete_calls.append(kwargs)

    def set_payload(self, **kwargs: object) -> None:
        self.set_payload_calls.append(kwargs)


def _repo(fake: _FakeQdrantClient, collection_name: str = "docs") -> QdrantVectorRepository:
    return QdrantVectorRepository(client=cast(QdrantClient, fake), collection_name=collection_name)


def test_init_creates_collection_with_named_dense_and_sparse_vectors_when_missing() -> None:
    fake = _FakeQdrantClient(existing_collections=[])

    _repo(fake)

    assert len(fake.created_collections) == 1
    created = fake.created_collections[0]
    assert created["collection_name"] == "docs"
    assert created["vectors_config"] == {"dense": VectorParams(size=1536, distance=Distance.COSINE)}
    assert created["sparse_vectors_config"] == {"bm25": SparseVectorParams(modifier=Modifier.IDF)}


def test_init_creates_payload_indexes_on_source_and_ingestion_status_when_collection_is_missing() -> (
    None
):
    fake = _FakeQdrantClient(existing_collections=[])

    _repo(fake)

    field_names = {idx["field_name"] for idx in fake.created_payload_indexes}
    assert field_names == {"source", "ingestion_status"}
    assert all(idx["collection_name"] == "docs" for idx in fake.created_payload_indexes)


def test_init_does_not_recreate_an_existing_collection() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])

    _repo(fake)

    assert fake.created_collections == []


def test_init_backfills_ingestion_status_on_an_existing_collection() -> None:
    """Upgrade safety: a collection that predates ingestion_status must
    have its points backfilled to "active" so the new active-only search
    filter doesn't make the entire pre-existing corpus invisible - see
    QdrantVectorRepository._ensure_ingestion_status_backfilled."""
    fake = _FakeQdrantClient(existing_collections=["docs"])

    _repo(fake)

    assert len(fake.set_payload_calls) == 1
    assert fake.set_payload_calls[0]["payload"] == {"ingestion_status": "active"}
    # Also (re)creates the index a pre-migration collection won't have.
    assert any(
        idx["field_name"] == "ingestion_status" for idx in fake.created_payload_indexes
    )


def test_init_treats_an_alias_as_an_existing_collection() -> None:
    """``collection_exists`` (not a literal ``get_collections()`` name match)
    is what gates creation - Qdrant resolves aliases transparently, and a
    blue/green cutover (see scripts/migrate_qdrant_hybrid.py) repoints the
    configured name to an alias, not always a physical collection."""
    fake = _FakeQdrantClient(existing_collections=[])
    fake.collection_exists = lambda name: name == "docs"  # type: ignore[method-assign]

    _repo(fake)

    assert fake.created_collections == []


def test_upsert_chunks_sends_named_dense_and_sparse_vectors_with_payload() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake)
    chunks = [
        DocumentChunk(text="hello", source="a.pdf", page_number=3),
        DocumentChunk(text="world", source="b.pdf"),
    ]
    dense_embeddings = [[0.1, 0.2], [0.3, 0.4]]
    sparse_embeddings = [
        SparseVector(indices=[1, 2], values=[0.5, 0.6]),
        SparseVector(indices=[3], values=[0.7]),
    ]

    repo.upsert_chunks(chunks, dense_embeddings, sparse_embeddings)

    assert len(fake.upsert_calls) == 1
    points = cast(list, fake.upsert_calls[0]["points"])
    assert len(points) == 2
    assert points[0].payload == {
        "text": "hello",
        "source": "a.pdf",
        "page_number": 3,
        "ingestion_status": "active",
    }
    assert points[1].payload == {
        "text": "world",
        "source": "b.pdf",
        "page_number": None,
        "ingestion_status": "active",
    }
    assert points[0].vector == {"dense": [0.1, 0.2], "bm25": sparse_embeddings[0]}
    assert points[0].id != points[1].id  # each point gets a unique id


def test_upsert_chunks_honors_an_explicit_ingestion_status() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake)

    repo.upsert_chunks(
        [DocumentChunk(text="pending", source="a.pdf")],
        [[0.1, 0.2]],
        [SparseVector(indices=[1], values=[1.0])],
        ingestion_status="quarantined",
    )

    points = cast(list, fake.upsert_calls[0]["points"])
    assert points[0].payload["ingestion_status"] == "quarantined"


def test_search_dense_filters_to_active_ingestion_status() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake)

    repo.search_dense([0.1], top_k=5)

    query_filter = cast(Filter, fake.query_points_calls[0]["query_filter"])
    condition = cast(FieldCondition, query_filter.must[0])  # type: ignore[index]
    assert condition.key == "ingestion_status"
    assert cast(MatchValue, condition.match).value == "active"


def test_set_ingestion_status_filters_on_source_and_sets_the_new_status() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake)
    fake.set_payload_calls.clear()  # drop the constructor's own backfill call

    repo.set_ingestion_status("a.pdf", "active")

    assert len(fake.set_payload_calls) == 1
    call = fake.set_payload_calls[0]
    assert call["payload"] == {"ingestion_status": "active"}
    points_filter = cast(Filter, call["points"])
    condition = cast(FieldCondition, points_filter.must[0])  # type: ignore[index]
    assert condition.key == "source"
    assert cast(MatchValue, condition.match).value == "a.pdf"


def test_upsert_chunks_raises_on_length_mismatch() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake)

    with pytest.raises(ValueError, match="shorter than"):
        repo.upsert_chunks([DocumentChunk(text="a", source="s")], [], [])


def test_search_dense_maps_points_and_uses_the_dense_vector() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    fake.query_response = _FakeQueryResponse(
        points=[
            _FakeScoredPoint(payload={"text": "hi", "source": "a.pdf", "page_number": 4}, score=0.9)
        ]
    )
    repo = _repo(fake)

    result = repo.search_dense([0.1, 0.2], top_k=5)

    assert result == [RetrievedChunk(text="hi", source="a.pdf", score=0.9, page_number=4)]
    call = fake.query_points_calls[0]
    assert call["using"] == "dense"
    assert call["query"] == [0.1, 0.2]
    assert call["limit"] == 5


def test_search_dense_handles_missing_payload() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    fake.query_response = _FakeQueryResponse(points=[_FakeScoredPoint(payload=None, score=0.5)])
    repo = _repo(fake)

    result = repo.search_dense([0.1], top_k=1)

    assert result == [RetrievedChunk(text="", source="", score=0.5, page_number=None)]


def test_search_sparse_maps_points_and_uses_the_bm25_vector() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    fake.query_response = _FakeQueryResponse(
        points=[_FakeScoredPoint(payload={"text": "hi", "source": "a.pdf"}, score=1.2)]
    )
    repo = _repo(fake)
    query_sparse = SparseVector(indices=[1], values=[1.0])

    result = repo.search_sparse(query_sparse, top_k=3)

    assert result == [RetrievedChunk(text="hi", source="a.pdf", score=1.2, page_number=None)]
    call = fake.query_points_calls[0]
    assert call["using"] == "bm25"
    assert call["query"] == query_sparse
    assert call["limit"] == 3


def test_search_hybrid_prefetches_both_vectors_and_fuses_via_configurable_rrf() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    fake.query_response = _FakeQueryResponse(
        points=[_FakeScoredPoint(payload={"text": "hi", "source": "a.pdf"}, score=0.03)]
    )
    repo = _repo(fake)
    query_sparse = SparseVector(indices=[1], values=[1.0])

    result = repo.search_hybrid(
        query_embedding=[0.1, 0.2],
        query_sparse=query_sparse,
        top_k=5,
        candidate_top_k=20,
        rrf_k=42,
    )

    assert result == [RetrievedChunk(text="hi", source="a.pdf", score=0.03, page_number=None)]
    call = fake.query_points_calls[0]
    prefetch = cast(list[Prefetch], call["prefetch"])
    assert len(prefetch) == 2
    dense_prefetch = next(p for p in prefetch if p.using == "dense")
    sparse_prefetch = next(p for p in prefetch if p.using == "bm25")
    assert dense_prefetch.query == [0.1, 0.2]
    assert dense_prefetch.limit == 20
    assert sparse_prefetch.query == query_sparse
    assert sparse_prefetch.limit == 20
    assert call["query"] == RrfQuery(rrf=Rrf(k=42))
    assert call["limit"] == 5


def test_scroll_all_chunks_returns_id_text_source_page_number_dicts() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    fake.scroll_response = (
        [_FakeRecord(id="abc", payload={"text": "hi", "source": "a.pdf", "page_number": 2})],
        None,
    )
    repo = _repo(fake)

    result = repo.scroll_all_chunks()

    assert result == [{"id": "abc", "text": "hi", "source": "a.pdf", "page_number": 2}]


def test_scroll_all_chunks_defaults_to_an_active_only_filter() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake)

    repo.scroll_all_chunks()

    scroll_filter = cast(Filter, fake.scroll_calls[0]["scroll_filter"])
    condition = cast(FieldCondition, scroll_filter.must[0])  # type: ignore[index]
    assert condition.key == "ingestion_status"
    assert cast(MatchValue, condition.match).value == "active"


def test_scroll_all_chunks_with_status_none_applies_no_filter() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake)

    repo.scroll_all_chunks(ingestion_status=None)

    assert fake.scroll_calls[0]["scroll_filter"] is None


def test_delete_by_source_deletes_from_the_configured_collection() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake, collection_name="docs")

    repo.delete_by_source("a.pdf")

    assert len(fake.delete_calls) == 1
    assert fake.delete_calls[0]["collection_name"] == "docs"


def test_delete_by_source_filters_on_the_source_payload_field() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake)

    repo.delete_by_source("a.pdf")

    points_filter = cast(Filter, fake.delete_calls[0]["points_selector"])
    condition = cast(FieldCondition, points_filter.must[0])  # type: ignore[index]
    assert condition.key == "source"
    assert cast(MatchValue, condition.match).value == "a.pdf"
