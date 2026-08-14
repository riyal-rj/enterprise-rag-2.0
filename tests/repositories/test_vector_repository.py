from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from qdrant_client import QdrantClient

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
    payload: dict[str, str] | None
    score: float


@dataclass
class _FakeQueryResponse:
    points: list[_FakeScoredPoint]


@dataclass
class _FakeRecord:
    id: str
    payload: dict[str, str] | None


class _FakeQdrantClient:
    def __init__(self, existing_collections: list[str] | None = None) -> None:
        self._existing = list(existing_collections or [])
        self.created_collections: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.query_response: _FakeQueryResponse = _FakeQueryResponse(points=[])
        self.scroll_response: tuple[list[_FakeRecord], None] = ([], None)

    def get_collections(self) -> _FakeCollectionsResponse:
        return _FakeCollectionsResponse(
            collections=[_FakeCollection(name=name) for name in self._existing]
        )

    def create_collection(self, **kwargs: object) -> None:
        self.created_collections.append(kwargs)
        self._existing.append(cast(str, kwargs["collection_name"]))

    def upsert(self, **kwargs: object) -> None:
        self.upsert_calls.append(kwargs)

    def query_points(self, **kwargs: object) -> _FakeQueryResponse:
        return self.query_response

    def scroll(self, **kwargs: object) -> tuple[list[_FakeRecord], None]:
        return self.scroll_response


def _repo(
    fake: _FakeQdrantClient, collection_name: str = "docs"
) -> QdrantVectorRepository:
    return QdrantVectorRepository(client=cast(QdrantClient, fake), collection_name=collection_name)


def test_init_creates_collection_when_missing() -> None:
    fake = _FakeQdrantClient(existing_collections=[])

    _repo(fake)

    assert len(fake.created_collections) == 1
    assert fake.created_collections[0]["collection_name"] == "docs"


def test_init_does_not_recreate_existing_collection() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])

    _repo(fake)

    assert fake.created_collections == []


def test_upsert_chunks_sends_text_and_source_payload() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake)
    chunks = [
        DocumentChunk(text="hello", source="a.pdf"),
        DocumentChunk(text="world", source="b.pdf"),
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    repo.upsert_chunks(chunks, embeddings)

    assert len(fake.upsert_calls) == 1
    points = cast(list, fake.upsert_calls[0]["points"])
    assert len(points) == 2
    assert points[0].payload == {"text": "hello", "source": "a.pdf"}
    assert points[0].vector == [0.1, 0.2]
    assert points[0].id != points[1].id  # each point gets a unique id


def test_upsert_chunks_raises_on_length_mismatch() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake)

    with pytest.raises(ValueError, match="shorter than"):
        repo.upsert_chunks([DocumentChunk(text="a", source="s")], [])


def test_search_maps_points_to_retrieved_chunks() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    fake.query_response = _FakeQueryResponse(
        points=[_FakeScoredPoint(payload={"text": "hi", "source": "a.pdf"}, score=0.9)]
    )
    repo = _repo(fake)

    result = repo.search([0.1, 0.2], top_k=5)

    assert result == [RetrievedChunk(text="hi", source="a.pdf", score=0.9)]


def test_search_handles_missing_payload() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    fake.query_response = _FakeQueryResponse(points=[_FakeScoredPoint(payload=None, score=0.5)])
    repo = _repo(fake)

    result = repo.search([0.1], top_k=1)

    assert result == [RetrievedChunk(text="", source="", score=0.5)]


def test_scroll_all_chunks_returns_id_text_source_dicts() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    fake.scroll_response = (
        [_FakeRecord(id="abc", payload={"text": "hi", "source": "a.pdf"})],
        None,
    )
    repo = _repo(fake)

    result = repo.scroll_all_chunks()

    assert result == [{"id": "abc", "text": "hi", "source": "a.pdf"}]
