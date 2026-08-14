from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

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
        self._existing = list(existing_collections or [])
        self.created_collections: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []
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

    def delete(self, **kwargs: object) -> None:
        self.delete_calls.append(kwargs)


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


def test_upsert_chunks_sends_text_source_and_page_number_payload() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    repo = _repo(fake)
    chunks = [
        DocumentChunk(text="hello", source="a.pdf", page_number=3),
        DocumentChunk(text="world", source="b.pdf"),
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    repo.upsert_chunks(chunks, embeddings)

    assert len(fake.upsert_calls) == 1
    points = cast(list, fake.upsert_calls[0]["points"])
    assert len(points) == 2
    assert points[0].payload == {"text": "hello", "source": "a.pdf", "page_number": 3}
    assert points[1].payload == {"text": "world", "source": "b.pdf", "page_number": None}
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
        points=[
            _FakeScoredPoint(payload={"text": "hi", "source": "a.pdf", "page_number": 4}, score=0.9)
        ]
    )
    repo = _repo(fake)

    result = repo.search([0.1, 0.2], top_k=5)

    assert result == [RetrievedChunk(text="hi", source="a.pdf", score=0.9, page_number=4)]


def test_search_handles_missing_payload() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    fake.query_response = _FakeQueryResponse(points=[_FakeScoredPoint(payload=None, score=0.5)])
    repo = _repo(fake)

    result = repo.search([0.1], top_k=1)

    assert result == [RetrievedChunk(text="", source="", score=0.5, page_number=None)]


def test_scroll_all_chunks_returns_id_text_source_page_number_dicts() -> None:
    fake = _FakeQdrantClient(existing_collections=["docs"])
    fake.scroll_response = (
        [_FakeRecord(id="abc", payload={"text": "hi", "source": "a.pdf", "page_number": 2})],
        None,
    )
    repo = _repo(fake)

    result = repo.scroll_all_chunks()

    assert result == [{"id": "abc", "text": "hi", "source": "a.pdf", "page_number": 2}]


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
