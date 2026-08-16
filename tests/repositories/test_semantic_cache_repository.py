from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.repositories.semantic_cache_repository import QdrantSemanticQueryCache


@dataclass
class _FakeScoredPoint:
    payload: dict[str, object] | None
    score: float


@dataclass
class _FakeQueryResponse:
    points: list[_FakeScoredPoint]


@dataclass
class _FakeCountResult:
    count: int


class _FakeQdrantClient:
    def __init__(self, existing_collections: list[str] | None = None, point_count: int = 0) -> None:
        self._existing = set(existing_collections or [])
        self._point_count = point_count
        self.created_collections: list[dict[str, object]] = []
        self.created_payload_indexes: list[dict[str, object]] = []
        self.deleted_collections: list[str] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.query_points_calls: list[dict[str, object]] = []
        self.query_response: _FakeQueryResponse = _FakeQueryResponse(points=[])

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._existing

    def create_collection(self, **kwargs: object) -> None:
        self.created_collections.append(kwargs)
        self._existing.add(str(kwargs["collection_name"]))

    def create_payload_index(self, **kwargs: object) -> None:
        self.created_payload_indexes.append(kwargs)

    def upsert(self, **kwargs: object) -> None:
        self.upsert_calls.append(kwargs)

    def query_points(self, **kwargs: object) -> _FakeQueryResponse:
        self.query_points_calls.append(kwargs)
        return self.query_response

    def count(self, collection_name: str) -> _FakeCountResult:
        return _FakeCountResult(count=self._point_count)

    def delete_collection(self, collection_name: str) -> None:
        self.deleted_collections.append(collection_name)
        self._existing.discard(collection_name)
        self._point_count = 0


def _cache(client: _FakeQdrantClient, *, threshold: float = 0.95) -> QdrantSemanticQueryCache:
    return QdrantSemanticQueryCache(
        client=client,  # type: ignore[arg-type]
        collection_name="rag_query_cache",
        similarity_threshold=threshold,
    )


def test_construction_creates_the_collection_and_payload_indexes_if_missing() -> None:
    client = _FakeQdrantClient()

    _cache(client)

    assert len(client.created_collections) == 1
    assert client.created_collections[0]["collection_name"] == "rag_query_cache"
    indexed_fields = {call["field_name"] for call in client.created_payload_indexes}
    assert indexed_fields == {"cache_namespace", "top_k"}


def test_construction_does_not_recreate_an_existing_collection() -> None:
    client = _FakeQdrantClient(existing_collections=["rag_query_cache"])

    _cache(client)

    assert client.created_collections == []
    assert client.created_payload_indexes == []


def test_constructor_rejects_threshold_outside_zero_to_one() -> None:
    client = _FakeQdrantClient()

    with pytest.raises(ValueError):
        _cache(client, threshold=1.5)

    with pytest.raises(ValueError):
        _cache(client, threshold=-0.1)


def test_find_candidates_returns_empty_list_when_no_points_match() -> None:
    client = _FakeQdrantClient()
    client.query_response = _FakeQueryResponse(points=[])
    cache = _cache(client)

    result = cache.find_candidates(query_embedding=[0.1, 0.2], cache_namespace="dense:v1", top_k=5)

    assert result == []


def test_find_candidates_excludes_points_below_threshold() -> None:
    client = _FakeQdrantClient()
    client.query_response = _FakeQueryResponse(
        points=[_FakeScoredPoint(payload={"cache_key": "abc"}, score=0.80)]
    )
    cache = _cache(client, threshold=0.95)

    result = cache.find_candidates(query_embedding=[0.1, 0.2], cache_namespace="dense:v1", top_k=5)

    assert result == []


def test_find_candidates_returns_the_cache_key_when_score_clears_threshold() -> None:
    client = _FakeQdrantClient()
    client.query_response = _FakeQueryResponse(
        points=[_FakeScoredPoint(payload={"cache_key": "abc123"}, score=0.97)]
    )
    cache = _cache(client, threshold=0.95)

    result = cache.find_candidates(query_embedding=[0.1, 0.2], cache_namespace="dense:v1", top_k=5)

    assert result == ["abc123"]


def test_find_candidates_score_exactly_at_threshold_counts_as_a_match() -> None:
    client = _FakeQdrantClient()
    client.query_response = _FakeQueryResponse(
        points=[_FakeScoredPoint(payload={"cache_key": "abc"}, score=0.95)]
    )
    cache = _cache(client, threshold=0.95)

    result = cache.find_candidates(query_embedding=[0.1, 0.2], cache_namespace="dense:v1", top_k=5)

    assert result == ["abc"]


def test_find_candidates_returns_multiple_matches_nearest_first_and_stops_at_threshold() -> None:
    """Regression: only checking the single nearest point means a stale
    pointer masks a perfectly good second-nearest match. find_candidates
    must surface every match above threshold, in score order, so the
    caller (RAGService) can fall through to the next one."""
    client = _FakeQdrantClient()
    client.query_response = _FakeQueryResponse(
        points=[
            _FakeScoredPoint(payload={"cache_key": "nearest"}, score=0.99),
            _FakeScoredPoint(payload={"cache_key": "second"}, score=0.96),
            _FakeScoredPoint(payload={"cache_key": "below-threshold"}, score=0.80),
        ]
    )
    cache = _cache(client, threshold=0.95)

    result = cache.find_candidates(query_embedding=[0.1], cache_namespace="dense:v1", top_k=5)

    assert result == ["nearest", "second"]


def test_find_candidates_filters_by_cache_namespace_and_top_k() -> None:
    client = _FakeQdrantClient()
    cache = _cache(client)

    cache.find_candidates(query_embedding=[0.1], cache_namespace="hybrid:v2", top_k=8)

    call = client.query_points_calls[0]
    query_filter = call["query_filter"]
    conditions = {c.key: c.match.value for c in query_filter.must}
    assert conditions == {"cache_namespace": "hybrid:v2", "top_k": 8}


def test_record_upserts_a_point_with_the_embedding_and_pointer_payload() -> None:
    client = _FakeQdrantClient()
    cache = _cache(client)

    cache.record(
        query_embedding=[0.1, 0.2, 0.3],
        cache_namespace="dense:v1",
        top_k=5,
        cache_key="deadbeef",
    )

    assert len(client.upsert_calls) == 1
    points = client.upsert_calls[0]["points"]
    assert len(points) == 1
    point = points[0]
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload == {"cache_namespace": "dense:v1", "top_k": 5, "cache_key": "deadbeef"}


def test_clear_deletes_and_recreates_the_collection_returning_prior_count() -> None:
    client = _FakeQdrantClient(existing_collections=["rag_query_cache"], point_count=7)
    cache = _cache(client)

    removed = cache.clear()

    assert removed == 7
    assert client.deleted_collections == ["rag_query_cache"]
    # Recreated afterwards, ready for new points again.
    assert client.collection_exists("rag_query_cache")
    assert len(client.created_collections) == 1
