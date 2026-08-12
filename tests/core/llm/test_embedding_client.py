from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from openai import OpenAI

from app.core.config.cache import CacheSettings
from app.core.llm import embedding_client
from app.core.llm.embedding_client import OpenAIEmbeddingClient
from app.services.query_cache_service import QueryCacheService


class _InMemoryCacheBackend:
    """Minimal in-memory :class:`~app.services.query_cache_service.CacheBackend`."""

    def __init__(self, *, fail_get: bool = False, fail_set: bool = False) -> None:
        self._store: dict[str, str] = {}
        self._fail_get = fail_get
        self._fail_set = fail_set

    def get(self, key: str) -> str | None:
        if self._fail_get:
            raise ConnectionError("cache unavailable")
        return self._store.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        if self._fail_set:
            raise ConnectionError("cache unavailable")
        self._store[key] = value

    def delete_prefix(self, prefix: str) -> int:
        keys = [k for k in self._store if k.startswith(prefix)]
        for k in keys:
            del self._store[k]
        return len(keys)


@dataclass
class _FakeEmbeddingItem:
    embedding: list[float]
    index: int


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingItem]


class _FakeEmbeddings:
    def __init__(self, queue: list[_FakeEmbeddingResponse | Exception]) -> None:
        self._queue = list(queue)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeEmbeddingResponse:
        self.calls.append(kwargs)
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeOpenAI:
    def __init__(self, queue: list[_FakeEmbeddingResponse | Exception]) -> None:
        self.embeddings = _FakeEmbeddings(queue)


def _client(
    *queue: _FakeEmbeddingResponse | Exception, backend: _InMemoryCacheBackend | None = None
) -> tuple[OpenAIEmbeddingClient, _FakeOpenAI, QueryCacheService]:
    fake = _FakeOpenAI(list(queue))
    cache = QueryCacheService(backend or _InMemoryCacheBackend(), CacheSettings())
    return (
        OpenAIEmbeddingClient(
            client=cast(OpenAI, fake), cache=cache, default_model="text-embedding-3-small"
        ),
        fake,
        cache,
    )


def test_embed_texts_empty_list_returns_empty_without_calling_api() -> None:
    llm, fake, _ = _client()

    assert llm.embed_texts([]) == []
    assert fake.embeddings.calls == []


def test_embed_texts_calls_api_then_hits_cache_on_second_call() -> None:
    llm, fake, _ = _client(
        _FakeEmbeddingResponse(data=[_FakeEmbeddingItem(embedding=[0.1, 0.2], index=0)])
    )

    first = llm.embed_texts(["hello"])
    second = llm.embed_texts(["hello"])

    assert first == [[0.1, 0.2]]
    assert second == [[0.1, 0.2]]
    assert len(fake.embeddings.calls) == 1  # second call was a cache hit


def test_embed_texts_deduplicates_repeated_text_within_one_call() -> None:
    llm, fake, _ = _client(
        _FakeEmbeddingResponse(
            data=[
                _FakeEmbeddingItem(embedding=[1.0], index=0),  # "a"
                _FakeEmbeddingItem(embedding=[2.0], index=1),  # "b"
            ]
        )
    )

    result = llm.embed_texts(["a", "a", "b"])

    assert result == [[1.0], [1.0], [2.0]]
    assert fake.embeddings.calls[0]["input"] == ["a", "b"]  # "a" sent once, not twice


def test_embed_texts_uses_response_index_not_array_position() -> None:
    """Response items out of order must still map back to the right text."""
    llm, _, _ = _client(
        _FakeEmbeddingResponse(
            data=[
                _FakeEmbeddingItem(embedding=[9.0], index=1),  # "second" first in the array
                _FakeEmbeddingItem(embedding=[1.0], index=0),  # "first" second in the array
            ]
        )
    )

    result = llm.embed_texts(["first", "second"])

    assert result == [[1.0], [9.0]]


def test_embed_texts_cache_key_includes_model() -> None:
    llm, fake, _ = _client(
        _FakeEmbeddingResponse(data=[_FakeEmbeddingItem(embedding=[1.0], index=0)]),
        _FakeEmbeddingResponse(data=[_FakeEmbeddingItem(embedding=[2.0], index=0)]),
    )

    llm.embed_texts(["hello"], model="model-a")
    llm.embed_texts(["hello"], model="model-b")

    assert len(fake.embeddings.calls) == 2  # different models -> both cache misses


def test_cache_read_failure_falls_back_to_miss_without_crashing() -> None:
    llm, fake, _ = _client(
        _FakeEmbeddingResponse(data=[_FakeEmbeddingItem(embedding=[1.0], index=0)]),
        backend=_InMemoryCacheBackend(fail_get=True),
    )

    result = llm.embed_texts(["hello"])

    assert result == [[1.0]]
    assert len(fake.embeddings.calls) == 1


def test_cache_write_failure_does_not_crash() -> None:
    llm, _, _ = _client(
        _FakeEmbeddingResponse(data=[_FakeEmbeddingItem(embedding=[1.0], index=0)]),
        backend=_InMemoryCacheBackend(fail_set=True),
    )

    result = llm.embed_texts(["hello"])

    assert result == [[1.0]]


def test_embed_texts_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_client, "_RETRY_BACKOFF_SECONDS", 0.001)
    llm, fake, _ = _client(
        RuntimeError("transient"),
        _FakeEmbeddingResponse(data=[_FakeEmbeddingItem(embedding=[1.0], index=0)]),
    )

    result = llm.embed_texts(["hello"])

    assert result == [[1.0]]
    assert len(fake.embeddings.calls) == 2


def test_embed_texts_raises_after_exhausting_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_client, "_RETRY_BACKOFF_SECONDS", 0.001)
    llm, fake, _ = _client(
        RuntimeError("fail 1"), RuntimeError("fail 2"), RuntimeError("permanent failure")
    )

    with pytest.raises(RuntimeError, match="permanent failure"):
        llm.embed_texts(["hello"])

    assert len(fake.embeddings.calls) == 3
