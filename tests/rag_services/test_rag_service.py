from __future__ import annotations

from typing import cast

from app.core.config.cache import CacheSettings
from app.core.ingestion.document_processor import DocumentChunk
from app.core.llm.chat_client import LLMClient, LLMResponse, TokenUsage
from app.core.llm.embedding_client import EmbeddingClient
from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.rag_service import RAGService
from app.rag_services.retrieval_strategy import DenseRetrievalStrategy
from app.repositories.vector_repository import VectorRepository
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


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2] for _ in texts]


class _FakeVectorRepository:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.search_calls: list[dict[str, object]] = []

    def upsert_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        raise NotImplementedError

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        self.search_calls.append({"query_embedding": query_embedding, "top_k": top_k})
        return self._results[:top_k]

    def scroll_all_chunks(self, limit: int = 10_000) -> list[dict[str, str]]:
        raise NotImplementedError


class _FakeLLMClient:
    def __init__(self, answer: str = "the answer") -> None:
        self._answer = answer
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append({"system_prompt": system_prompt, "user_message": user_message})
        return LLMResponse(text=self._answer, usage=TokenUsage())

    def generate_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise NotImplementedError


def _service(
    results: list[RetrievedChunk] | None = None,
    answer: str = "the answer",
    backend: _InMemoryCacheBackend | None = None,
) -> tuple[RAGService, _FakeEmbeddingClient, _FakeVectorRepository, _FakeLLMClient]:
    embedding_client = _FakeEmbeddingClient()
    vector_repository = _FakeVectorRepository(results or [])
    llm_client = _FakeLLMClient(answer)
    cache = QueryCacheService(backend or _InMemoryCacheBackend(), CacheSettings())
    retrieval_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository)
    )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, embedding_client),
        retrieval_strategy=retrieval_strategy,
        llm_client=cast(LLMClient, llm_client),
        cache=cache,
    )
    return service, embedding_client, vector_repository, llm_client


def test_answer_returns_llm_text_and_sorted_unique_sources() -> None:
    chunks = [
        RetrievedChunk(text="a", source="b.pdf", score=0.9),
        RetrievedChunk(text="c", source="a.pdf", score=0.8),
        RetrievedChunk(text="d", source="a.pdf", score=0.7),
    ]
    service, _, _, _ = _service(results=chunks, answer="hello")

    response = service.answer("what is the policy?")

    assert response.answer == "hello"
    assert response.sources == ["a.pdf", "b.pdf"]
    assert response.cache_hit is False


def test_answer_confidence_is_computed_not_a_fixed_constant() -> None:
    strong_chunks = [
        RetrievedChunk(text="refunds within 30 days of purchase", source="a.pdf", score=0.9),
        RetrievedChunk(text="30 day refund window applies", source="b.pdf", score=0.88),
    ]
    weak_chunks = [RetrievedChunk(text="unrelated boilerplate", source="z.pdf", score=0.2)]

    strong_service, *_ = _service(
        results=strong_chunks, answer="Refunds are available within 30 days [a.pdf][b.pdf]."
    )
    weak_service, *_ = _service(results=weak_chunks, answer="I don't know.")

    strong_response = strong_service.answer("refund policy?")
    weak_response = weak_service.answer("refund policy?")

    assert strong_response.confidence != 0.7
    assert weak_response.confidence < strong_response.confidence


def test_answer_embeds_question_and_searches_with_top_k() -> None:
    service, embedding_client, vector_repository, _ = _service()

    service.answer("what is the policy?", top_k=3)

    assert embedding_client.calls == [["what is the policy?"]]
    assert vector_repository.search_calls[0]["top_k"] == 3


def test_answer_builds_context_with_source_citations() -> None:
    chunks = [RetrievedChunk(text="refunds within 30 days", source="policy.pdf", score=0.9)]
    service, _, _, llm_client = _service(results=chunks)

    service.answer("what is the refund policy?")

    user_message = cast(str, llm_client.calls[0]["user_message"])
    assert "[policy.pdf]" in user_message
    assert "refunds within 30 days" in user_message
    assert "what is the refund policy?" in user_message


def test_answer_with_no_chunks_still_generates_an_answer() -> None:
    service, _, _, llm_client = _service(results=[])

    response = service.answer("anything?")

    assert response.sources == []
    assert response.metadata.retrieved_chunks == []
    assert "No relevant context" in cast(str, llm_client.calls[0]["user_message"])


def test_metadata_retrieved_chunks_mirror_search_results() -> None:
    chunks = [RetrievedChunk(text="hi", source="a.pdf", score=0.5, page_number=4)]
    service, _, _, _ = _service(results=chunks)

    response = service.answer("q")

    assert response.metadata.route == "rag"
    assert len(response.metadata.retrieved_chunks) == 1
    assert response.metadata.retrieved_chunks[0].text == "hi"
    assert response.metadata.retrieved_chunks[0].source == "a.pdf"
    assert response.metadata.retrieved_chunks[0].score == 0.5
    assert response.metadata.retrieved_chunks[0].page_number == 4


def test_second_call_with_same_args_is_a_cache_hit_and_skips_llm() -> None:
    chunks = [RetrievedChunk(text="hi", source="a.pdf", score=0.5)]
    service, embedding_client, _, llm_client = _service(results=chunks)

    first = service.answer("what is the policy?", top_k=5)
    second = service.answer("what is the policy?", top_k=5)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.answer == first.answer
    assert len(llm_client.calls) == 1  # second call was a cache hit
    assert len(embedding_client.calls) == 1


def test_different_top_k_is_not_a_cache_hit() -> None:
    service, _, _, llm_client = _service()

    service.answer("q", top_k=5)
    service.answer("q", top_k=10)

    assert len(llm_client.calls) == 2


def test_cache_read_failure_falls_back_to_computing_fresh() -> None:
    service, _, _, llm_client = _service(backend=_InMemoryCacheBackend(fail_get=True))

    response = service.answer("q")

    assert response.answer == "the answer"
    assert len(llm_client.calls) == 1


def test_cache_write_failure_does_not_crash() -> None:
    service, _, _, _ = _service(backend=_InMemoryCacheBackend(fail_set=True))

    response = service.answer("q")

    assert response.answer == "the answer"
