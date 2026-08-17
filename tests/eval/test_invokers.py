from __future__ import annotations

from typing import cast

import pytest

from app.core.exceptions import EvaluationPipelineError
from app.core.llm.chat_client import LLMResponse, TokenUsage
from app.eval import invokers
from app.eval.invokers import RetrievedChunk, ServiceInvoker, SkippedIntent
from app.eval.profiles import PROFILES
from app.eval.schemas import Intent
from app.rag_services.rag_service import RAGService
from app.rag_services.reranker import NoOpReranker
from app.rag_services.retrieval_strategy import DenseRetrievalStrategy
from app.repositories.vector_repository import VectorRepository
from app.schemas.chat import (
    ChatResponse,
    HyDEMetadata,
    RerankingMetadata,
    ResponseMetadata,
    RetrievedChunkPreview,
)


class _FakeSecretStr:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class _FakeExternalApis:
    def __init__(self, tavily_api_key: str) -> None:
        self.tavily_api_key = _FakeSecretStr(tavily_api_key)


class _FakeSettings:
    def __init__(self, tavily_api_key: str) -> None:
        self.external_apis = _FakeExternalApis(tavily_api_key)


class _FakeRAGService:
    def __init__(self, response: ChatResponse) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def answer(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str | None = None,
        *,
        reranking_enabled: bool | None = None,
        hyde_enabled: bool | None = None,
    ) -> ChatResponse:
        self.calls.append(
            {
                "question": question,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
                "reranking_enabled": reranking_enabled,
                "hyde_enabled": hyde_enabled,
            }
        )
        return self._response


def _response(answer: str = "the answer", *, hyde: HyDEMetadata | None = None) -> ChatResponse:
    return ChatResponse(
        answer=answer,
        sources=["a.pdf"],
        confidence=0.9,
        metadata=ResponseMetadata(
            route="rag",
            retrieval_mode="dense",
            hyde=hyde or HyDEMetadata(enabled=False, backend="none"),
            reranking=RerankingMetadata(enabled=False, backend="none"),
            retrieved_chunks=[
                RetrievedChunkPreview(text="hi", source="a.pdf", score=0.9, page_number=None)
            ],
        ),
        conversation_id=1,
    )


def _hyde_applied_response(answer: str = "the answer") -> ChatResponse:
    """A response shaped like a real, successfully-applied HyDE run - used
    by tests that pass ``enable_hyde=True`` and must clear
    ServiceInvoker._call_pipeline's EvaluationPipelineError check (see
    app.eval.invokers)."""
    return _response(
        answer,
        hyde=HyDEMetadata(
            enabled=True, applied=True, fallback=False, backend="hyde", hypothesis_count=2
        ),
    )


def test_service_invoker_skips_unsupported_intents_before_touching_the_pipeline() -> None:
    """SQL/hybrid need human-in-the-loop approval — not runnable headlessly.
    No rag_service is injected, so this also proves the check runs before
    the (expensive) real pipeline would ever be built."""
    invoker = ServiceInvoker()

    with pytest.raises(SkippedIntent, match="not supported in service mode"):
        invoker.invoke("question", PROFILES["naive"], Intent.SQL)


def test_service_invoker_skips_web_fallback_regardless_of_tavily_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No answer-generation pipeline exists for web_fallback yet - it must
    skip honestly rather than silently re-running RAG against the policy
    corpus and being scored as if it tested something else."""
    monkeypatch.setattr(invokers, "get_settings", lambda: _FakeSettings(tavily_api_key="tvly-test"))
    invoker = ServiceInvoker()

    with pytest.raises(SkippedIntent, match="not supported in service mode"):
        invoker.invoke("question", PROFILES["naive"], Intent.WEB_FALLBACK)


def test_service_invoker_calls_the_real_rag_service_for_a_supported_profile() -> None:
    fake_rag_service = _FakeRAGService(_response("hello"))
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    response, chunks = invoker.invoke("what is the policy?", PROFILES["naive"], Intent.RAG)

    assert response.answer == "hello"
    assert response.sources == ["a.pdf"]
    assert chunks == [RetrievedChunk(text="hi", source="a.pdf")]
    assert fake_rag_service.calls == [
        {
            "question": "what is the policy?",
            "top_k": PROFILES["naive"].top_k,
            "retrieval_mode": "dense",
            "reranking_enabled": False,
            "hyde_enabled": False,
        }
    ]


def test_service_invoker_passes_search_mode_as_retrieval_mode_override() -> None:
    fake_rag_service = _FakeRAGService(_response())
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    invoker.invoke("q", PROFILES["hybrid"], Intent.RAG)

    assert fake_rag_service.calls[0]["retrieval_mode"] == "hybrid"


def test_service_invoker_passes_enable_rerank_as_a_per_call_override() -> None:
    """hybrid+rerank must actually exercise reranking through the real
    RAGService.answer() override, not silently no-op - this is what makes
    a baseline-vs-reranker RAGAS comparison possible."""
    fake_rag_service = _FakeRAGService(_response())
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    invoker.invoke("q", PROFILES["hybrid+rerank"], Intent.RAG)

    assert fake_rag_service.calls[0]["reranking_enabled"] is True
    assert fake_rag_service.calls[0]["retrieval_mode"] == "hybrid"


def test_service_invoker_passes_enable_hyde_as_a_per_call_override() -> None:
    """hybrid+rerank+hyde must actually exercise HyDE through the real
    RAGService.answer() override, not silently no-op - this is what makes
    a baseline-vs-HyDE RAGAS comparison possible. HyDE is no longer in the
    unimplemented-features skip list - see
    test_service_invoker_skips_profiles_requesting_unimplemented_features."""
    fake_rag_service = _FakeRAGService(_hyde_applied_response())
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    invoker.invoke("q", PROFILES["hybrid+rerank+hyde"], Intent.RAG)

    assert fake_rag_service.calls[0]["hyde_enabled"] is True
    assert fake_rag_service.calls[0]["reranking_enabled"] is True
    assert fake_rag_service.calls[0]["retrieval_mode"] == "hybrid"


def test_hyde_fallback_raises_evaluation_pipeline_error_instead_of_scoring_baseline() -> None:
    """A HyDE case that silently fell back (LLM/embedding/fusion failure)
    must fail loudly, not be scored as if HyDE ran - see
    app.core.exceptions.EvaluationPipelineError."""
    response = _response(
        hyde=HyDEMetadata(enabled=True, applied=False, fallback=True, backend="hyde")
    )
    fake_rag_service = _FakeRAGService(response)
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    with pytest.raises(EvaluationPipelineError, match="hybrid\\+rerank\\+hyde"):
        invoker.invoke("q", PROFILES["hybrid+rerank+hyde"], Intent.RAG)


def test_hyde_bypass_reason_raises_evaluation_pipeline_error() -> None:
    """applied=False with a bypass_reason (even without fallback=True) is
    still "HyDE did not run" - the eval harness's StaticPlannedQueryTransformer
    never bypasses for rollout/emergency reasons in practice, but the check
    itself doesn't assume that; it rejects any non-treatment outcome."""
    response = _response(
        hyde=HyDEMetadata(enabled=True, applied=False, fallback=False, bypass_reason="rollout")
    )
    fake_rag_service = _FakeRAGService(response)
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    with pytest.raises(EvaluationPipelineError):
        invoker.invoke("q", PROFILES["hybrid+rerank+hyde"], Intent.RAG)


def test_non_hyde_profile_ignores_hyde_metadata_entirely() -> None:
    """A profile with enable_hyde=False must never trigger the HyDE
    integrity check, regardless of what the response's hyde metadata says -
    that field is meaningless when HyDE wasn't even requested."""
    response = _response(
        hyde=HyDEMetadata(enabled=False, applied=False, fallback=True, bypass_reason="disabled")
    )
    fake_rag_service = _FakeRAGService(response)
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    result, _ = invoker.invoke("q", PROFILES["naive"], Intent.RAG)

    assert result.answer == response.answer


@pytest.mark.parametrize("profile_name", ["hybrid+rerank+crag", "all"])
def test_service_invoker_skips_profiles_requesting_unimplemented_features(
    profile_name: str,
) -> None:
    """CRAG/self-reflective don't exist in the pipeline yet - silently
    ignoring the flag would produce misleading pass/fail results, so these
    skip cleanly instead, even with a working rag_service wired up.
    HyDE and reranking alone are no longer in this list - see
    test_service_invoker_passes_enable_hyde_as_a_per_call_override and
    test_service_invoker_passes_enable_rerank_as_a_per_call_override."""
    fake_rag_service = _FakeRAGService(_response())
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    with pytest.raises(SkippedIntent, match="aren't implemented"):
        invoker.invoke("question", PROFILES[profile_name], Intent.RAG)

    assert fake_rag_service.calls == []


def test_service_invoker_builds_a_real_rag_service_lazily_when_none_is_injected() -> None:
    """Guard-check-only paths (unsupported intent) must not require the
    full DI chain (Qdrant/OpenAI/FastEmbed) to be importable/configured -
    only reaching `_call_pipeline` should trigger building the real
    RAGService."""
    invoker = ServiceInvoker()

    with pytest.raises(SkippedIntent, match="not supported in service mode"):
        invoker.invoke("question", PROFILES["naive"], Intent.SQL)

    assert "_rag_service" not in invoker.__dict__  # cached_property never touched


class _FakeEmbeddingClient:
    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        return [[0.1] for _ in texts]


class _FakeVectorRepository:
    def upsert_chunks(self, *args, **kwargs):
        raise NotImplementedError

    def search_dense(self, query_embedding, top_k: int = 5):
        return []

    def search_sparse(self, *args, **kwargs):
        raise NotImplementedError

    def search_hybrid(self, *args, **kwargs):
        raise NotImplementedError

    def scroll_all_chunks(self, limit: int = 10_000):
        raise NotImplementedError

    def delete_by_source(self, source: str) -> None:
        raise NotImplementedError


class _CountingLLMClient:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(text="answer", usage=TokenUsage())

    def generate_json(self, *args, **kwargs) -> LLMResponse:
        raise NotImplementedError


def test_service_invoker_bypasses_the_shared_production_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazily-built real RAGService must run the pipeline for real on
    every call, not serve a cached answer from the shared production
    cache - two identical questions must both reach the LLM client."""
    llm_client = _CountingLLMClient()
    dense_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, _FakeVectorRepository())
    )
    monkeypatch.setattr("app.api.deps.get_embedding_client", lambda: _FakeEmbeddingClient())
    monkeypatch.setattr("app.api.deps.get_llm_client", lambda: llm_client)
    monkeypatch.setattr("app.api.deps.get_retrieval_strategies", lambda: {"dense": dense_strategy})
    monkeypatch.setattr("app.api.deps.get_default_retrieval_mode", lambda: "dense")
    monkeypatch.setattr("app.api.deps.get_reranker", lambda: NoOpReranker())
    invoker = ServiceInvoker()

    invoker.invoke("same question", PROFILES["naive"], Intent.RAG)
    invoker.invoke("same question", PROFILES["naive"], Intent.RAG)

    assert llm_client.call_count == 2
