from __future__ import annotations

from typing import cast

import pytest

from app.eval import invokers
from app.eval.invokers import RetrievedChunk, ServiceInvoker, SkippedIntent
from app.eval.profiles import PROFILES
from app.eval.schemas import Intent
from app.rag_services.rag_service import RAGService
from app.schemas.chat import ChatResponse, ResponseMetadata, RetrievedChunkPreview


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
        self, question: str, top_k: int = 5, retrieval_mode: str | None = None
    ) -> ChatResponse:
        self.calls.append({"question": question, "top_k": top_k, "retrieval_mode": retrieval_mode})
        return self._response


def _response(answer: str = "the answer") -> ChatResponse:
    return ChatResponse(
        answer=answer,
        sources=["a.pdf"],
        confidence=0.9,
        metadata=ResponseMetadata(
            route="rag",
            retrieval_mode="dense",
            retrieved_chunks=[
                RetrievedChunkPreview(text="hi", source="a.pdf", score=0.9, page_number=None)
            ],
        ),
        conversation_id=1,
    )


def test_service_invoker_skips_unsupported_intents_before_touching_the_pipeline() -> None:
    """SQL/hybrid need human-in-the-loop approval — not runnable headlessly.
    No rag_service is injected, so this also proves the check runs before
    the (expensive) real pipeline would ever be built."""
    invoker = ServiceInvoker()

    with pytest.raises(SkippedIntent, match="not supported in service mode"):
        invoker.invoke("question", PROFILES["naive"], Intent.SQL)


def test_service_invoker_skips_web_fallback_without_tavily_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(invokers, "get_settings", lambda: _FakeSettings(tavily_api_key=""))
    invoker = ServiceInvoker()

    with pytest.raises(SkippedIntent, match="tavily_unset"):
        invoker.invoke("question", PROFILES["naive"], Intent.WEB_FALLBACK)


def test_service_invoker_calls_the_real_rag_service_for_a_supported_profile() -> None:
    fake_rag_service = _FakeRAGService(_response("hello"))
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    response, chunks = invoker.invoke("what is the policy?", PROFILES["naive"], Intent.RAG)

    assert response.answer == "hello"
    assert response.sources == ["a.pdf"]
    assert chunks == [RetrievedChunk(text="hi", source="a.pdf")]
    assert fake_rag_service.calls == [
        {"question": "what is the policy?", "top_k": PROFILES["naive"].top_k, "retrieval_mode": "dense"}
    ]


def test_service_invoker_passes_search_mode_as_retrieval_mode_override() -> None:
    fake_rag_service = _FakeRAGService(_response())
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    invoker.invoke("q", PROFILES["hybrid"], Intent.RAG)

    assert fake_rag_service.calls[0]["retrieval_mode"] == "hybrid"


def test_service_invoker_calls_pipeline_for_web_fallback_when_tavily_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passes both guards, then reaches the real pipeline like RAG does."""
    monkeypatch.setattr(invokers, "get_settings", lambda: _FakeSettings(tavily_api_key="tvly-test"))
    fake_rag_service = _FakeRAGService(_response())
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    response, _chunks = invoker.invoke("question", PROFILES["naive"], Intent.WEB_FALLBACK)

    assert response.answer == "the answer"


@pytest.mark.parametrize(
    "profile_name", ["hybrid+rerank", "hybrid+rerank+hyde", "hybrid+rerank+crag", "all"]
)
def test_service_invoker_skips_profiles_requesting_unimplemented_features(profile_name: str) -> None:
    """Reranking/HyDE/CRAG/self-reflective don't exist in the pipeline yet -
    silently ignoring the flag would produce misleading pass/fail results,
    so these skip cleanly instead, even with a working rag_service wired up."""
    fake_rag_service = _FakeRAGService(_response())
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_rag_service))

    with pytest.raises(SkippedIntent, match="aren't implemented"):
        invoker.invoke("question", PROFILES[profile_name], Intent.RAG)

    assert fake_rag_service.calls == []


def test_service_invoker_builds_a_real_rag_service_lazily_when_none_is_injected() -> None:
    """Guard-check-only paths (unsupported intent, missing Tavily key) must
    not require the full DI chain (Qdrant/OpenAI/FastEmbed) to be
    importable/configured - only reaching `_call_pipeline` should trigger
    building the real RAGService."""
    invoker = ServiceInvoker()

    with pytest.raises(SkippedIntent, match="not supported in service mode"):
        invoker.invoke("question", PROFILES["naive"], Intent.SQL)

    assert "_rag_service" not in invoker.__dict__  # cached_property never touched
