"""Tests for StructuredIntentRouter - same local _FakeLLMClient convention
as tests/rag_services/test_crag_refiner.py etc."""

from __future__ import annotations

from app.core.llm.chat_client import StructuredLLMResponse, TokenUsage
from app.query_orchestration.intent_router import StructuredIntentRouter
from app.sql.models import QueryRoute


class _FakeLLMClient:
    def __init__(self, *, payload: dict[str, object]) -> None:
        self._payload = payload
        self.calls: list[str] = []

    def generate(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def generate_json(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def generate_structured(
        self,
        system_prompt: str,
        user_message: str,
        *,
        response_model: type,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 1_000,
        timeout_seconds: float = 30.0,
        max_attempts: int = 2,
    ) -> StructuredLLMResponse:
        del system_prompt, model, temperature, max_completion_tokens, timeout_seconds, max_attempts
        self.calls.append(user_message)
        value = response_model(**self._payload)
        return StructuredLLMResponse(value=value, usage=TokenUsage(total_tokens=7))


def _router(llm_client: object, min_sql_confidence: float = 0.90) -> StructuredIntentRouter:
    return StructuredIntentRouter(
        llm_client=llm_client,  # type: ignore[arg-type]
        model="gpt-4o-mini",
        min_sql_confidence=min_sql_confidence,
        prompt_version="sql-router-v1",
        timeout_seconds=8.0,
        max_completion_tokens=200,
    )


def test_confident_rag_route_is_passed_through() -> None:
    llm = _FakeLLMClient(
        payload={"route": "rag", "confidence": 0.95, "reason_code": "policy_question"}
    )

    decision = _router(llm).route("what is the refund policy?")

    assert decision.route is QueryRoute.RAG
    assert decision.confidence == 0.95


def test_confident_sql_route_is_passed_through() -> None:
    llm = _FakeLLMClient(
        payload={"route": "sql", "confidence": 0.96, "reason_code": "structured_data_question"}
    )

    decision = _router(llm).route("how many accounts opened last month?")

    assert decision.route is QueryRoute.SQL


def test_low_confidence_sql_route_is_downgraded_to_reject() -> None:
    llm = _FakeLLMClient(payload={"route": "sql", "confidence": 0.5, "reason_code": "unclear"})

    decision = _router(llm, min_sql_confidence=0.90).route("some ambiguous question")

    assert decision.route is QueryRoute.REJECT
    assert decision.reason_code == "uncertain_sql_intent"


def test_low_confidence_hybrid_route_is_also_downgraded_to_reject() -> None:
    llm = _FakeLLMClient(
        payload={"route": "hybrid_rag_sql", "confidence": 0.4, "reason_code": "unclear"}
    )

    decision = _router(llm, min_sql_confidence=0.90).route("some ambiguous question")

    assert decision.route is QueryRoute.REJECT


def test_low_confidence_rag_route_is_not_downgraded() -> None:
    """The confidence gate only applies to SQL/hybrid routes - a
    low-confidence RAG classification still routes to RAG, since RAG is the
    safe, no-side-effect default this app already serves unconditionally."""
    llm = _FakeLLMClient(payload={"route": "rag", "confidence": 0.3, "reason_code": "uncertain"})

    decision = _router(llm, min_sql_confidence=0.90).route("some question")

    assert decision.route is QueryRoute.RAG


def test_cache_namespace_reflects_model_prompt_and_threshold() -> None:
    llm = _FakeLLMClient(payload={"route": "rag", "confidence": 0.9, "reason_code": "x"})

    namespace = _router(llm, min_sql_confidence=0.85).cache_namespace

    assert "gpt-4o-mini" in namespace
    assert "0.850" in namespace
