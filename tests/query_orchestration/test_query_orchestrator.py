"""Tests for QueryOrchestrator's routing decisions - the sql_enabled=False
short-circuit is the single most important guarantee here: it's what keeps
/chat's behavior and cost unchanged for every deployment that hasn't opted
into SQL routing (the default)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

from app.core.exceptions import SQLGenerationFailedError
from app.models.auth_user import AuthenticatedUser
from app.query_orchestration.intent_router import IntentRouter
from app.query_orchestration.query_orchestrator import QueryOrchestrator
from app.rag_services.rag_runtime_config import RagRuntimeConfig, RagRuntimeConfigStore
from app.rag_services.rag_service import RAGService
from app.schemas.chat import (
    ChatResponse,
    CRAGMetadata,
    HyDEMetadata,
    RerankingMetadata,
    ResponseMetadata,
    SelfReflectionMetadata,
)
from app.sql.models import QueryRoute, RouteDecision, SQLPrincipal, SQLProposal, SQLProposalStatus
from app.sql.sql_service import SQLService


def _rag_response(answer: str = "rag answer") -> ChatResponse:
    return ChatResponse(
        answer=answer,
        sources=["a.pdf"],
        confidence=0.8,
        metadata=ResponseMetadata(
            route="rag",
            hyde=HyDEMetadata(enabled=False),
            reranking=RerankingMetadata(enabled=False, backend="none"),
            crag=CRAGMetadata(enabled=False),
            self_reflection=SelfReflectionMetadata(enabled=False),
            retrieved_chunks=[],
        ),
    )


class _FakeRAGService:
    def __init__(self, response: ChatResponse | None = None) -> None:
        self._response = response or _rag_response()
        self.calls = 0

    def answer(
        self, question: str, top_k: int = 5, retrieval_mode: str | None = None
    ) -> ChatResponse:
        del question, top_k, retrieval_mode
        self.calls += 1
        return self._response


class _FakeRouter:
    def __init__(self, decision: RouteDecision) -> None:
        self._decision = decision
        self.calls = 0

    @property
    def cache_namespace(self) -> str:
        return "fake-router"

    def route(self, question: str) -> RouteDecision:
        del question
        self.calls += 1
        return self._decision


class _FakeSQLService:
    def __init__(
        self, *, proposal: SQLProposal | None = None, error: Exception | None = None
    ) -> None:
        self._proposal = proposal
        self._error = error
        self.propose_calls = 0

    def propose(
        self, *, principal: SQLPrincipal, conversation_id: int, question: str
    ) -> SQLProposal:
        del principal, conversation_id, question
        self.propose_calls += 1
        if self._error is not None:
            raise self._error
        assert self._proposal is not None
        return self._proposal


def _config(*, sql_enabled: bool) -> RagRuntimeConfig:
    return RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=False,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
        sql_enabled=sql_enabled,
    )


def _proposal() -> SQLProposal:
    now = datetime.now(UTC)
    return SQLProposal(
        id=uuid4(),
        username="admin",
        conversation_id=1,
        question="how many accounts?",
        sql="SELECT a.id FROM approved_analytics.accounts a LIMIT 10",
        sql_fingerprint="f" * 64,
        referenced_tables=("approved_analytics.accounts",),
        assumptions=(),
        catalog_version=1,
        policy_version="sql-policy-v1",
        status=SQLProposalStatus.PROPOSED,
        expires_at=now + timedelta(minutes=5),
        created_at=now,
    )


def _orchestrator(
    *, sql_enabled: bool, decision: RouteDecision, sql_service: _FakeSQLService | None = None
) -> tuple[QueryOrchestrator, _FakeRAGService, _FakeRouter]:
    rag_service = _FakeRAGService()
    router = _FakeRouter(decision)
    orchestrator = QueryOrchestrator(
        rag_service=cast(RAGService, rag_service),
        router=cast(IntentRouter, router),
        sql_service=cast(SQLService, sql_service or _FakeSQLService()),
        config_store=RagRuntimeConfigStore(_config(sql_enabled=sql_enabled)),
    )
    return orchestrator, rag_service, router


def _admin() -> AuthenticatedUser:
    return AuthenticatedUser(username="admin", is_admin=True)


def _regular_user() -> AuthenticatedUser:
    return AuthenticatedUser(username="alice", is_admin=False)


def test_sql_disabled_short_circuits_to_rag_without_calling_router() -> None:
    orchestrator, rag_service, router = _orchestrator(
        sql_enabled=False,
        decision=RouteDecision(route=QueryRoute.SQL, confidence=0.99, reason_code="x"),
    )

    response = orchestrator.answer(
        principal=_regular_user(),
        conversation_id=1,
        question="how many accounts?",
        top_k=5,
        retrieval_mode=None,
    )

    assert response.answer == "rag answer"
    assert rag_service.calls == 1
    assert router.calls == 0  # the whole point: no LLM router call when the feature is off


def test_sql_enabled_rag_route_delegates_to_rag_service() -> None:
    orchestrator, rag_service, router = _orchestrator(
        sql_enabled=True,
        decision=RouteDecision(route=QueryRoute.RAG, confidence=0.95, reason_code="policy"),
    )

    response = orchestrator.answer(
        principal=_admin(),
        conversation_id=1,
        question="what is the policy?",
        top_k=5,
        retrieval_mode=None,
    )

    assert response.answer == "rag answer"
    assert router.calls == 1


def test_sql_enabled_admin_gets_approval_required_response() -> None:
    proposal = _proposal()
    sql_service = _FakeSQLService(proposal=proposal)
    orchestrator, _, _ = _orchestrator(
        sql_enabled=True,
        decision=RouteDecision(route=QueryRoute.SQL, confidence=0.97, reason_code="structured"),
        sql_service=sql_service,
    )

    response = orchestrator.answer(
        principal=_admin(),
        conversation_id=1,
        question="how many accounts?",
        top_k=5,
        retrieval_mode=None,
    )

    assert response.status == "approval_required"
    assert response.metadata.sql.enabled is True
    assert response.metadata.sql.proposal_id == str(proposal.id)
    assert sql_service.propose_calls == 1


def test_sql_enabled_non_admin_is_rejected_without_proposing() -> None:
    sql_service = _FakeSQLService(proposal=_proposal())
    orchestrator, _, _ = _orchestrator(
        sql_enabled=True,
        decision=RouteDecision(route=QueryRoute.SQL, confidence=0.97, reason_code="structured"),
        sql_service=sql_service,
    )

    response = orchestrator.answer(
        principal=_regular_user(),
        conversation_id=1,
        question="how many accounts?",
        top_k=5,
        retrieval_mode=None,
    )

    assert response.status == "rejected"
    assert response.metadata.sql.reason_code == "sql_admin_only_initial_rollout"
    assert sql_service.propose_calls == 0


def test_sql_generation_failure_returns_a_rejected_response_not_a_500() -> None:
    sql_service = _FakeSQLService(error=SQLGenerationFailedError("parse_error"))
    orchestrator, _, _ = _orchestrator(
        sql_enabled=True,
        decision=RouteDecision(route=QueryRoute.SQL, confidence=0.97, reason_code="structured"),
        sql_service=sql_service,
    )

    response = orchestrator.answer(
        principal=_admin(),
        conversation_id=1,
        question="how many accounts?",
        top_k=5,
        retrieval_mode=None,
    )

    assert response.status == "rejected"
    assert response.metadata.sql.reason_code == "sql_generation_failed"


def test_reject_route_returns_a_rejected_response() -> None:
    orchestrator, _, _ = _orchestrator(
        sql_enabled=True,
        decision=RouteDecision(
            route=QueryRoute.REJECT, confidence=0.2, reason_code="uncertain_sql_intent"
        ),
    )

    response = orchestrator.answer(
        principal=_admin(), conversation_id=1, question="asdf", top_k=5, retrieval_mode=None
    )

    assert response.status == "rejected"
    assert response.metadata.sql.reason_code == "uncertain_sql_intent"
