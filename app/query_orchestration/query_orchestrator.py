"""Routes a chat request to RAG, SQL (proposal-only), or a rejection -
ahead of either pipeline running.

When ``config.sql_enabled`` is off (the default), this short-circuits
straight to ``RAGService.answer`` without even calling the router - today's
behavior and cost are unchanged for every deployment that hasn't opted in.
See the architecture blueprint's rollout stages: this module implements
"admin proposal-only" (stage 4) - the only stage this release supports;
automatic execution stays out of scope regardless of any admin-mutable flag
(see ``RagRuntimeConfig.__post_init__``'s ``sql_proposal_only`` invariant).
"""

from __future__ import annotations

import logging

from app.core.exceptions import SQLGenerationFailedError
from app.models.auth_user import AuthenticatedUser
from app.query_orchestration.intent_router import IntentRouter
from app.rag_services.rag_runtime_config import RagRuntimeConfigStore
from app.rag_services.rag_service import RAGService
from app.schemas.chat import ChatResponse as ChatResponseSchema
from app.schemas.chat import (
    CRAGMetadata,
    HyDEMetadata,
    RerankingMetadata,
    ResponseMetadata,
    SelfReflectionMetadata,
    SQLMetadata,
)
from app.sql.models import QueryRoute, RouteDecision, SQLProposal, to_sql_principal
from app.sql.sql_service import SQLService

logger = logging.getLogger(__name__)

_DISABLED_HYDE = HyDEMetadata(enabled=False)
_DISABLED_RERANKING = RerankingMetadata(enabled=False, backend="none")
_DISABLED_CRAG = CRAGMetadata(enabled=False)
_DISABLED_SELF_REFLECTION = SelfReflectionMetadata(enabled=False)


class QueryOrchestrator:
    def __init__(
        self,
        *,
        rag_service: RAGService,
        router: IntentRouter,
        sql_service: SQLService,
        config_store: RagRuntimeConfigStore,
    ) -> None:
        self._rag_service = rag_service
        self._router = router
        self._sql_service = sql_service
        self._config_store = config_store

    def answer(
        self,
        *,
        principal: AuthenticatedUser,
        conversation_id: int,
        question: str,
        top_k: int,
        retrieval_mode: str | None,
    ) -> ChatResponseSchema:
        config = self._config_store.current
        if not config.sql_enabled:
            return self._rag_service.answer(question, top_k=top_k, retrieval_mode=retrieval_mode)

        guarded_question = self._guard_input(question)
        decision = self._router.route(guarded_question)

        if decision.route is QueryRoute.RAG:
            return self._rag_service.answer(
                guarded_question, top_k=top_k, retrieval_mode=retrieval_mode
            )

        if decision.route in (QueryRoute.SQL, QueryRoute.HYBRID_RAG_SQL):
            if not principal.is_admin:
                return self._sql_response(
                    decision,
                    status="rejected",
                    answer=(
                        "This question needs structured-data access, which is currently "
                        "limited to administrators while Text-to-SQL is in early rollout."
                    ),
                    sql_metadata=SQLMetadata(
                        enabled=True, reason_code="sql_admin_only_initial_rollout"
                    ),
                )
            try:
                proposal = self._sql_service.propose(
                    principal=to_sql_principal(principal),
                    conversation_id=conversation_id,
                    question=guarded_question,
                )
            except SQLGenerationFailedError as exc:
                logger.info(
                    "query_orchestrator.sql_generation_failed",
                    extra={"username": principal.username, "error": str(exc)},
                )
                return self._sql_response(
                    decision,
                    status="rejected",
                    answer=(
                        "A safe SQL query could not be generated for this question. "
                        "Try rephrasing it or ask about a narrower set of data."
                    ),
                    sql_metadata=SQLMetadata(enabled=True, reason_code="sql_generation_failed"),
                )
            return self._proposal_response(decision, proposal)

        return self._sql_response(
            decision,
            status="rejected",
            answer="This question could not be safely routed to either document search or data lookup.",
            sql_metadata=SQLMetadata(enabled=True, reason_code=decision.reason_code),
        )

    def _guard_input(self, question: str) -> str:
        """Hook point for the input-guardrail pipeline the architecture
        blueprint requires before SQL treatment can exceed 0% for real
        traffic (see its Non-negotiable controls). A pass-through today -
        SQL stays admin-only and proposal-only regardless, so nothing
        downstream depends on this doing more yet."""
        return question

    def _proposal_response(
        self, decision: RouteDecision, proposal: SQLProposal
    ) -> ChatResponseSchema:
        sql_metadata = SQLMetadata(
            enabled=True,
            proposal_id=str(proposal.id),
            proposal_status=proposal.status.value,
            normalized_sql=proposal.sql,
            referenced_tables=list(proposal.referenced_tables),
            expires_at=proposal.expires_at,
            catalog_version=proposal.catalog_version,
            policy_version=proposal.policy_version,
        )
        return ChatResponseSchema(
            status="approval_required",
            answer=(
                "A SQL query has been generated for this question and is waiting for your "
                "approval before it runs. Review the query and approve or reject it."
            ),
            sources=[],
            confidence=decision.confidence,
            metadata=self._metadata(decision.route.value, sql_metadata),
        )

    def _sql_response(
        self,
        decision: RouteDecision,
        *,
        status: str,
        answer: str,
        sql_metadata: SQLMetadata,
    ) -> ChatResponseSchema:
        return ChatResponseSchema(
            status=status,  # type: ignore[arg-type]
            answer=answer,
            sources=[],
            confidence=decision.confidence,
            metadata=self._metadata(decision.route.value, sql_metadata),
        )

    @staticmethod
    def _metadata(route: str, sql_metadata: SQLMetadata) -> ResponseMetadata:
        return ResponseMetadata(
            route=route,
            hyde=_DISABLED_HYDE,
            reranking=_DISABLED_RERANKING,
            crag=_DISABLED_CRAG,
            self_reflection=_DISABLED_SELF_REFLECTION,
            retrieved_chunks=[],
            sql=sql_metadata,
        )
