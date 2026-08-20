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
from app.guardrails.contracts import GuardrailBlockedError, GuardrailDecision
from app.guardrails.input_pipeline import InputGuardPipeline
from app.guardrails.tool_guardrail import ToolGuardrail
from app.models.auth_user import AuthenticatedUser
from app.query_orchestration.intent_router import IntentRouter
from app.rag_services.rag_runtime_config import RagRuntimeConfigStore
from app.rag_services.rag_service import RAGService
from app.rag_services.rollout import sampled_in
from app.schemas.chat import ChatResponse as ChatResponseSchema
from app.schemas.chat import (
    CRAGMetadata,
    GuardrailMetadata,
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
        input_guard: InputGuardPipeline,
        tool_guardrail: ToolGuardrail,
    ) -> None:
        self._rag_service = rag_service
        self._router = router
        self._sql_service = sql_service
        self._config_store = config_store
        self._input_guard = input_guard
        self._tool_guardrail = tool_guardrail

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
        # Runs unconditionally, before any routing decision - guardrails
        # apply to every request regardless of which pipeline (RAG/SQL/
        # hybrid) ends up handling it, not just the SQL-eligible path. See
        # app.guardrails.input_pipeline.InputGuardPipeline.
        try:
            guarded_question, input_decision = self._guard_input(
                question, mode=config.guardrail_mode, actor=principal.username
            )
        except GuardrailBlockedError:
            return self._blocked_response(question)

        # emergency_disabled is the same global kill switch every other
        # feature (HyDE/CRAG/reranking) already gates on - SQL routing must
        # not be an exception to it. See RagRuntimeConfig's docstring.
        if not config.sql_enabled or config.emergency_disabled:
            return self._overlay_input_guardrail(
                self._rag_service.answer(
                    guarded_question, top_k=top_k, retrieval_mode=retrieval_mode
                ),
                input_decision,
            )

        # sql_rollout_percentage gates SQL routing itself, not an
        # enhancement layered on top of it - a question sampled out falls
        # straight back to RAG, the same as sql_enabled=False, rather than
        # reaching the router at all.
        if not sampled_in(guarded_question, config.sql_rollout_percentage, salt="sql"):
            return self._overlay_input_guardrail(
                self._rag_service.answer(
                    guarded_question, top_k=top_k, retrieval_mode=retrieval_mode
                ),
                input_decision,
            )

        decision = self._router.route(guarded_question)

        if decision.route is QueryRoute.RAG:
            return self._overlay_input_guardrail(
                self._rag_service.answer(
                    guarded_question, top_k=top_k, retrieval_mode=retrieval_mode
                ),
                input_decision,
            )

        if decision.route in (QueryRoute.SQL, QueryRoute.HYBRID_RAG_SQL):
            try:
                self._tool_guardrail.authorize_sql(principal=principal, config=config)
            except GuardrailBlockedError:
                reason_code = (
                    "sql_safety_lockdown"
                    if config.safety_lockdown_enabled
                    else "sql_admin_only_initial_rollout"
                )
                answer = (
                    "This data operation is not authorized or is temporarily unavailable."
                    if config.safety_lockdown_enabled
                    else (
                        "This question needs structured-data access, which is currently "
                        "limited to administrators while Text-to-SQL is in early rollout."
                    )
                )
                return self._sql_response(
                    decision,
                    status="rejected",
                    answer=answer,
                    sql_metadata=SQLMetadata(enabled=True, reason_code=reason_code),
                    input_decision=input_decision,
                )

            if decision.route is QueryRoute.HYBRID_RAG_SQL:
                return self._generate_hybrid_answer(
                    decision,
                    principal=principal,
                    conversation_id=conversation_id,
                    question=guarded_question,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                    input_decision=input_decision,
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
                    input_decision=input_decision,
                )
            return self._proposal_response(decision, proposal, input_decision)

        return self._sql_response(
            decision,
            status="rejected",
            answer="This question could not be safely routed to either document search or data lookup.",
            sql_metadata=SQLMetadata(enabled=True, reason_code=decision.reason_code),
            input_decision=input_decision,
        )

    def _generate_hybrid_answer(
        self,
        decision: RouteDecision,
        *,
        principal: AuthenticatedUser,
        conversation_id: int,
        question: str,
        top_k: int,
        retrieval_mode: str | None,
        input_decision: GuardrailDecision,
    ) -> ChatResponseSchema:
        """Real fan-out for HYBRID_RAG_SQL: a grounded RAG answer (real
        citations, synchronous) plus a real generated-and-policy-validated
        SQL proposal for the structured-data half - not the SQL-only path
        this route used to alias to. The SQL half can never be executed
        synchronously here - ``RagRuntimeConfig.__post_init__`` forbids
        ``sql_proposal_only=False``, there is no automatic-execution path
        this release - so the merged answer surfaces the RAG-grounded text
        immediately and attaches the pending proposal for the caller to
        approve via the existing SQL approval endpoint, rather than
        inventing a number that hasn't actually been computed yet.
        """
        rag_response = self._overlay_input_guardrail(
            self._rag_service.answer(question, top_k=top_k, retrieval_mode=retrieval_mode),
            input_decision,
        )
        try:
            proposal = self._sql_service.propose(
                principal=to_sql_principal(principal),
                conversation_id=conversation_id,
                question=question,
            )
        except SQLGenerationFailedError as exc:
            # The structured-data half failed to generate - still return the
            # genuinely useful RAG half rather than failing the whole
            # response, but make the gap visible in metadata rather than
            # silently dropping it.
            logger.info(
                "query_orchestrator.hybrid_sql_generation_failed",
                extra={"username": principal.username, "error": str(exc)},
            )
            degraded_metadata = rag_response.metadata.model_copy(
                update={
                    "route": decision.route.value,
                    "sql": SQLMetadata(enabled=True, reason_code="sql_generation_failed"),
                }
            )
            return rag_response.model_copy(update={"metadata": degraded_metadata})

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
        merged_answer = (
            f"{rag_response.answer}\n\n"
            "A related data lookup has also been generated for the "
            "structured-data part of this question and is waiting for "
            "admin approval before it runs - review and approve it to get "
            "the exact figure."
        )
        merged_metadata = rag_response.metadata.model_copy(
            update={"route": decision.route.value, "sql": sql_metadata}
        )
        return rag_response.model_copy(
            update={
                "status": "approval_required",
                "answer": merged_answer,
                "metadata": merged_metadata,
            }
        )

    def _guard_input(
        self, question: str, *, mode: str, actor: str | None = None
    ) -> tuple[str, GuardrailDecision]:
        """Runs the input guardrail pipeline (canonicalize, deterministic
        secret/PII scan, ML injection/toxicity/banned-topic scan - see
        ``app.guardrails.input_pipeline.InputGuardPipeline``) ahead of
        routing/caching. Raises ``GuardrailBlockedError`` on a BLOCK
        decision; returns the sanitized (possibly PII-redacted) question
        plus the decision otherwise - the decision is threaded through
        every response path below (see ``_overlay_input_guardrail``) so a
        REDACT is still visible in ``metadata.guardrail`` even though
        ``RAGService`` (which builds the rest of that metadata) has no
        knowledge this stage ran at all."""
        return self._input_guard.check_with_decision(
            question, mode=mode, actor=actor  # type: ignore[arg-type]
        )

    @staticmethod
    def _apply_input_guardrail_metadata(
        guardrail: GuardrailMetadata, decision: GuardrailDecision
    ) -> GuardrailMetadata:
        if decision.action.value == "allow":
            return guardrail
        categories = sorted(
            {f.category.value for f in decision.findings} | set(guardrail.categories_flagged)
        )
        return guardrail.model_copy(
            update={"input_action": decision.action.value, "categories_flagged": categories}
        )

    def _overlay_input_guardrail(
        self, response: ChatResponseSchema, decision: GuardrailDecision
    ) -> ChatResponseSchema:
        """Patches a ``RAGService``-built response's ``metadata.guardrail``
        to reflect the input-stage decision, which ``RAGService`` itself
        never sees. No-op when the input was a plain ALLOW."""
        if decision.action.value == "allow":
            return response
        updated_guardrail = self._apply_input_guardrail_metadata(
            response.metadata.guardrail, decision
        )
        updated_metadata = response.metadata.model_copy(update={"guardrail": updated_guardrail})
        return response.model_copy(update={"metadata": updated_metadata})

    def _blocked_response(self, question: str) -> ChatResponseSchema:
        del question  # never echoed back - see GuardrailBlockedError's own discipline
        return ChatResponseSchema(
            status="rejected",
            answer=(
                "I can't process that request safely. Please restate the question "
                "without instructions to reveal or override system behavior."
            ),
            sources=[],
            confidence=0.0,
            metadata=ResponseMetadata(
                route="reject",
                hyde=_DISABLED_HYDE,
                reranking=_DISABLED_RERANKING,
                crag=_DISABLED_CRAG,
                self_reflection=_DISABLED_SELF_REFLECTION,
                retrieved_chunks=[],
                sql=SQLMetadata(enabled=False, reason_code="input_guardrail_blocked"),
                guardrail=GuardrailMetadata(input_action="block"),
            ),
        )

    def _proposal_response(
        self,
        decision: RouteDecision,
        proposal: SQLProposal,
        input_decision: GuardrailDecision,
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
            metadata=self._metadata(decision.route.value, sql_metadata, input_decision),
        )

    def _sql_response(
        self,
        decision: RouteDecision,
        *,
        status: str,
        answer: str,
        sql_metadata: SQLMetadata,
        input_decision: GuardrailDecision,
    ) -> ChatResponseSchema:
        return ChatResponseSchema(
            status=status,  # type: ignore[arg-type]
            answer=answer,
            sources=[],
            confidence=decision.confidence,
            metadata=self._metadata(decision.route.value, sql_metadata, input_decision),
        )

    @classmethod
    def _metadata(
        cls,
        route: str,
        sql_metadata: SQLMetadata,
        input_decision: GuardrailDecision | None = None,
    ) -> ResponseMetadata:
        guardrail = GuardrailMetadata()
        if input_decision is not None:
            guardrail = cls._apply_input_guardrail_metadata(guardrail, input_decision)
        return ResponseMetadata(
            route=route,
            hyde=_DISABLED_HYDE,
            reranking=_DISABLED_RERANKING,
            crag=_DISABLED_CRAG,
            self_reflection=_DISABLED_SELF_REFLECTION,
            retrieved_chunks=[],
            sql=sql_metadata,
            guardrail=guardrail,
        )
