"""Request/response schemas for the ``/chat`` (RAG) route."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal["dense", "hybrid"]
EvidenceOriginValue = Literal["policy", "regulatory_web"]


class ChatRequest(BaseModel):
    """Body for ``POST /chat``."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, gt=0, le=50)
    retrieval_mode: RetrievalMode | None = Field(
        default=None,
        description="Overrides the server-configured default retrieval strategy for this request.",
    )
    conversation_id: int | None = Field(
        default=None,
        description="Continues an existing conversation thread; omit to start a new one.",
    )


class RetrievedChunkPreview(BaseModel):
    """A chunk as surfaced back to the caller, alongside the answer.

    ``score`` is always the retrieval-stage score (dense cosine similarity,
    RRF-derived for hybrid, or BM25 for sparse) - reranking never mutates
    it, only reorders the list. ``rerank_score``/``original_rank`` are
    ``None`` unless reranking actually ran; when they're set,
    ``original_rank`` is this chunk's 1-based position in the
    pre-reranking retrieval order, so a caller can tell e.g. "the
    reranker promoted the candidate retrieval ranked #8 to position #1."
    """

    text: str
    source: str
    score: float
    page_number: int | None = None
    rerank_score: float | None = None
    original_rank: int | None = None


class EvidencePreview(BaseModel):
    """Final evidence actually supplied to the answer model.

    Distinct from ``RetrievedChunkPreview``: that field reflects raw
    retrieval/reranking output, while this reflects CRAG's post-grading,
    post-refinement, possibly web-augmented evidence set - the exact
    context the answer LLM saw. ``canonical_url``/``retrieved_at_iso`` are
    only set for ``origin="regulatory_web"`` items.
    """

    text: str
    source: str
    page_number: int | None = None
    score: float
    origin: EvidenceOriginValue
    canonical_url: str | None = None
    retrieved_at_iso: str | None = None


class RerankingMetadata(BaseModel):
    """Diagnostic detail about the reranking stage specifically.

    Always present so a caller can distinguish "reranking is off"
    (``enabled=False``) from "reranking is on but degraded to retrieval
    order" (``enabled=True, applied=False, fallback=True``) - the same
    distinction ``FailOpenReranker``/``ReRankOutcome`` already carry
    internally (see ``app.rag_services.reranker.reranker``), now surfaced instead
    of being discarded at the API boundary.
    """

    enabled: bool
    applied: bool = False
    fallback: bool = False
    backend: str
    candidate_count: int = 0
    usage_tokens: int | None = None


class HyDEMetadata(BaseModel):
    """Diagnostic detail about the HyDE query-transformation stage.

    ``enabled=True`` only means the HyDE feature toggle was on and the
    resolved retrieval strategy uses dense embeddings for this request - it
    does **not** imply the query was sampled into the rollout treatment
    cohort. A query can have ``enabled=True`` and still show
    ``applied=False, bypass_reason="rollout"`` (sampled into the control
    cohort) or ``bypass_reason="emergency_disabled"`` (kill switch active) -
    see ``DynamicQueryTransformer.plan`` in
    ``app.rag_services.hyde.dynamic_query_transformer`` for how the
    disabled/control/treatment cohort is actually decided. ``bypass_reason``
    is the field that distinguishes those cases; ``enabled`` alone does not.
    This mirrors the disabled/bypass/applied/fallback distinction
    ``RerankingMetadata`` carries for reranking. Hypothetical passage text
    itself is never included here - it's a retrieval probe, not evidence,
    and must not leak into the response, logs, or cache (see
    ``app.rag_services.hyde.hyde_query_transformer``).
    """

    enabled: bool
    applied: bool = False
    fallback: bool = False
    backend: str = "none"
    hypothesis_count: int = 0
    usage_tokens: int = 0
    duration_ms: float = 0.0
    bypass_reason: str | None = None


class CRAGMetadata(BaseModel):
    """Diagnostic detail about the Corrective-RAG stage.

    ``enabled=True`` only means the CRAG feature toggle was on for this
    request - it does **not** imply the query was sampled into the rollout
    treatment cohort. A query can have ``enabled=True`` and still show
    ``applied=False, bypass_reason="rollout"`` (sampled into the control
    cohort) or ``bypass_reason="emergency_disabled"`` - same
    enabled/bypass/applied/fallback distinction ``HyDEMetadata`` carries for
    HyDE. ``decision`` is ``None`` unless CRAG actually graded retrieval for
    this request. Raw grader reasons, refiner selections, and web snippet
    text are deliberately never included here - only counts and identifiers
    (see ``app.rag_services.rag_service._build_evidence_context``).
    """

    enabled: bool
    applied: bool = False
    decision: Literal["correct", "ambiguous", "incorrect"] | None = None
    fallback: bool = False
    abstain: bool = False
    web_used: bool = False
    evidence_count: int = 0
    usage_tokens: int = 0
    duration_ms: float = 0.0
    bypass_reason: str | None = None


class SelfReflectionMetadata(BaseModel):
    """Diagnostic detail about the self-reflective answer critique/revision
    stage.

    ``enabled=True`` only means the feature toggle was on for this request -
    it does **not** imply the query was sampled into the rollout treatment
    cohort. A query can have ``enabled=True`` and still show
    ``applied=False, bypass_reason="rollout"`` (control cohort) or
    ``bypass_reason="emergency_disabled"`` - same enabled/bypass/applied/
    fallback distinction ``CRAGMetadata``/``HyDEMetadata`` carry.
    ``final_action`` is ``None`` unless reflection actually ran for this
    request. Raw critic reasoning, missing-aspect/unsupported-claim text,
    and revision history are deliberately never included here - only
    counts, scores, and identifiers (see
    ``app.rag_services.reflection.reflection.SelfReflectionOutcome``).
    """

    enabled: bool
    applied: bool = False
    accepted: bool = False
    final_action: Literal["accept", "revise", "retrieve_more", "abstain"] | None = None
    iterations: int = 0
    additional_retrievals: int = 0
    fallback: bool = False
    abstain: bool = False
    support_level: Literal["full", "partial", "none"] | None = None
    answer_relevance: float | None = None
    citation_completeness: float | None = None
    utility: int | None = None
    usage_tokens: int = 0
    duration_ms: float = 0.0
    bypass_reason: str | None = None


class SQLMetadata(BaseModel):
    """Diagnostic detail about the Text-to-SQL route.

    ``enabled=True`` only means the SQL feature toggle was on for this
    request - most fields stay ``None``/empty unless the router actually
    selected ``sql``/``hybrid_rag_sql`` for this question. ``normalized_sql``
    (the AST-policy-rewritten query, not the raw model output) is only ever
    populated for the principal that owns the proposal - see
    ``app.query_orchestration.query_orchestrator.QueryOrchestrator``. Raw
    result rows are never included here - only counts and identifiers,
    same discipline ``CRAGMetadata``/``SelfReflectionMetadata`` apply to
    their own internals.
    """

    enabled: bool
    proposal_id: str | None = None
    proposal_status: str | None = None
    normalized_sql: str | None = None
    referenced_tables: list[str] = Field(default_factory=list)
    expires_at: datetime | None = None
    row_count: int | None = None
    truncated: bool = False
    execution_ms: float | None = None
    catalog_version: int | None = None
    policy_version: str | None = None
    reason_code: str | None = None


class GuardrailMetadata(BaseModel):
    """Diagnostic detail about the guardrails layer (``app.guardrails``).

    Categories only, never raw scanned text or matched content - same
    discipline ``CRAGMetadata``/``SelfReflectionMetadata`` apply to their
    own internals. ``mode="enforce"``/``input_action="allow"`` is the
    default so every pre-existing construction site (tests, the eval
    harness) keeps working unchanged.
    """

    mode: Literal["enforce", "monitor"] = "enforce"
    input_action: Literal["allow", "redact", "block"] = "allow"
    context_chunks_dropped: int = 0
    output_action: Literal["allow", "redact", "block"] = "allow"
    categories_flagged: list[str] = Field(default_factory=list)
    policy_version: str = ""


class ResponseMetadata(BaseModel):
    """Diagnostic detail about how an answer was produced.

    ``route`` was fixed to ``"rag"`` before ``QueryOrchestrator`` existed;
    it's now a real routing decision (``"rag"``/``"sql"``/``"hybrid_rag_sql"``/
    ``"reject"``) whenever SQL routing is enabled, and still ``"rag"``
    unconditionally when it's not (see
    ``app.query_orchestration.query_orchestrator.QueryOrchestrator``).
    """

    route: str
    retrieval_mode: str = "dense"
    hyde: HyDEMetadata
    reranking: RerankingMetadata
    crag: CRAGMetadata
    self_reflection: SelfReflectionMetadata
    # Raw retrieval/reranking output.
    retrieved_chunks: list[RetrievedChunkPreview]
    # Final post-CRAG evidence sent to the answer LLM.
    final_evidence: list[EvidencePreview] = Field(default_factory=list)
    flagged_claims: list[str] = Field(default_factory=list)
    # Defaulted (disabled) so every pre-existing construction site
    # (RAGService.answer, tests) keeps working unchanged - same reasoning as
    # RagRuntimeConfig's crag_shadow_enabled/self_reflective_enabled fields.
    sql: SQLMetadata = Field(default_factory=lambda: SQLMetadata(enabled=False))
    # Guardrails layer - see app.guardrails. Defaulted (allow/enforce) so
    # every pre-existing construction site keeps working unchanged, same
    # reasoning as sql above.
    guardrail: GuardrailMetadata = Field(default_factory=GuardrailMetadata)


class ChatResponse(BaseModel):
    """Response for ``POST /chat``.

    ``status`` defaults to ``"completed"`` - the only value every existing
    caller has ever seen, since SQL routing (the only path that can produce
    ``"approval_required"``/``"rejected"``) is off by default. See
    ``app.query_orchestration.query_orchestrator.QueryOrchestrator``.
    """

    status: Literal["completed", "approval_required", "rejected"] = "completed"
    answer: str
    sources: list[str]
    confidence: float
    cache_hit: bool = False
    metadata: ResponseMetadata
    # RAGService (retrieval/generation only) doesn't know about conversations;
    # this defaults to 0 there and is always overwritten by ChatController
    # with the real thread id before the response reaches a caller.
    conversation_id: int = 0


class ChatHistoryItem(BaseModel):
    """A single past question/answer turn, as surfaced back to its owner."""

    id: int
    question: str
    answer: str
    sources: list[str]
    confidence: float
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    """Response for ``GET /chat/history``, most recent turn first.

    Also backs ``GET /chat/conversations/{id}/messages`` - a conversation's
    full thread is the same shape, oldest turn first.
    """

    items: list[ChatHistoryItem]


class ConversationSummary(BaseModel):
    """A conversation thread, as surfaced in the sidebar's history list."""

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """Response for ``GET /chat/conversations``, most recently active first."""

    items: list[ConversationSummary]
