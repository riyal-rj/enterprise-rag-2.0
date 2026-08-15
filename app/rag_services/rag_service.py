"""Basic RAG answer generation: embed, dense-search, generate, cache.

The full pipeline this is meant to grow into - CRAG relevance grading,
self-reflective answer refinement, HyDE query expansion, reranking,
hybrid/sparse search, SQL intent routing, prompt-injection spotlighting -
layers on top of this. This is deliberately the minimal "retrieve then
generate" core those strategies plug into later, not a placeholder for
any one of them: no query expansion, no relevance grading, no answer
refinement loop, and no prompt-injection defense yet.

Caching follows the same cache-aside shape as
``app.core.llm.embedding_client``: check the cache first, compute and
write back on a miss, treat a cache read/write failure as a miss/no-op
rather than propagate it - the cache is an optimization, not a
correctness requirement.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping

from app.core.exceptions import HybridRetrievalDisabledError
from app.core.llm.chat_client import LLMClient
from app.core.llm.embedding_client import EmbeddingClient
from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.claim_checker import find_unsupported_claims
from app.rag_services.confidence_scorer import compute_confidence_breakdown
from app.rag_services.retrieval_strategy import RetrievalStrategy
from app.schemas.chat import ChatResponse, ResponseMetadata, RetrievedChunkPreview
from app.services.query_cache_service import CacheTier, QueryCacheService

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You are an enterprise banking-policy RAG assistant. Answer exclusively from the supplied policy context.

ANSWERING RULES

1. Give the direct operational answer first. Do not begin with a long explanation.

2. Answer every part of the question separately. If the question asks whether a customer can receive or send transactions, address receiving and sending independently.

3. Before generating the answer, internally identify:

   * Triggering facts
   * Mandatory actions
   * Conditional actions
   * Thresholds and aggregation rules
   * Deadlines and their starting events
   * Escalation and reporting requirements
   * Exceptions
   * Information not stated in the retrieved context

4. Preserve policy conditions exactly. Do not convert “if,” “where,” “unless,” “may,” or “subject to” into unconditional conclusions.

5. Distinguish between:

   * Mandatory now
   * Mandatory only if a stated condition is satisfied
   * Reasonable inference
   * Not established by the supplied context

6. Do not treat a reasonable inference as an explicit policy rule. If the context states “debit freeze,” conclude that account debits are restricted. Do not automatically conclude that incoming credits are either allowed or blocked unless the context explicitly states this.

7. A conditional policy requirement is not “context silence.” For example, if STR/SAR filing is required when suspicion is confirmed, state the conditional filing requirement and its deadlines.

8. For multi-policy scenarios, provide every applicable control needed to answer the question. Check for:

   * KYC or EDD requirements
   * Transaction review requirements
   * CTR aggregation and thresholds
   * STR/SAR escalation and filing deadlines
   * Information-security escalation
   * Privacy or regulatory notification

9. Do not include unrelated facts merely because they are true. A high-risk customer’s annual Re-KYC cycle should not be added unless it changes the required action in the scenario.

10. Use the response format that best fits the query:

* Binary decision: direct answer followed by evidence boundaries
* Multi-action scenario: ordered action list
* Deadline question: chronological table
* Comparison question: comparison table
* Insufficient evidence: explicit abstention identifying the missing policy or procedure

11. Cite every material action, threshold, deadline, exception, and conclusion using:
    [Policy ID, version, page, section]

12. Cite only evidence actually used in the answer. Do not cite unrelated or merely similar retrieved documents.

13. If the required answer is not established by the retrieved context, state:
    “The supplied policy context does not explicitly determine this point.”

14. Never invent a policy, deadline, threshold, permission, prohibition, exception, or operational process.

15. Before returning the answer, verify:

* Every part of the question has been answered.
* Every mandatory claim has supporting evidence.
* Every conditional rule remains conditional.
* All relevant deadlines and thresholds are included.
* No unrelated policy information has been added.

CONFIDENCE RULE

Do not generate an arbitrary numeric confidence score. Confidence should be calculated externally from retrieval quality, claim-evidence coverage, citation entailment, completeness, and calibration data. If a confidence score is supplied by the application, explain any evidence limitation that materially affects it.

"""


class RAGService:
    """Retrieve-then-generate RAG answering, with response caching."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        retrieval_strategies: Mapping[str, RetrievalStrategy],
        llm_client: LLMClient,
        cache: QueryCacheService,
        default_retrieval_mode: str,
        allowed_retrieval_modes: frozenset[str] | None = None,
    ) -> None:
        if default_retrieval_mode not in retrieval_strategies:
            raise ValueError(
                f"default_retrieval_mode {default_retrieval_mode!r} is not among "
                f"the available retrieval_strategies: {sorted(retrieval_strategies)}"
            )
        # None means "everything registered is allowed" - the eval harness's
        # unrestricted RAGService instance uses this, since it must be able
        # to run hybrid/sparse for real before the rollout flag is ever
        # flipped on for real HTTP traffic (see app.eval.invokers).
        if allowed_retrieval_modes is None:
            allowed_retrieval_modes = frozenset(retrieval_strategies)
        if default_retrieval_mode not in allowed_retrieval_modes:
            raise ValueError(
                f"default_retrieval_mode {default_retrieval_mode!r} is not among "
                f"allowed_retrieval_modes: {sorted(allowed_retrieval_modes)}"
            )
        self._embedding_client = embedding_client
        self._retrieval_strategies = retrieval_strategies
        self._llm_client = llm_client
        self._cache = cache
        self._default_retrieval_mode = default_retrieval_mode
        self._allowed_retrieval_modes = allowed_retrieval_modes

    def answer(
        self, question: str, top_k: int = 5, retrieval_mode: str | None = None
    ) -> ChatResponse:
        mode = retrieval_mode or self._default_retrieval_mode
        if mode in self._retrieval_strategies:
            if mode not in self._allowed_retrieval_modes:
                raise HybridRetrievalDisabledError(mode)
            strategy = self._retrieval_strategies[mode]
        else:
            strategy = self._retrieval_strategies[self._default_retrieval_mode]

        cache_key = self._cache_key(question, top_k, strategy.cache_namespace)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached.model_copy(update={"cache_hit": True})

        query_embedding = (
            self._embedding_client.embed_texts([question])[0]
            if strategy.requires_dense_embedding
            else None
        )
        chunks = strategy.retrieve(
            query_text=question,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        user_message = f"{self._build_context(chunks)}\n\nQuestion: {question}"
        llm_response = self._llm_client.generate(_SYSTEM_PROMPT, user_message)

        confidence = compute_confidence_breakdown(
            chunks, llm_response.text, retrieval_mode=strategy.name
        )
        logger.debug(
            "rag.confidence_computed",
            extra={
                "confidence": confidence.total,
                "evidence_coverage": confidence.evidence_coverage,
                "faithfulness": confidence.faithfulness,
                "retrieval_strength": confidence.retrieval_strength,
                "citation_precision": confidence.citation_precision,
                "answerability": confidence.answerability,
            },
        )

        flagged_claims = find_unsupported_claims(llm_response.text, chunks)
        if flagged_claims:
            logger.warning(
                "rag.unsupported_claim_flagged",
                extra={"question": question, "flagged_claims": flagged_claims},
            )

        response = ChatResponse(
            answer=llm_response.text,
            sources=sorted({chunk.source for chunk in chunks}),
            confidence=confidence.total,
            metadata=ResponseMetadata(
                route="rag",
                retrieval_mode=strategy.name,
                retrieved_chunks=[
                    RetrievedChunkPreview(
                        text=c.text, source=c.source, score=c.score, page_number=c.page_number
                    )
                    for c in chunks
                ],
                flagged_claims=flagged_claims,
            ),
        )
        self._set_cached(cache_key, response)
        return response

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context was found."

        sections: list[str] = []
        for chunk in chunks:
            source_label = chunk.source

            if chunk.page_number is not None:
                source_label = f"{source_label} page. {chunk.page_number}"

            sections.append(
                f"[{source_label}]\n{chunk.text}"
            )
        return "\n\n".join(sections)

    def _cache_key(self, question: str, top_k: int, cache_namespace: str) -> str:
       normalized_question = " ".join(question.split())

       raw_key = (
           f"rag:v2:"
           f"{cache_namespace}:"
           f"{top_k}:"
           f"{normalized_question}"
       )

       return hashlib.sha256(raw_key.encode()).hexdigest()

    def _get_cached(self, key: str) -> ChatResponse | None:
        try:
            raw = self._cache.get(CacheTier.RAG_ANSWER, key)
        except Exception as exc:  # noqa: BLE001 - cache is an optimization, not a correctness requirement
            logger.warning("rag.cache_read_failed", extra={"error": str(exc)})
            return None
        return ChatResponse.model_validate_json(raw) if raw is not None else None

    def _set_cached(self, key: str, response: ChatResponse) -> None:
        try:
            self._cache.set(CacheTier.RAG_ANSWER, key, response.model_dump_json())
        except Exception as exc:  # noqa: BLE001 - cache is an optimization, not a correctness requirement
            logger.warning("rag.cache_write_failed", extra={"error": str(exc)})
