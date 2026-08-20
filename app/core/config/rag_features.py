"""Retrieval/reasoning strategy toggles and their tuning parameters."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr

from app.core.config.base import EnvBaseSettings

RerankerBackend = Literal["local", "voyage"]


class RAGFeatureSettings(EnvBaseSettings):
    """Feature flags and parameters for the retrieval/reasoning pipeline.

    Each ``*_enabled_by_default`` flag selects a Strategy implementation at
    runtime (e.g. hybrid vs. dense-only retrieval, self-reflective vs.
    single-pass answering); ``reranker_backend`` similarly selects the
    reranking Strategy/Adapter (local cross-encoder vs. Voyage API).
    """

    hyde_enabled_by_default: bool = Field(default=False)
    hyde_model: str = Field(default="gpt-4o-mini", min_length=1)
    hyde_prompt_version: str = Field(default="bank-policy-v1", min_length=1)
    hyde_num_hypotheses: int = Field(default=3, ge=1, le=5)
    hyde_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    hyde_max_completion_tokens: int = Field(default=600, ge=64, le=2_000)
    hyde_timeout_seconds: float = Field(default=12.0, gt=0.0, le=60.0)
    hyde_max_attempts: int = Field(default=2, ge=1, le=3)

    hybrid_candidate_top_k: int = Field(default=20, gt=0)
    hybrid_search_enabled: bool = Field(default=True)
    rrf_k: int = Field(default=60, gt=0)

    reranker_backend: RerankerBackend = Field(default="local")
    # Canonical Hub id (no dash before "6") - see
    # https://sbert.net/docs/cross_encoder/pretrained_models.html. The
    # dashed form resolves too (Hub redirects it), but pin the documented
    # spelling rather than rely on that redirect staying in place.
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L6-v2")
    voyage_api_key: SecretStr = Field(default=SecretStr(""))
    voyage_model: str = Field(default="rerank-2.5")
    reranker_initial_top_k: int = Field(default=20, gt=0)
    # Off by default: the local cross-encoder cold-starts (downloads/loads
    # model weights) on the first request that actually reranks, which can
    # be a large, unexpected latency spike - or a hard failure - in a
    # restricted production container. Enable deliberately per-deployment
    # once the model is preloaded/packaged and an offline RAGAS comparison
    # has validated the quality lift (see app.eval.profiles' rerank
    # variants), not as an out-of-the-box default.
    reranking_enabled_by_default: bool = Field(default=False)

    # Embedding-similarity ("semantic") cache: matches a new question
    # against previously-answered paraphrases instead of only exact text,
    # so two differently-worded questions with the same meaning can still
    # hit the cache. Conservative default threshold - a false-positive
    # match would silently serve a subtly wrong policy answer, which
    # matters more than the extra cache misses a stricter threshold costs.
    # Off by default for the same rollout-safety reason as reranking above -
    # this subsystem (its own Qdrant collection, no TTL/eviction yet - see
    # QdrantSemanticQueryCache) deserves a deliberate opt-in, not to be live
    # the moment this settings module is imported in a fresh deployment.
    semantic_cache_enabled: bool = Field(default=False)
    semantic_cache_similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    semantic_cache_collection: str = Field(default="rag_query_cache")

    crag_enabled_by_default: bool = Field(default=False)
    # No corresponding DB "by default" seed field exists for crag_web_enabled
    # itself (it defaults FALSE directly in migration 008's column DEFAULT,
    # with no settings-level seed the way most other flags have) - this
    # field exists purely so app.core.config.settings's production-hardening
    # check has a frozen deploy-time value to read alongside
    # sql.sql_enabled_by_default, not to seed a new column.
    crag_web_enabled_by_default: bool = Field(default=False)
    crag_grader_model: str = Field(default="gpt-4o-mini", min_length=1)
    # v2: grader system prompt tightened to stop conflating "wrong_policy"
    # with "on-topic chunk that doesn't restate the question's exact
    # scenario" - bumped so a stale cached CRAG answer computed under v1's
    # over-eager wrong_policy behavior can't keep being served under an
    # unchanged cache identity (see StructuredLLMRetrievalGrader.cache_namespace).
    crag_prompt_version: str = Field(default="bank-policy-v2", min_length=1)
    crag_relevance_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    crag_ambiguous_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    crag_max_chunks_to_grade: int = Field(default=8, ge=1, le=20)
    crag_max_documents_to_refine: int = Field(default=5, ge=1, le=10)
    crag_max_sentences_per_document: int = Field(default=80, ge=5, le=200)
    crag_grader_timeout_seconds: float = Field(default=10.0, gt=0.0, le=30.0)
    crag_refiner_timeout_seconds: float = Field(default=10.0, gt=0.0, le=30.0)
    crag_max_completion_tokens: int = Field(default=800, ge=128, le=2_000)
    crag_max_attempts: int = Field(default=2, ge=1, le=2)
    crag_min_evidence_chunks: int = Field(default=1, ge=1, le=5)
    crag_policy_version: str = Field(default="crag-policy-v1", min_length=1)
    # Rewrites a conversational question into a short, keyword-oriented
    # query before it hits the domain-restricted regulatory web search - see
    # app.rag_services.crag.web_query_formulator's module docstring for why
    # this exists at all. Short max-tokens: the output is a few keywords,
    # not prose.
    crag_web_query_model: str = Field(default="gpt-4o-mini", min_length=1)
    crag_web_query_prompt_version: str = Field(default="web-query-v1", min_length=1)
    crag_web_query_timeout_seconds: float = Field(default=5.0, gt=0.0, le=15.0)
    crag_web_query_max_completion_tokens: int = Field(default=100, ge=16, le=300)

    self_reflective_enabled_by_default: bool = Field(default=False)
    reflection_critic_model: str = Field(default="gpt-4o-mini", min_length=1)
    reflection_reviser_model: str = Field(default="gpt-4o-mini", min_length=1)
    reflection_prompt_version: str = Field(default="bank-policy-v1", min_length=1)
    # Answer-relevance acceptance floor - historically named reflection_min_score;
    # kept as-is rather than renamed, since RagOpsConfig/eval goldens don't
    # reference it directly and a rename buys nothing here.
    reflection_min_score: float = Field(default=0.85, ge=0.0, le=1.0)
    reflection_min_evidence_relevance: float = Field(default=0.70, ge=0.0, le=1.0)
    reflection_min_citation_completeness: float = Field(default=0.90, ge=0.0, le=1.0)
    reflection_min_utility: int = Field(default=4, ge=1, le=5)
    # Iteration budget - each iteration is one critique + (revise or
    # retrieve_more) round; the initial answer's first critique doesn't
    # count against this.
    max_reflection_retries: int = Field(default=2, ge=0, le=3)
    reflection_max_additional_retrievals: int = Field(default=1, ge=0, le=1)
    reflection_retrieval_top_k: int = Field(default=5, ge=1, le=10)
    reflection_max_evidence_chunks: int = Field(default=10, ge=1, le=20)
    reflection_max_evidence_chars: int = Field(default=30_000, ge=1_000, le=80_000)
    reflection_max_total_tokens: int = Field(default=6_000, ge=500, le=20_000)
    reflection_max_completion_tokens: int = Field(default=1_500, ge=128, le=4_000)
    reflection_stage_timeout_seconds: float = Field(default=12.0, gt=0.0, le=60.0)
    reflection_total_timeout_seconds: float = Field(default=25.0, gt=0.0, le=90.0)
    reflection_max_attempts: int = Field(default=2, ge=1, le=2)
