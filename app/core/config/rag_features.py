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

    reflection_min_score: float = Field(default=0.85, ge=0.0, le=1.0)
    max_reflection_retries: int = Field(default=2, ge=0)
    self_reflective_enabled_by_default: bool = Field(default=False)
