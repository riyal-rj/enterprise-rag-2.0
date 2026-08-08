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

    hyde_num_hypotheses: int = Field(default=3, gt=0)
    hyde_enabled_by_default: bool = Field(default=False)

    hybrid_search_enabled: bool = Field(default=True)
    rrf_k: int = Field(default=60, gt=0)

    reranker_backend: RerankerBackend = Field(default="local")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    voyage_api_key: SecretStr = Field(default=SecretStr(""))
    voyage_model: str = Field(default="rerank-2.5")
    reranker_initial_top_k: int = Field(default=20, gt=0)
    reranking_enabled_by_default: bool = Field(default=True)

    crag_relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    crag_ambiguous_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    crag_enabled_by_default: bool = Field(default=True)

    reflection_min_score: float = Field(default=0.85, ge=0.0, le=1.0)
    max_reflection_retries: int = Field(default=2, ge=0)
    self_reflective_enabled_by_default: bool = Field(default=False)
