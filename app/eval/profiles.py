"""Pipeline profiles: named combinations of RAG feature toggles.

Mirrors ``app.core.config.rag_features.RAGFeatureSettings`` — a profile is
just a named preset of those flags, selected via the eval CLI's
``--profile`` flag (see the Makefile's ``eval-*`` targets).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PipelineProfile(BaseModel):
    """A named combination of retrieval/reasoning features to evaluate."""

    model_config = ConfigDict(frozen=True)

    name: str
    hybrid_search_enabled: bool = False
    reranking_enabled: bool = False
    hyde_enabled: bool = False
    crag_enabled: bool = False
    self_reflective_enabled: bool = False


PROFILES: dict[str, PipelineProfile] = {
    "naive": PipelineProfile(name="naive"),
    "hybrid": PipelineProfile(name="hybrid", hybrid_search_enabled=True),
    "hybrid+rerank": PipelineProfile(
        name="hybrid+rerank", hybrid_search_enabled=True, reranking_enabled=True
    ),
    "hybrid+rerank+hyde": PipelineProfile(
        name="hybrid+rerank+hyde",
        hybrid_search_enabled=True,
        reranking_enabled=True,
        hyde_enabled=True,
    ),
    "hybrid+rerank+crag": PipelineProfile(
        name="hybrid+rerank+crag",
        hybrid_search_enabled=True,
        reranking_enabled=True,
        crag_enabled=True,
    ),
    "all": PipelineProfile(
        name="all",
        hybrid_search_enabled=True,
        reranking_enabled=True,
        hyde_enabled=True,
        crag_enabled=True,
        self_reflective_enabled=True,
    ),
}


def get_profile(name: str) -> PipelineProfile:
    try:
        return PROFILES[name]
    except KeyError:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {name!r}. Available: {available}") from None
