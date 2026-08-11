"""Pipeline profiles: named combinations of RAG feature toggles.

Loosely mirrors ``app.core.config.rag_features.RAGFeatureSettings`` — a
profile is a named preset of retrieval/reasoning flags, selected via the
eval CLI's ``--profile`` flag (see the Makefile's ``eval-*`` targets).
Not a 1:1 mapping: ``search_mode`` (dense/sparse/hybrid) and ``top_k``
aren't modeled as settings there yet.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

SearchMode = Literal["dense", "sparse", "hybrid"]


class PipelineProfile(BaseModel):
    """A named combination of retrieval/reasoning features to evaluate."""

    model_config = ConfigDict(frozen=True)

    name: str
    search_mode: SearchMode
    enable_hyde: bool = False
    enable_rerank: bool = False
    enable_crag: bool = False
    enable_self_reflective: bool = False
    top_k: int = 5


PROFILES: dict[str, PipelineProfile] = {
    "naive": PipelineProfile(name="naive", search_mode="dense"),
    "sparse_only": PipelineProfile(name="sparse_only", search_mode="sparse"),
    "hybrid": PipelineProfile(name="hybrid", search_mode="hybrid"),
    "hybrid+rerank": PipelineProfile(
        name="hybrid+rerank", search_mode="hybrid", enable_rerank=True
    ),
    "hybrid+rerank+hyde": PipelineProfile(
        name="hybrid+rerank+hyde",
        search_mode="hybrid",
        enable_hyde=True,
        enable_rerank=True,
    ),
    "hybrid+rerank+crag": PipelineProfile(
        name="hybrid+rerank+crag",
        search_mode="hybrid",
        enable_rerank=True,
        enable_crag=True,
    ),
    "all": PipelineProfile(
        name="all",
        search_mode="hybrid",
        enable_hyde=True,
        enable_rerank=True,
        enable_crag=True,
        enable_self_reflective=True,
    ),
}


def get_profile(name: str) -> PipelineProfile:
    try:
        return PROFILES[name]
    except KeyError:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {name!r}. Available: {available}") from None
