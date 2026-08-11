"""The system under test, behind a swappable interface (Strategy).

``QueryPipeline`` is the seam this eval harness plugs into: once real
RAG/SQL orchestration exists elsewhere in the app, wire it up in
``build_pipeline`` below. Until then, ``NotWiredPipeline`` keeps the eval
CLI runnable end-to-end — every case reports an empty answer and fails
loudly, rather than the CLI crashing — so the harness itself (loading,
grading, reporting) is fully testable today.
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, Field

from app.eval.profiles import PipelineProfile

logger = logging.getLogger(__name__)


class PipelineAnswer(BaseModel):
    """What the system under test produced for one question."""

    answer: str
    sources: list[str] = Field(default_factory=list)


class QueryPipeline(Protocol):
    """Contract for the thing being evaluated."""

    async def answer(self, question: str) -> PipelineAnswer: ...


class NotWiredPipeline:
    """Placeholder :class:`QueryPipeline` used until the real RAG/SQL
    orchestration is implemented."""

    def __init__(self, profile: PipelineProfile) -> None:
        self._profile = profile

    async def answer(self, question: str) -> PipelineAnswer:
        logger.warning("eval.pipeline_not_wired", extra={"profile": self._profile.name})
        return PipelineAnswer(answer="", sources=[])


def build_pipeline(profile: PipelineProfile) -> QueryPipeline:
    """Construct the :class:`QueryPipeline` for a given profile.

    Replace this with real dispatch to the RAG/SQL orchestration once it
    exists (e.g. ``app.rag.pipeline.RAGPipeline(profile)``).
    """
    return NotWiredPipeline(profile)
