"""Invokes the system under test (Strategy: transport into the pipeline).

``ServiceInvoker`` calls the RAG/SQL pipeline in-process (direct Python
call — what CI should use). An HTTP-based invoker for ``--mode api``
doesn't exist yet, matching ``run_ragas.main``'s own "Phase B" placeholder
for that mode. Neither actually reaches a real pipeline yet — no RAG/SQL
engine exists in this repo — so ``ServiceInvoker.invoke`` raises
:class:`SkippedIntent` for every question until one is wired up here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.eval.profiles import PipelineProfile
from app.eval.schemas import Intent


class SkippedIntent(Exception):
    """Raised when a case can't be run against the current wiring.

    Not a failure — the caller buckets these separately from scored rows
    (see ``run_ragas.main``'s ``skipped`` list).
    """


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved context passage, in the shape RAGAS needs."""

    text: str
    source: str


@dataclass(frozen=True)
class InvokeResponse:
    """The system under test's final answer for one question."""

    answer: str
    sources: list[str] = field(default_factory=list)


class Invoker(Protocol):
    """Contract for reaching the system under test."""

    def invoke(
        self, question: str, flags: PipelineProfile, intent: Intent
    ) -> tuple[InvokeResponse, list[RetrievedChunk]]: ...


class ServiceInvoker:
    """In-process invoker. Placeholder until the real pipeline exists.

    Replace the body here with a direct call into the actual RAG/SQL
    service once it exists (e.g. ``app.rag.pipeline.answer(question,
    flags, intent)``), returning its answer/sources plus the chunks it
    retrieved.
    """

    def invoke(
        self, question: str, flags: PipelineProfile, intent: Intent
    ) -> tuple[InvokeResponse, list[RetrievedChunk]]:
        raise SkippedIntent(f"service pipeline not wired yet (intent={intent.value})")
