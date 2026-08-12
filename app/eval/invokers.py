"""Invokes the system under test (Strategy: transport into the pipeline).

``ServiceInvoker`` calls the RAG/SQL pipeline in-process (direct Python
call — what CI should use). An HTTP-based invoker for ``--mode api``
doesn't exist yet, matching ``run_ragas.main``'s own "Phase B" placeholder
for that mode.

Kept as a ``Protocol`` (not an ABC): every other swappable interface in
this codebase — ``RateLimiter``, ``PasswordHasher``, ``TokenIssuer``,
``CacheBackend``, ``UserRepository``, ``HealthCheck`` — is a ``Protocol``,
so implementations don't need to inherit from anything, just match the
shape. Switching this one to an ABC would be the odd one out.

``ServiceInvoker`` only attempts intents the pipeline can run headlessly.
SQL and hybrid cases are excluded: Text2SQL requires a human-in-the-loop
``interrupt()`` approval step before execution (see the ``sql``-tagged
golden cases' notes in ``data/goldens.yaml``), which a batch eval run has
no one to answer. ``web_fallback`` additionally requires Tavily to be
configured. Both are checked *before* calling the pipeline, so an
unsupported/misconfigured case is cleanly skipped with a clear reason
rather than erroring partway through a real call.

No RAG/SQL engine exists in this repo yet, so a case that passes both
checks still hits :class:`SkippedIntent` in ``_call_pipeline`` — that's the
one remaining seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.core.config import get_settings
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
    """In-process invoker. Pipeline call is a placeholder until it exists.

    Replace ``_call_pipeline``'s body with a direct call into the actual
    RAG/SQL service once it exists (e.g. ``app.rag.pipeline.answer(question,
    flags)``), translating its response into ``InvokeResponse``/
    ``RetrievedChunk``. The two guard checks in ``invoke`` don't need to
    change when that happens.
    """

    SUPPORTED_INTENTS = frozenset({Intent.RAG, Intent.WEB_FALLBACK})

    def invoke(
        self, question: str, flags: PipelineProfile, intent: Intent
    ) -> tuple[InvokeResponse, list[RetrievedChunk]]:
        if intent not in self.SUPPORTED_INTENTS:
            raise SkippedIntent(
                f"intent={intent.value} not supported in service mode "
                "(sql/hybrid need human-in-the-loop approval, not runnable headlessly)"
            )

        if intent == Intent.WEB_FALLBACK and not self._tavily_configured():
            raise SkippedIntent("tavily_unset: TAVILY_API_KEY not configured")

        return self._call_pipeline(question, flags)

    def _tavily_configured(self) -> bool:
        return bool(get_settings().external_apis.tavily_api_key.get_secret_value())

    def _call_pipeline(
        self, question: str, flags: PipelineProfile
    ) -> tuple[InvokeResponse, list[RetrievedChunk]]:
        raise SkippedIntent("service pipeline not wired yet")
