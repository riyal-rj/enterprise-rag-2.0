"""Fleet-wide (cross-worker/cross-process) CRAG telemetry contract.

``app.services.rag_metrics_service.RagMetricsService`` is process-local -
each worker's rolling window only reflects the traffic *that worker*
handled, which is fine for the single-process admin panel it backs but
cannot answer a fleet-wide rollout question ("what's the abstention rate
across every worker in production right now?"). ``CRAGTelemetry`` is the
seam for a real metrics backend (Prometheus, OpenTelemetry, ...) to answer
that - this module defines the contract and label discipline only; wiring
an actual exporter is a deployment-specific decision left to whoever
operates this service (no telemetry SDK is a hard dependency of this
codebase).

Required, low-cardinality metrics a concrete implementation should export -
all four of ``rag_crag_attempts_total``/``rag_crag_fallback_total``/
``rag_crag_abstentions_total``/``rag_crag_web_used_total`` are boolean-gated
counts over calls to :meth:`CRAGTelemetry.record_attempt`, distinguished by
``served`` (``cohort == "treatment"``) vs. ``cohort == "shadow"``
(``rag_crag_shadow_decision_total``):

* ``rag_crag_attempts_total`` - every ``record_attempt`` call with ``served=True``
* ``rag_crag_fallback_total`` - ``served=True`` calls with ``fallback=True``
* ``rag_crag_abstentions_total`` - ``served=True`` calls with ``abstain=True``
* ``rag_crag_web_used_total`` - ``served=True`` calls with ``web_used=True``
* ``rag_crag_duration_ms`` - a histogram/summary over ``duration_ms``
* ``rag_crag_tokens_total`` - a counter summing ``usage_tokens``
* ``rag_crag_shadow_decision_total`` - every call with ``cohort == "shadow"``
  (``served=False``)

Allowed labels: ``cohort``, ``decision``, ``fallback``, ``abstain``,
``web_used``, ``served``, ``model``. Never label a metric with ``question``,
``username``, a source URL, a policy filename, or a conversation ID - those
are unbounded/PII-bearing and would blow up cardinality on a real time
series backend.
"""

from __future__ import annotations

from typing import Protocol


class CRAGTelemetry(Protocol):
    """Fleet-wide exporter for one CRAG correction attempt (treatment or
    shadow - see module docstring). Called once per attempt from
    ``RAGService.answer`` alongside (not instead of) the process-local
    ``RagMetricsService`` recording."""

    def record_attempt(
        self,
        *,
        cohort: str,
        decision: str | None,
        fallback: bool,
        abstain: bool,
        web_used: bool,
        duration_ms: float,
        usage_tokens: int,
        served: bool,
    ) -> None: ...


class NoOpCRAGTelemetry:
    """Default when no fleet-wide exporter is configured for this
    deployment - lets ``RAGService`` call ``record_attempt`` unconditionally
    instead of every call site guarding on ``self._telemetry is not None``,
    matching ``_NoOpRagServiceMetrics``'s role for the process-local
    recorder."""

    def record_attempt(
        self,
        *,
        cohort: str,
        decision: str | None,
        fallback: bool,
        abstain: bool,
        web_used: bool,
        duration_ms: float,
        usage_tokens: int,
        served: bool,
    ) -> None:
        pass
