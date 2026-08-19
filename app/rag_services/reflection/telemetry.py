"""Fleet-wide (cross-worker/cross-process) self-reflection telemetry contract.

Mirrors ``app.rag_services.crag.telemetry`` exactly, for the same reason:
``app.services.rag_metrics_service.RagMetricsService`` is process-local -
each worker's rolling window only reflects the traffic *that worker*
handled, which is fine for the single-process admin panel it backs but
cannot answer a fleet-wide rollout question ("what's the abstention rate
across every worker in production right now?"). ``SelfReflectionTelemetry``
is the seam for a real metrics backend (Prometheus, OpenTelemetry, ...) to
answer that - this module defines the contract and label discipline only;
wiring an actual exporter is a deployment-specific decision left to whoever
operates this service (no telemetry SDK is a hard dependency of this
codebase).

Required, low-cardinality metrics a concrete implementation should export -
all counts below are over calls to
:meth:`SelfReflectionTelemetry.record_attempt`, distinguished by ``served``
(``cohort == "treatment"``) vs. ``cohort == "shadow"``
(``rag_self_reflection_shadow_decision_total``):

* ``rag_self_reflection_attempts_total`` - every ``record_attempt`` call with ``served=True``
* ``rag_self_reflection_fallback_total`` - ``served=True`` calls with ``fallback=True``
* ``rag_self_reflection_abstentions_total`` - ``served=True`` calls with ``abstain=True``
* ``rag_self_reflection_accepted_total`` - ``served=True`` calls with ``final_action="accept"``
* ``rag_self_reflection_duration_ms`` - a histogram/summary over ``duration_ms``
* ``rag_self_reflection_tokens_total`` - a counter summing ``usage_tokens`` (the
  primary cost-alerting signal - a deployment should alert on a sustained
  jump in this counter's rate, same as it would for any other per-request
  LLM token spend)
* ``rag_self_reflection_iterations`` - a histogram over ``iterations`` (a
  sustained rise here is the leading indicator of a cost/latency regression
  before ``tokens_total`` itself moves enough to alert on)
* ``rag_self_reflection_shadow_decision_total`` - every call with
  ``cohort == "shadow"`` (``served=False``)

Allowed labels: ``cohort``, ``final_action``, ``fallback``, ``abstain``,
``served``, ``model``. Never label a metric with ``question``, ``username``,
a source URL, a policy filename, or a conversation ID - those are
unbounded/PII-bearing and would blow up cardinality on a real time series
backend.
"""

from __future__ import annotations

from typing import Protocol


class SelfReflectionTelemetry(Protocol):
    """Fleet-wide exporter for one self-reflection attempt (treatment or
    shadow - see module docstring). Called once per attempt from
    ``RAGService.answer`` alongside (not instead of) the process-local
    ``RagMetricsService`` recording."""

    def record_attempt(
        self,
        *,
        cohort: str,
        final_action: str | None,
        fallback: bool,
        abstain: bool,
        iterations: int,
        additional_retrievals: int,
        duration_ms: float,
        usage_tokens: int,
        served: bool,
    ) -> None: ...


class NoOpSelfReflectionTelemetry:
    """Default when no fleet-wide exporter is configured for this
    deployment - lets ``RAGService`` call ``record_attempt`` unconditionally
    instead of every call site guarding on ``self._telemetry is not None``,
    matching ``NoOpCRAGTelemetry``'s role."""

    def record_attempt(
        self,
        *,
        cohort: str,
        final_action: str | None,
        fallback: bool,
        abstain: bool,
        iterations: int,
        additional_retrievals: int,
        duration_ms: float,
        usage_tokens: int,
        served: bool,
    ) -> None:
        pass
