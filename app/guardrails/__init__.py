"""Guardrails: input/context/output/tool enforcement around the RAG and SQL
pipelines.

This package is deliberately layered so nothing else in the app needs to
import ``llm_guard`` directly:

- ``contracts`` - shared enums/dataclasses/Protocols every other module
  depends on. Zero business logic.
- ``canonicalizer`` / ``sensitive_data`` - deterministic, dependency-free
  checks (Unicode normalization, regex-based secret/PII detection). Work
  standalone with no ML model.
- ``llm_guard_adapter`` - the *only* module that imports ``llm_guard``.
  Wraps its scanners behind the ``TextScanner``/``OutputScanner`` Protocols
  from ``contracts``.
- ``policy`` - the pure decision function turning scan findings into a
  ``GuardrailDecision`` (allow/redact/block), aware of ``enforce``/
  ``monitor`` mode.
- ``input_pipeline`` / ``context_pipeline`` / ``output_pipeline`` /
  ``tool_guardrail`` / ``ingestion_security`` - the orchestration each call
  site actually uses.
- ``factory`` - assembles pipelines from already-built scanner singletons;
  called from ``app.api.deps`` (the composition root).

See ``app.query_orchestration.query_orchestrator`` (input guard, ahead of
routing) and ``app.rag_services.rag_service`` (context guard around
retrieved evidence, output guard ahead of caching) for the two real
integration points.
"""

from __future__ import annotations

from app.guardrails.contracts import (
    GuardrailAction,
    GuardrailBlockedError,
    GuardrailCategory,
    GuardrailDecision,
    GuardrailMode,
    GuardrailStage,
    OutputScanner,
    ScanFinding,
    TextScanner,
)
from app.guardrails.policy import GuardrailPolicy

__all__ = [
    "GuardrailAction",
    "GuardrailBlockedError",
    "GuardrailCategory",
    "GuardrailDecision",
    "GuardrailMode",
    "GuardrailPolicy",
    "GuardrailStage",
    "OutputScanner",
    "ScanFinding",
    "TextScanner",
]
