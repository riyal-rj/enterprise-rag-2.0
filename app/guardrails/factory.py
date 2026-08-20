"""Assembles guardrail pipelines from already-built scanner singletons.

Separate from ``app.api.deps`` (the app-wide composition root) the same way
``app.rag_services.crag``/``app.rag_services.reflection`` each keep their
own internal wiring helpers that ``deps.py`` still calls. Nothing here
triggers a model load - that only happens in
``app.guardrails.llm_guard_adapter.build_llm_guard_input_scanner``/
``build_llm_guard_output_scanner``, called once from ``deps.py``'s
``@lru_cache`` factories and passed in here as an already-built (or
``None``) scanner.
"""

from __future__ import annotations

from app.guardrails.context_pipeline import ContextGuardPipeline
from app.guardrails.contracts import OutputScanner, TextScanner
from app.guardrails.ingestion_security import IngestionSecurityScanner
from app.guardrails.input_pipeline import InputGuardPipeline
from app.guardrails.output_pipeline import OutputGuardPipeline
from app.guardrails.policy import GuardrailPolicy
from app.guardrails.sensitive_data import RegexSensitiveDataScanner
from app.guardrails.tool_guardrail import ToolGuardrail
from app.repositories.security_events_repository import SecurityEventsRepository


def build_guardrail_policy() -> GuardrailPolicy:
    return GuardrailPolicy()


def build_input_guard_pipeline(
    *,
    ml_scanner: TextScanner | None,
    policy: GuardrailPolicy,
    security_events: SecurityEventsRepository | None = None,
) -> InputGuardPipeline:
    return InputGuardPipeline(
        deterministic_scanners=(RegexSensitiveDataScanner(),),
        ml_scanner=ml_scanner,
        policy=policy,
        security_events=security_events,
    )


def build_context_guard_pipeline(
    *,
    injection_scanner: TextScanner | None,
    policy: GuardrailPolicy,
    security_events: SecurityEventsRepository | None = None,
) -> ContextGuardPipeline:
    return ContextGuardPipeline(
        injection_scanner=injection_scanner, policy=policy, security_events=security_events
    )


def build_output_guard_pipeline(
    *,
    ml_scanner: OutputScanner | None,
    policy: GuardrailPolicy,
    security_events: SecurityEventsRepository | None = None,
) -> OutputGuardPipeline:
    return OutputGuardPipeline(
        deterministic_scanners=(RegexSensitiveDataScanner(),),
        ml_scanner=ml_scanner,
        policy=policy,
        security_events=security_events,
    )


def build_tool_guardrail(
    *, security_events: SecurityEventsRepository | None = None
) -> ToolGuardrail:
    return ToolGuardrail(security_events=security_events)


def build_ingestion_security_scanner(
    *,
    ml_scanner: TextScanner | None,
    policy: GuardrailPolicy,
    security_events: SecurityEventsRepository | None = None,
) -> IngestionSecurityScanner:
    return IngestionSecurityScanner(
        deterministic_scanners=(RegexSensitiveDataScanner(),),
        ml_scanner=ml_scanner,
        policy=policy,
        security_events=security_events,
    )
