"""Shared helper for building the sanitized ``security_events.changes``
payload from a :class:`app.guardrails.contracts.GuardrailDecision`.

One function, used by every pipeline that records to
``app.repositories.security_events_repository.SecurityEventsRepository``,
so the "category/score/detector only, never raw text" discipline is
enforced in one place rather than trusted to each call site.
"""

from __future__ import annotations

from typing import Any

from app.guardrails.contracts import GuardrailDecision


def summarize_decision(decision: GuardrailDecision) -> dict[str, Any]:
    return {
        "findings": [
            {"category": f.category.value, "score": f.score, "detector": f.detector}
            for f in decision.findings
        ],
        "would_block": decision.would_block,
    }
