"""Sanitized audit trail for guardrail decisions - see ``app.guardrails``
and ``app.repositories.security_events_repository``.

Same append-only shape as ``app.models.rag_ops.RagOpsAuditEntry`` (the
existing ``rag_ops_audit_log`` template), but a separate table: these rows
are keyed by request/finding, not by an admin actor mutating global config,
and are written far more often (every BLOCK/REDACT decision, not every
config change).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SecurityEvent:
    """One recorded guardrail decision. ``changes`` is a sanitized finding
    summary (category/score/detector only, see
    ``app.guardrails.security_events.summarize_decision``) - never raw
    scanned text, matched content, or the user's question/answer."""

    id: int
    actor: str | None
    action: str
    stage: str
    category: str | None
    mode: str
    changes: dict[str, Any]
    reason: str | None
    created_at: datetime
