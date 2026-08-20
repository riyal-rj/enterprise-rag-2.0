"""Tool authorization - deliberately thin.

SQL propose/execute is the only "tool" this codebase has; there is no
general tool-calling framework here and none is added by this module. This
centralizes what was previously an inline ``if not principal.is_admin``
check in ``QueryOrchestrator``, and adds the new ``safety_lockdown``
security-incident kill switch (see
``app.rag_services.rag_runtime_config.RagRuntimeConfig.safety_lockdown_enabled``)
- distinct from ``emergency_disabled``, which degrades rollout-gated
*quality* features rather than force-closing the SQL route.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.guardrails.contracts import GuardrailBlockedError, GuardrailStage
from app.models.auth_user import AuthenticatedUser
from app.repositories.security_events_repository import SecurityEventsRepository

if TYPE_CHECKING:
    from app.rag_services.rag_runtime_config import RagRuntimeConfig


class ToolGuardrail:
    def __init__(self, *, security_events: SecurityEventsRepository | None = None) -> None:
        self._security_events = security_events

    def authorize_sql(self, *, principal: AuthenticatedUser, config: RagRuntimeConfig) -> None:
        """Raises :class:`GuardrailBlockedError` if SQL routing is not
        authorized for this principal/config; no-op (returns) otherwise."""
        if config.safety_lockdown_enabled:
            self._record(principal, config, action="tool_lockdown_denied")
            raise GuardrailBlockedError(stage=GuardrailStage.TOOL)
        if not principal.is_admin:
            self._record(principal, config, action="tool_permission_denied")
            raise GuardrailBlockedError(stage=GuardrailStage.TOOL)

    def _record(
        self, principal: AuthenticatedUser, config: RagRuntimeConfig, *, action: str
    ) -> None:
        if self._security_events is None:
            return
        self._security_events.record(
            actor=principal.username,
            action=action,
            stage=GuardrailStage.TOOL.value,
            category=None,
            mode=config.guardrail_mode,
            changes={"tool": "sql"},
        )
