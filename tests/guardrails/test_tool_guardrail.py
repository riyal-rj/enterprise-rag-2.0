"""Tool authorization matrix: admin/non-admin x lockdown on/off."""

from __future__ import annotations

import pytest

from app.guardrails.contracts import GuardrailBlockedError, GuardrailStage
from app.guardrails.tool_guardrail import ToolGuardrail
from app.models.auth_user import AuthenticatedUser
from app.rag_services.rag_runtime_config import RagRuntimeConfig


def _config(*, safety_lockdown_enabled: bool = False) -> RagRuntimeConfig:
    return RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=False,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
        sql_enabled=True,
        sql_rollout_percentage=100,
        safety_lockdown_enabled=safety_lockdown_enabled,
    )


def test_admin_is_authorized_when_not_locked_down() -> None:
    ToolGuardrail().authorize_sql(
        principal=AuthenticatedUser(username="admin", is_admin=True), config=_config()
    )


def test_non_admin_is_denied() -> None:
    with pytest.raises(GuardrailBlockedError) as excinfo:
        ToolGuardrail().authorize_sql(
            principal=AuthenticatedUser(username="alice", is_admin=False), config=_config()
        )

    assert excinfo.value.stage is GuardrailStage.TOOL


def test_lockdown_denies_even_an_admin() -> None:
    """safety_lockdown is a security-incident switch: it force-closes the
    SQL route regardless of sql_enabled or who is asking, which is exactly
    what distinguishes it from emergency_disabled."""
    with pytest.raises(GuardrailBlockedError):
        ToolGuardrail().authorize_sql(
            principal=AuthenticatedUser(username="admin", is_admin=True),
            config=_config(safety_lockdown_enabled=True),
        )


def test_lockdown_denies_a_non_admin_too() -> None:
    with pytest.raises(GuardrailBlockedError):
        ToolGuardrail().authorize_sql(
            principal=AuthenticatedUser(username="alice", is_admin=False),
            config=_config(safety_lockdown_enabled=True),
        )


def test_denial_message_never_names_the_reason() -> None:
    with pytest.raises(GuardrailBlockedError) as excinfo:
        ToolGuardrail().authorize_sql(
            principal=AuthenticatedUser(username="alice", is_admin=False), config=_config()
        )

    assert str(excinfo.value) == "Your request could not be processed."
