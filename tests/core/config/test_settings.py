"""Tests demonstrating that config can be overridden without touching disk."""

from __future__ import annotations

import pytest

from app.core.config.environment import Environment
from app.core.config.settings import Settings, get_settings


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_defaults_are_safe_for_local_development() -> None:
    settings = Settings()
    assert settings.environment is Environment.DEVELOPMENT
    assert settings.storage.storage_backend == "local"


def test_production_requires_real_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET", "")

    with pytest.raises(ValueError, match="insecure/missing"):
        Settings()


def test_production_passes_with_real_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("JWT_SECRET", "a-real-secret")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://user:pw@prod-host:5432/adv_rag"
    )

    settings = Settings()
    assert settings.environment is Environment.PRODUCTION


def test_production_rejects_monitor_mode_with_sql_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("JWT_SECRET", "a-real-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@prod-host:5432/adv_rag")
    monkeypatch.setenv("GUARDRAIL_MODE_DEFAULT", "monitor")
    monkeypatch.setenv("SQL_ENABLED_BY_DEFAULT", "true")

    with pytest.raises(ValueError, match="guardrail mode 'monitor'"):
        Settings()


def test_production_rejects_monitor_mode_with_crag_web_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("JWT_SECRET", "a-real-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@prod-host:5432/adv_rag")
    monkeypatch.setenv("GUARDRAIL_MODE_DEFAULT", "monitor")
    monkeypatch.setenv("CRAG_WEB_ENABLED_BY_DEFAULT", "true")

    with pytest.raises(ValueError, match="guardrail mode 'monitor'"):
        Settings()


def test_production_allows_monitor_mode_with_sql_and_crag_web_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("JWT_SECRET", "a-real-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@prod-host:5432/adv_rag")
    monkeypatch.setenv("GUARDRAIL_MODE_DEFAULT", "monitor")

    settings = Settings()
    assert settings.safety.guardrail_mode_default == "monitor"


def test_invalid_database_url_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "mysql://user:pw@localhost/db")

    with pytest.raises(ValueError, match="postgresql://"):
        Settings()


def test_invalid_storage_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORAGE_BACKEND", "azure")

    with pytest.raises(ValueError):
        Settings()
