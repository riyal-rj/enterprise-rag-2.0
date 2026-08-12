from __future__ import annotations

import pytest

from app.eval import invokers
from app.eval.invokers import ServiceInvoker, SkippedIntent
from app.eval.profiles import PROFILES
from app.eval.schemas import Intent


class _FakeSecretStr:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class _FakeExternalApis:
    def __init__(self, tavily_api_key: str) -> None:
        self.tavily_api_key = _FakeSecretStr(tavily_api_key)


class _FakeSettings:
    def __init__(self, tavily_api_key: str) -> None:
        self.external_apis = _FakeExternalApis(tavily_api_key)


def test_service_invoker_raises_skipped_intent_when_not_wired() -> None:
    invoker = ServiceInvoker()

    with pytest.raises(SkippedIntent, match="not wired yet"):
        invoker.invoke("What is a pod?", PROFILES["naive"], Intent.RAG)


@pytest.mark.parametrize("intent", [Intent.SQL, Intent.HYBRID])
def test_service_invoker_skips_unsupported_intents(intent: Intent) -> None:
    """SQL/hybrid need human-in-the-loop approval — not runnable headlessly."""
    invoker = ServiceInvoker()

    with pytest.raises(SkippedIntent, match="not supported in service mode"):
        invoker.invoke("question", PROFILES["naive"], intent)


def test_service_invoker_skips_web_fallback_without_tavily_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(invokers, "get_settings", lambda: _FakeSettings(tavily_api_key=""))
    invoker = ServiceInvoker()

    with pytest.raises(SkippedIntent, match="tavily_unset"):
        invoker.invoke("question", PROFILES["naive"], Intent.WEB_FALLBACK)


def test_service_invoker_reaches_pipeline_seam_when_tavily_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passes both guards, then hits the same not-wired-yet seam as RAG."""
    monkeypatch.setattr(invokers, "get_settings", lambda: _FakeSettings(tavily_api_key="tvly-test"))
    invoker = ServiceInvoker()

    with pytest.raises(SkippedIntent, match="not wired yet"):
        invoker.invoke("question", PROFILES["naive"], Intent.WEB_FALLBACK)
