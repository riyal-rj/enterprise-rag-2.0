"""Tests for the Tavily-backed regulatory web retriever."""

from __future__ import annotations

import requests

from app.rag_services.crag.web_retriever import (
    KeywordRegulatoryScopePolicy,
    TavilyRegulatoryWebRetriever,
    TimeoutSession,
    _normalize_external_text,
    _validated_canonical_url,
)


def test_search_depth_is_passed_to_tavily() -> None:
    retriever = TavilyRegulatoryWebRetriever(
        api_key="test",
        allowed_domains=frozenset({"rbi.org.in"}),
        max_results=3,
        search_depth="advanced",
    )
    assert retriever.cache_namespace.startswith("tavily:depth=advanced")


def test_timeout_session_overrides_a_default_timeout_the_caller_already_set(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """Regression: the Tavily SDK explicitly passes its own timeout=60 on
    every call - kwargs.setdefault("timeout", ...) is therefore a no-op
    (the key is already present) and connect_timeout_seconds/
    read_timeout_seconds would silently never apply. The configured tuple
    must win regardless of what the caller already put in kwargs."""
    captured: dict[str, object] = {}

    def fake_request(self: requests.Session, method: str, url: str, **kwargs: object) -> str:
        captured.update(kwargs)
        return "response"

    monkeypatch.setattr(requests.Session, "request", fake_request)

    session = TimeoutSession(connect_timeout=3.0, read_timeout=8.0)
    session.request("POST", "https://api.tavily.com/search", timeout=60)

    assert captured["timeout"] == (3.0, 8.0)


def _retriever(**overrides: object) -> TavilyRegulatoryWebRetriever:
    defaults: dict[str, object] = dict(
        api_key="test",
        allowed_domains=frozenset({"rbi.org.in"}),
        max_results=5,
        max_content_chars=6_000,
        max_total_chars=20_000,
    )
    defaults.update(overrides)
    return TavilyRegulatoryWebRetriever(**defaults)  # type: ignore[arg-type]


class _StubClient:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self._results = results

    def search(self, **kwargs: object) -> dict[str, object]:
        return {"results": self._results}


def _item(
    url: str, content: str = "some content", title: str = "title", score: float = 0.5
) -> dict[str, object]:
    return {"url": url, "content": content, "title": title, "score": score}


# --- _validated_canonical_url -----------------------------------------


def test_validated_canonical_url_accepts_https_allowlisted_domain() -> None:
    result = _validated_canonical_url("https://rbi.org.in/circular/1", frozenset({"rbi.org.in"}))

    assert result == ("https://rbi.org.in/circular/1", "rbi.org.in")


def test_validated_canonical_url_rejects_http() -> None:
    assert (
        _validated_canonical_url("http://rbi.org.in/circular/1", frozenset({"rbi.org.in"})) is None
    )


def test_validated_canonical_url_rejects_embedded_credentials() -> None:
    assert (
        _validated_canonical_url("https://user:pass@rbi.org.in/x", frozenset({"rbi.org.in"}))
        is None
    )


def test_validated_canonical_url_rejects_non_allowlisted_domain() -> None:
    assert _validated_canonical_url("https://evil.example.com/x", frozenset({"rbi.org.in"})) is None


def test_validated_canonical_url_accepts_subdomain_of_allowlisted_domain() -> None:
    result = _validated_canonical_url(
        "https://notifications.rbi.org.in/x", frozenset({"rbi.org.in"})
    )

    assert result is not None
    assert result[1] == "notifications.rbi.org.in"


def test_validated_canonical_url_drops_fragment_and_normalizes_empty_path() -> None:
    result = _validated_canonical_url("https://rbi.org.in#section-2", frozenset({"rbi.org.in"}))

    assert result == ("https://rbi.org.in/", "rbi.org.in")


# --- _normalize_external_text -------------------------------------------


def test_normalize_external_text_strips_control_and_zero_width_characters() -> None:
    dirty = "hello​world‪override"
    assert _normalize_external_text(dirty, max_chars=100) == "helloworldoverride"


def test_normalize_external_text_collapses_whitespace() -> None:
    assert _normalize_external_text("a   b\n\nc", max_chars=100) == "a b c"


def test_normalize_external_text_truncates_to_max_chars() -> None:
    assert _normalize_external_text("abcdef", max_chars=3) == "abc"


# --- search() budgets and validation -------------------------------------


def test_search_drops_results_from_non_allowlisted_domains() -> None:
    retriever = _retriever()
    retriever._client = _StubClient(  # type: ignore[attr-defined]
        [_item("https://evil.example.com/a"), _item("https://rbi.org.in/b")]
    )

    results = retriever.search("q")

    assert len(results) == 1
    assert results[0].domain == "rbi.org.in"


def test_search_drops_non_https_results() -> None:
    retriever = _retriever()
    retriever._client = _StubClient([_item("http://rbi.org.in/a")])  # type: ignore[attr-defined]

    assert retriever.search("q") == []


def test_search_caps_results_at_max_results_even_if_provider_returns_more() -> None:
    retriever = _retriever(max_results=2)
    retriever._client = _StubClient(  # type: ignore[attr-defined]
        [_item(f"https://rbi.org.in/{i}") for i in range(5)]
    )

    results = retriever.search("q")

    assert len(results) == 2


def test_search_enforces_per_item_content_budget() -> None:
    retriever = _retriever(max_content_chars=5)
    retriever._client = _StubClient(  # type: ignore[attr-defined]
        [_item("https://rbi.org.in/a", content="0123456789")]
    )

    results = retriever.search("q")

    assert results[0].text == "01234"


def test_search_enforces_total_content_budget_across_items() -> None:
    retriever = _retriever(max_content_chars=100, max_total_chars=8)
    retriever._client = _StubClient(  # type: ignore[attr-defined]
        [
            _item("https://rbi.org.in/a", content="12345"),
            _item("https://rbi.org.in/b", content="67890"),
        ]
    )

    results = retriever.search("q")

    total_chars = sum(len(r.text) for r in results)
    assert total_chars <= 8


def test_search_skips_items_with_empty_content_after_normalization() -> None:
    retriever = _retriever()
    retriever._client = _StubClient([_item("https://rbi.org.in/a", content="​​")])  # type: ignore[attr-defined]

    assert retriever.search("q") == []


def test_search_canonical_url_is_used_not_raw_provider_url() -> None:
    retriever = _retriever()
    retriever._client = _StubClient(  # type: ignore[attr-defined]
        [_item("https://rbi.org.in/x?y=1#frag")]
    )

    results = retriever.search("q")

    assert results[0].canonical_url == "https://rbi.org.in/x?y=1"


def test_construction_rejects_empty_api_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="API key"):
        TavilyRegulatoryWebRetriever(
            api_key="", allowed_domains=frozenset({"rbi.org.in"}), max_results=3
        )


def test_construction_rejects_empty_allowlist() -> None:
    import pytest

    with pytest.raises(ValueError, match="allowlist"):
        TavilyRegulatoryWebRetriever(api_key="test", allowed_domains=frozenset(), max_results=3)


def test_close_closes_the_underlying_session() -> None:
    retriever = _retriever()

    retriever.close()  # must not raise


# --- KeywordRegulatoryScopePolicy ----------------------------------------


def test_scope_policy_cache_namespace_encodes_policy_version() -> None:
    policy = KeywordRegulatoryScopePolicy(policy_version="crag-policy-v2")

    assert policy.cache_namespace == "scope-policy=crag-policy-v2"


def test_scope_policy_permits_known_regulatory_markers() -> None:
    policy = KeywordRegulatoryScopePolicy(policy_version="v1")

    assert policy.permits_public_regulatory_web("What does the latest RBI circular say?") is True
    assert (
        policy.permits_public_regulatory_web("What is our internal wire transfer limit?") is False
    )
