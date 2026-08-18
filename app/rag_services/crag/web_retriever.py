"""Bounded, allowlisted regulatory-web retrieval."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import requests
from tavily import TavilyClient

from app.rag_services.crag.crag import WebEvidence

SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]

# Control/formatting characters that have no place in extracted web content:
# C0/C1 controls, zero-width spaces, and bidi-override characters that could
# otherwise be used to visually disguise text (e.g. reversing displayed
# order). Not a substitute for a future context-injection guardrail - just a
# deterministic normalization boundary applied to every web snippet before
# it can reach the refiner/answer LLM.
_CONTROL = re.compile(
    r"[\u0000-\u0008\u000b\u000c\u000e-\u001f"
    r"\u007f\u200b-\u200f\u202a-\u202e\u2066-\u2069]"
)


def _normalize_external_text(value: str, *, max_chars: int) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _CONTROL.sub("", normalized)
    normalized = " ".join(normalized.split())
    return normalized[:max_chars]


def _is_allowed(domain: str, allowlist: frozenset[str]) -> bool:
    return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in allowlist)


def _validated_canonical_url(raw_url: str, allowlist: frozenset[str]) -> tuple[str, str] | None:
    """Returns ``(canonical_url, domain)`` iff ``raw_url`` is HTTPS, carries
    no embedded credentials, and resolves to an allowlisted domain - or
    ``None`` otherwise. The canonical form drops any fragment (never needed
    for citation, and a vector for smuggling extra text past a naive
    display) and normalizes an empty path to ``/``."""
    parsed = urlsplit(raw_url)
    if parsed.scheme.casefold() != "https":
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    domain = (parsed.hostname or "").casefold().rstrip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    if not domain or not _is_allowed(domain, allowlist):
        return None
    canonical = urlunsplit(("https", parsed.netloc, parsed.path or "/", parsed.query, ""))
    return canonical, domain


class TimeoutSession(requests.Session):
    """A ``requests.Session`` that always applies a connect/read timeout.

    Tavily's SDK doesn't expose a per-call timeout parameter, so this is
    injected as its transport instead - without it, a slow or hanging
    regulatory-web endpoint could block a CRAG request indefinitely.
    """

    def __init__(self, *, connect_timeout: float, read_timeout: float) -> None:
        super().__init__()
        self._timeout = (connect_timeout, read_timeout)

    def request(self, method: str, url: str, **kwargs: object) -> requests.Response:  # type: ignore[override]
        kwargs.setdefault("timeout", self._timeout)
        return super().request(method, url, **kwargs)  # type: ignore[arg-type]


class KeywordRegulatoryScopePolicy:
    """Conservative first adapter.

    This policy is intentionally explicit. A future structured intent
    classifier can implement the same protocol after its own evaluation.
    """

    _PUBLIC_MARKERS = (
        "rbi circular",
        "reserve bank of india",
        "sebi circular",
        "npci circular",
        "public regulation",
        "regulatory circular",
    )

    def __init__(self, *, policy_version: str) -> None:
        self._policy_version = policy_version

    @property
    def cache_namespace(self) -> str:
        return f"scope-policy={self._policy_version}"

    def permits_public_regulatory_web(self, question: str) -> bool:
        normalized = " ".join(question.casefold().split())
        return any(marker in normalized for marker in self._PUBLIC_MARKERS)


class TavilyRegulatoryWebRetriever:
    def __init__(
        self,
        *,
        api_key: str,
        allowed_domains: frozenset[str],
        max_results: int,
        search_depth: SearchDepth = "advanced",
        max_content_chars: int = 6_000,
        max_total_chars: int = 20_000,
        connect_timeout_seconds: float = 3.0,
        read_timeout_seconds: float = 8.0,
    ) -> None:
        if not api_key:
            raise ValueError("Tavily API key is required for web correction")
        if not allowed_domains:
            raise ValueError("regulatory domain allowlist must not be empty")
        self._domains = frozenset(d.casefold().rstrip(".") for d in allowed_domains)
        self._max_results = max_results
        self._depth: SearchDepth = search_depth
        self._max_content_chars = max_content_chars
        self._max_total_chars = max_total_chars
        self._session = TimeoutSession(
            connect_timeout=connect_timeout_seconds,
            read_timeout=read_timeout_seconds,
        )
        self._client = TavilyClient(api_key=api_key, session=self._session)

    @property
    def cache_namespace(self) -> str:
        domains = ",".join(sorted(self._domains))
        return f"tavily:depth={self._depth}:max={self._max_results}:domains={domains}"

    def search(self, query: str) -> list[WebEvidence]:
        response = self._client.search(
            query=query,
            max_results=self._max_results,
            search_depth=self._depth,
            include_domains=sorted(self._domains),
            include_answer=False,
            include_raw_content=False,
        )
        retrieved_at = datetime.now(UTC).isoformat()
        results: list[WebEvidence] = []
        total_chars = 0
        # Never trust the provider's own max_results enforcement alone -
        # results are re-capped here, and the per-item/total character
        # budgets below bound the worst case regardless of what the
        # provider returns.
        for item in response.get("results", []):
            if len(results) >= self._max_results:
                break
            validated = _validated_canonical_url(str(item.get("url", "")), self._domains)
            if validated is None:
                continue
            canonical_url, domain = validated
            remaining = self._max_total_chars - total_chars
            if remaining <= 0:
                break
            content = _normalize_external_text(
                str(item.get("content", "")),
                max_chars=min(self._max_content_chars, remaining),
            )
            if not content:
                continue
            total_chars += len(content)
            results.append(
                WebEvidence(
                    title=_normalize_external_text(str(item.get("title", domain)), max_chars=200),
                    text=content,
                    canonical_url=canonical_url,
                    domain=domain,
                    retrieved_at_iso=retrieved_at,
                    score=float(item.get("score", 0.0)),
                )
            )
        return results

    def close(self) -> None:
        self._session.close()
