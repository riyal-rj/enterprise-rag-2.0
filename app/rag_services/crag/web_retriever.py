"""Bounded, allowlisted regulatory-web retrieval."""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import urlparse

from tavily import TavilyClient

from app.rag_services.crag.crag import WebEvidence


def _normalized_domain(url: str) -> str:
    domain = (urlparse(url).hostname or "").casefold().rstrip(".")
    return domain[4:] if domain.startswith("www.") else domain


def _is_allowed(domain: str, allowlist: frozenset[str]) -> bool:
    return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in allowlist)


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
        search_depth: str = "advanced",
    ) -> None:
        if not api_key:
            raise ValueError("Tavily API key is required for web correction")
        if not allowed_domains:
            raise ValueError("regulatory domain allowlist must not be empty")
        self._client = TavilyClient(api_key=api_key)
        self._domains = frozenset(d.casefold().rstrip(".") for d in allowed_domains)
        self._max_results = max_results
        self._depth = search_depth

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
        for item in response.get("results", []):
            url = str(item.get("url", ""))
            domain = _normalized_domain(url)
            if not _is_allowed(domain, self._domains):
                continue
            content = " ".join(str(item.get("content", "")).split())
            if not content:
                continue
            results.append(
                WebEvidence(
                    title=" ".join(str(item.get("title", domain)).split()),
                    text=content,
                    canonical_url=url,
                    domain=domain,
                    retrieved_at_iso=retrieved_at,
                    score=float(item.get("score", 0.0)),
                )
            )
        return results
