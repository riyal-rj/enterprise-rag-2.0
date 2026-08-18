"""Fixture-backed :class:`WebRetriever` for offline CRAG web-correction eval.

CI must never depend on a live Tavily call - live web results are
non-deterministic (ranking, content, availability) and would make eval runs
flaky and unreproducible. This adapter satisfies the same
``app.rag_services.crag.crag.WebRetriever`` protocol as
``TavilyRegulatoryWebRetriever`` but always returns pre-recorded, approved
fixtures keyed by normalized query text - see ``data/crag_web_responses.json``.

Not yet wired into any :mod:`app.eval.profiles` entry - every current
profile's CRAG stage runs with ``allow_web=False`` (see
``StaticPlannedCorrectiveRetriever.plan`` and ``app.api.deps.get_eval_crag``).
This module exists so a future web-correction eval suite has a
deterministic retriever to associate golden cases with, without depending on
live Tavily traffic.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.rag_services.crag.crag import WebEvidence

_DEFAULT_FIXTURES_PATH = Path(__file__).parent / "data" / "crag_web_responses.json"


def _normalize(query: str) -> str:
    return " ".join(query.casefold().split())


class RecordedWebRetriever:
    """Returns immutable, approved fixture results for a known query.

    Unknown queries return an empty list rather than raising - matching how
    a real search provider behaves when nothing relevant is found, and
    keeping this adapter fail-safe for eval cases that don't need web
    correction fixtures at all.
    """

    def __init__(self, *, fixtures_path: Path | None = None) -> None:
        path = fixtures_path or _DEFAULT_FIXTURES_PATH
        raw = json.loads(path.read_text()) if path.exists() else {}
        self._fixtures: dict[str, tuple[WebEvidence, ...]] = {
            _normalize(query): tuple(
                WebEvidence(
                    title=item["title"],
                    text=item["text"],
                    canonical_url=item["canonical_url"],
                    domain=item["domain"],
                    retrieved_at_iso=item["retrieved_at_iso"],
                    score=item["score"],
                )
                for item in results
            )
            for query, results in raw.items()
        }

    @property
    def cache_namespace(self) -> str:
        return f"recorded:fixtures={len(self._fixtures)}"

    def search(self, query: str) -> list[WebEvidence]:
        return list(self._fixtures.get(_normalize(query), ()))
