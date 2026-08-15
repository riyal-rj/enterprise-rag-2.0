"""Small, independent post-hoc checks run against each row after invocation.

Kept separate from the RAGAS metrics (``ragas_adapter.run``): these are
deterministic, don't need an LLM judge, and answer different questions —
"did the answer leak something it shouldn't" and "did retrieval actually
find the right document", not "is the answer any good".
"""

from __future__ import annotations

from app.eval.types import ForbiddenCheckResult, RetrievalMetrics, SourceOverlapResult


def forbidden_keywords_check(answer: str, forbidden_keywords: list[str]) -> ForbiddenCheckResult:
    """Hard security/safety gate: flags any forbidden keyword found in the answer."""
    answer_lower = answer.lower()
    found = [kw for kw in forbidden_keywords if kw.lower() in answer_lower]
    return {"passed": not found, "found": found}


def source_overlap(actual_sources: list[str], golden_sources: list[str]) -> SourceOverlapResult:
    """Fraction of ``golden_sources`` present in ``actual_sources``.

    A case with no ``golden_sources`` (shouldn't happen — the schema
    requires at least one) trivially passes rather than dividing by zero.
    """
    matched = [source for source in golden_sources if source in actual_sources]
    ratio = (len(matched) / len(golden_sources)) if golden_sources else 1.0
    return {"ratio": ratio, "matched": matched, "passed": bool(matched)}


def retrieval_metrics(ranked_sources: list[str], golden_sources: list[str]) -> RetrievalMetrics:
    """Deterministic, rank-aware retrieval quality for one case.

    ``ranked_sources`` must be in actual retrieval order (rank 1 first) -
    unlike ``actual_sources``/``ChatResponse.sources``, which is an
    alphabetically sorted set with rank information already discarded.

    - ``hit_rate``: whether *any* golden source was retrieved at all
      (Recall@K collapses to this at K = len(ranked_sources); no separate
      function needed for that).
    - ``mrr``: 1 / (1-indexed rank of the first golden source found), or
      ``0.0`` if none were retrieved.
    """
    for rank, source in enumerate(ranked_sources, start=1):
        if source in golden_sources:
            return {"hit_rate": True, "mrr": 1.0 / rank}
    return {"hit_rate": False, "mrr": 0.0}
