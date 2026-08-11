"""Small, independent post-hoc checks run against each row after invocation.

Kept separate from the RAGAS metrics (``ragas_adapter.run``): these are
deterministic, don't need an LLM judge, and answer different questions —
"did the answer leak something it shouldn't" and "did retrieval actually
find the right document", not "is the answer any good".
"""

from __future__ import annotations


def forbidden_keywords_check(answer: str, 
                             forbidden_keywords: list[str]) -> dict[str, object]:
    """Hard security/safety gate: flags any forbidden keyword found in the answer."""
    answer_lower = answer.lower()
    found = [kw for kw in forbidden_keywords if kw.lower() in answer_lower]
    return {"passed": not found, "found": found}


def source_overlap(actual_sources: list[str], 
                   golden_sources: list[str]) -> dict[str, object]:
    """Fraction of ``golden_sources`` present in ``actual_sources``.

    A case with no ``golden_sources`` (shouldn't happen — the schema
    requires at least one) trivially passes rather than dividing by zero.
    """
    matched = [source for source in golden_sources if source in actual_sources]
    ratio = (len(matched) / len(golden_sources)) if golden_sources else 1.0
    return {"ratio": ratio, "matched": matched, "passed": bool(matched)}
