"""Composite confidence scoring for generated RAG answers.

Replaces the flat ``confidence=0.7`` literal that used to live in
``RAGService.answer()`` with a weighted, explainable score computed from
signals already available post-retrieval/post-generation - no extra LLM
call, no reranker, no eval harness required (see RAG-102).

Weights mirror the illustrative model from the evaluation report, Section
7: 30% evidence coverage + 25% faithfulness + 20% retrieval
strength/margin + 15% citation precision + 10% answerability. Two of
those five signals are cheap stand-ins for pieces that don't exist yet
in this pipeline:

- Faithfulness here is a lexical-overlap proxy against the retrieved
  context, not a real entailment/NLI check. It cannot catch an answer
  that reuses the right words but asserts the wrong claim (a scope or
  negation error) - only a judge-based check can. Swap in one once
  RAG-111's RAGAS harness is running.
- Retrieval strength is raw dense cosine similarity. Swap in the
  reranker's relevance score once RAG-104 lands - it's a stronger
  relevance signal than dense similarity alone.

Both swaps are additive: keep the same weighting scheme, replace one
component's inputs. Calibration (Brier score / ECE against a golden
set) is deliberately not attempted here - that requires RAG-111's
labeled evaluation set and is tracked separately.

``retrieval_strength``/``evidence_coverage`` further assume ``chunk.score``
is a dense cosine similarity, calibrated so ``_RELEVANCE_FLOOR`` means
something. That assumption breaks for hybrid: RRF fusion produces a
rank-derived score (see
``app.rag_services.retrieval_strategy.HybridRetrievalStrategy``'s
``max_rrf_score`` normalization) - a chunk ranked first by only one of
the two branches lands at ~0.5 regardless of its actual relevance, so
comparing it against a cosine-calibrated floor as if the two were on the
same scale would misjudge it. Callers pass ``retrieval_mode`` so these
two components can fall back to a neutral, non-cosine-scale treatment
for hybrid/sparse instead of silently misapplying the dense calibration
(see ``_evidence_coverage``/``_retrieval_strength``). This is a stopgap,
not a real hybrid calibration - that also needs RAG-111's labeled set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.retrieved_chunk import RetrievedChunk

_COVERAGE_WEIGHT = 0.30
_FAITHFULNESS_WEIGHT = 0.25
_RETRIEVAL_STRENGTH_WEIGHT = 0.20
_CITATION_PRECISION_WEIGHT = 0.15
_ANSWERABILITY_WEIGHT = 0.10

_WEIGHTS_SUM = (
    _COVERAGE_WEIGHT
    + _FAITHFULNESS_WEIGHT
    + _RETRIEVAL_STRENGTH_WEIGHT
    + _CITATION_PRECISION_WEIGHT
    + _ANSWERABILITY_WEIGHT
)
assert abs(_WEIGHTS_SUM - 1.0) < 1e-9, "confidence component weights must sum to 1.0"

# Below this dense cosine score a chunk is treated as noise, not
# supporting evidence, for coverage purposes. Matches the existing
# ``crag_ambiguous_threshold`` in ``app.core.config.rag_features`` - both
# describe the same "is this dense score even relevant" cutoff for the
# same embedding space.
_RELEVANCE_FLOOR = 0.5
# Coverage saturates once this many chunks clear the relevance floor - a
# 4th or 5th corroborating chunk doesn't make an answer meaningfully more
# trustworthy than 3 already did.
_COVERAGE_SATURATION_COUNT = 3

_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
# Splits the inside of a single bracket on commas/semicolons so
# "[a.pdf, b.pdf]" is treated the same as "[a.pdf][b.pdf]".
_CITATION_SEPARATOR_PATTERN = re.compile(r"[,;]")

_REFUSAL_PHRASES = (
    "does not contain",
    "doesn't contain",
    "no relevant context",
    "not contain the answer",
    "don't know",
    "do not know",
    "cannot determine",
    "unable to determine",
    "no information",
    "not provided in the context",
)
# A short answer built mostly around a refusal phrase is a genuine "the
# model couldn't answer". The same phrase surfacing once inside an
# otherwise substantive answer (a caveat, not the whole answer) is a much
# weaker signal and shouldn't be scored the same way.
_REFUSAL_SHORT_ANSWER_WORD_LIMIT = 25
_REFUSAL_SCORE = 0.3
_REFUSAL_CAVEAT_SCORE = 0.7


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Per-component scores plus the weighted total.

    Returned (not just the float total) so callers can log the
    components for observability - "why was this answer scored low" is
    a debugging question production needs an answer to - and so future
    calibration work (RAG-111) can consume the components directly
    instead of re-deriving them.
    """

    evidence_coverage: float
    faithfulness: float
    retrieval_strength: float
    citation_precision: float
    answerability: float
    total: float


def compute_confidence(chunks: list[RetrievedChunk], answer: str, *, retrieval_mode: str = "dense") -> float:
    """Composite, explainable confidence in ``[0.0, 1.0]``."""
    return compute_confidence_breakdown(chunks, answer, retrieval_mode=retrieval_mode).total


def compute_confidence_breakdown(
    chunks: list[RetrievedChunk], answer: str, *, retrieval_mode: str = "dense"
) -> ConfidenceBreakdown:
    """Same score as :func:`compute_confidence`, with each component exposed.

    Each component is independently a ``[0.0, 1.0]`` score; the weighted
    sum is a probability-flavored estimate, not a calibrated probability.

    ``retrieval_mode`` should be the retrieval strategy's ``name`` (e.g.
    ``"dense"``, ``"hybrid"``, ``"sparse"``) - only ``"dense"`` scores are
    on a cosine-similarity scale ``_RELEVANCE_FLOOR`` was calibrated for
    (see the module docstring). Any other mode uses a neutral treatment
    for the two components that assume that scale.
    """
    coverage = _evidence_coverage(chunks, retrieval_mode)
    faithfulness = _faithfulness(chunks, answer)
    retrieval_strength = _retrieval_strength(chunks, retrieval_mode)
    citation_precision = _citation_precision(chunks, answer)
    answerability = _answerability(chunks, answer)

    total = (
        _COVERAGE_WEIGHT * coverage
        + _FAITHFULNESS_WEIGHT * faithfulness
        + _RETRIEVAL_STRENGTH_WEIGHT * retrieval_strength
        + _CITATION_PRECISION_WEIGHT * citation_precision
        + _ANSWERABILITY_WEIGHT * answerability
    )
    return ConfidenceBreakdown(
        evidence_coverage=coverage,
        faithfulness=faithfulness,
        retrieval_strength=retrieval_strength,
        citation_precision=citation_precision,
        answerability=answerability,
        total=max(0.0, min(1.0, total)),
    )


def _evidence_coverage(chunks: list[RetrievedChunk], retrieval_mode: str) -> float:
    """How much independent, above-floor evidence backs the answer."""
    if retrieval_mode == "dense":
        supporting = [c for c in chunks if c.score >= _RELEVANCE_FLOOR]
    else:
        # Non-dense scores aren't on the cosine scale _RELEVANCE_FLOOR was
        # calibrated for (see module docstring) - treat every returned
        # chunk as supporting evidence instead of gating on a magnitude
        # comparison that isn't meaningful for a rank-derived score.
        supporting = list(chunks)
    if not supporting:
        return 0.0
    count_score = min(len(supporting) / _COVERAGE_SATURATION_COUNT, 1.0)
    diversity_score = (
        min(len({c.source for c in supporting}), _COVERAGE_SATURATION_COUNT)
        / _COVERAGE_SATURATION_COUNT
    )
    return (count_score + diversity_score) / 2


def _retrieval_strength(chunks: list[RetrievedChunk], retrieval_mode: str) -> float:
    """How strong and how clearly-dominant the top match is."""
    if not chunks:
        return 0.0
    if retrieval_mode != "dense":
        # No calibrated absolute-relevance signal exists yet for a
        # rank-derived (hybrid) or lexical (sparse) score - stay neutral
        # rather than feeding an uncalibrated score through cosine-scale
        # math (see module docstring; real calibration needs RAG-111's
        # labeled evaluation set).
        return 0.5
    top_score = max(0.0, min(1.0, chunks[0].score))
    rest = chunks[1:]
    if not rest:
        return top_score
    mean_rest = sum(max(0.0, min(1.0, c.score)) for c in rest) / len(rest)
    margin = max(0.0, top_score - mean_rest)
    return max(0.0, min(1.0, 0.7 * top_score + 0.3 * margin))


def _faithfulness(chunks: list[RetrievedChunk], answer: str) -> float:
    """Lexical-overlap proxy for whether the answer sticks to the context."""
    context_words = _content_words(" ".join(c.text for c in chunks))
    answer_words = _content_words(answer)
    if not context_words or not answer_words:
        return 0.0
    grounded = sum(1 for word in answer_words if word in context_words)
    return grounded / len(answer_words)


def _citation_precision(chunks: list[RetrievedChunk], answer: str) -> float:
    """Fraction of bracket-cited sources that were actually retrieved."""
    available_sources = {c.source for c in chunks}
    if not available_sources:
        return 1.0  # nothing was retrieved to cite in the first place
    cited = _extract_cited_sources(answer)
    if not cited:
        return 0.5  # evidence existed but the answer didn't cite any of it
    correct = sum(1 for source in cited if source in available_sources)
    return correct / len(cited)


def _answerability(chunks: list[RetrievedChunk], answer: str) -> float:
    """Whether the model appears to have had enough to actually answer."""
    if not chunks:
        return 0.0
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    answer_lower = answer.lower()
    if not any(phrase in answer_lower for phrase in _REFUSAL_PHRASES):
        return 1.0
    is_short_answer = len(answer_words) <= _REFUSAL_SHORT_ANSWER_WORD_LIMIT
    return _REFUSAL_SCORE if is_short_answer else _REFUSAL_CAVEAT_SCORE


def _extract_cited_sources(answer: str) -> set[str]:
    """Bracket contents, split on commas/semicolons for multi-source brackets.

    Handles both ``[a.pdf][b.pdf]`` and ``[a.pdf, b.pdf]`` - the system
    prompt only shows the single-citation form, so the model is free to
    use either.
    """
    cited: set[str] = set()
    for bracket_content in _CITATION_PATTERN.findall(answer):
        for part in _CITATION_SEPARATOR_PATTERN.split(bracket_content):
            stripped = part.strip()
            if stripped:
                cited.add(stripped)
    return cited


def _content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2}
