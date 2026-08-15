"""Post-generation check for absolute-claim language that isn't traceable to context.

Catches the generalization failure mode from the evaluation report (Q07): a
clause restricting one specific action ("cannot process outgoing debits") gets
rewritten by the model into a broader, unsupported claim ("cannot send or
receive transactions"). This is a lexical-overlap heuristic over sentences
using absolute-claim language, not an entailment check - same caveat as
``confidence_scorer._faithfulness``. It flags a claim asserted in vocabulary
the context never used; it cannot catch a fluent misuse of the *right*
vocabulary to assert the wrong thing.
"""

from __future__ import annotations

import re

from app.models.retrieved_chunk import RetrievedChunk

_ABSOLUTE_CLAIM_PATTERN = re.compile(
    r"\b(cannot|can not|can't|must|prohibited|forbidden|never|always|"
    r"required to|not allowed|not permitted)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

# Below this fraction of a claim sentence's content words appearing in the
# retrieved context, the claim is treated as not traceable to the source
# text and gets flagged.
_OVERLAP_FLOOR = 0.4


def find_unsupported_claims(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    """Sentences with absolute-claim language whose wording isn't grounded in ``chunks``."""
    context_words = _content_words(" ".join(c.text for c in chunks))
    flagged: list[str] = []
    for sentence in _sentences(answer):
        if not _ABSOLUTE_CLAIM_PATTERN.search(sentence):
            continue
        sentence_words = _content_words(sentence)
        if not sentence_words:
            continue
        overlap = sum(1 for word in sentence_words if word in context_words) / len(
            sentence_words
        )
        if overlap < _OVERLAP_FLOOR:
            flagged.append(sentence.strip())
    return flagged


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_PATTERN.split(text.strip()) if s]


def _content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2}
