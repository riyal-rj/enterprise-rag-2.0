"""Data model for a golden eval case (see ``app/eval/data/goldens.yaml``)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    """What kind of query this case represents."""

    RAG = "rag"
    SQL = "sql"
    HYBRID = "hybrid"
    WEB_FALLBACK = "web_fallback"


class EvalFeature(str, Enum):
    """Which pipeline feature this case is designed to exercise.

    Doubles as the ``--filter`` value for ``app.eval.run_ragas``. ``SPARSE``,
    ``DENSE``, and ``SECURITY`` currently have no entries in
    ``goldens.yaml`` — they're demo-only categories, intentionally excluded
    from the eval-progression set (see ``loader.load_golden_cases``'s
    coverage warning).
    """

    BASELINE = "baseline"
    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"
    RERANK = "rerank"
    HYDE = "hyde"
    CRAG = "crag"
    SELF_RAG = "self_rag"
    SQL = "sql"
    HYBRID_RAG_SQL = "hybrid_rag_sql"
    SECURITY = "security"
    WILD = "wild"


class ExpectedOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class GoldenCase(BaseModel):
    """One curated question with its expected sources, keywords, and outcomes.

    ``expected_with_feature`` is typed ``Literal["pass"]`` rather than
    ``ExpectedOutcome``: every golden case exists to prove a feature *fixes*
    or *preserves* a passing case, never that a feature is expected to make
    a case fail. The type itself enforces that — no separate validator
    needed to reject "fail" here.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^q-\d{3}$")
    question: str = Field(min_length=1)
    intent: Intent
    golden_sources: list[str] = Field(min_length=1)
    golden_answer_keywords: list[str] = Field(min_length=1)
    demonstrates_feature: EvalFeature
    expected_baseline: ExpectedOutcome
    expected_with_feature: Literal["pass"]
    notes: str
    forbidden_keywords: list[str] = Field(default_factory=list)
