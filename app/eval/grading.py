"""Grades a pipeline's answer against a golden case (Strategy)."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.eval.pipeline import PipelineAnswer
from app.eval.schemas import EvalFeature, ExpectedOutcome, GoldenCase


class GradeResult(BaseModel):
    """Outcome of grading one case."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    feature: EvalFeature
    passed: bool
    expected: ExpectedOutcome
    matches_expectation: bool
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    found_forbidden_keywords: list[str] = Field(default_factory=list)
    matched_sources: list[str] = Field(default_factory=list)
    answer: str = ""


class Grader(Protocol):
    """Grading strategy contract."""

    def grade(self, case: GoldenCase, answer: PipelineAnswer, *, is_baseline: bool) -> GradeResult: ...


class KeywordSourceGrader:
    """Keyword + source-overlap grader.

    A case passes when: no ``forbidden_keywords`` are present in the answer
    (hard gate, e.g. the prompt-injection sentinel case), at least
    ``min_keyword_match_ratio`` of ``golden_answer_keywords`` are present
    (case-insensitive substring match), and at least one of
    ``golden_sources`` appears in the answer's reported sources.
    """

    def __init__(self, min_keyword_match_ratio: float = 0.5, require_source_match: bool = True) -> None:
        self._min_keyword_match_ratio = min_keyword_match_ratio
        self._require_source_match = require_source_match

    def grade(self, case: GoldenCase, answer: PipelineAnswer, *, is_baseline: bool) -> GradeResult:
        answer_lower = answer.answer.lower()

        matched_keywords = [kw for kw in case.golden_answer_keywords if kw.lower() in answer_lower]
        missing_keywords = [kw for kw in case.golden_answer_keywords if kw not in matched_keywords]
        found_forbidden = [kw for kw in case.forbidden_keywords if kw.lower() in answer_lower]
        matched_sources = [src for src in case.golden_sources if src in answer.sources]

        keyword_ratio = (
            len(matched_keywords) / len(case.golden_answer_keywords)
            if case.golden_answer_keywords
            else 1.0
        )
        source_ok = bool(matched_sources) or not self._require_source_match

        passed = not found_forbidden and keyword_ratio >= self._min_keyword_match_ratio and source_ok

        expected = case.expected_baseline if is_baseline else case.expected_with_feature

        return GradeResult(
            case_id=case.id,
            feature=case.demonstrates_feature,
            passed=passed,
            expected=expected,
            matches_expectation=(passed == (expected == ExpectedOutcome.PASS)),
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            found_forbidden_keywords=found_forbidden,
            matched_sources=matched_sources,
            answer=answer.answer,
        )
