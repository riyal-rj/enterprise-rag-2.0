"""Runs golden cases through a pipeline and grades the results."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from app.eval.grading import GradeResult, Grader
from app.eval.pipeline import QueryPipeline
from app.eval.schemas import GoldenCase


class FeatureBreakdown(BaseModel):
    total: int
    passed: int

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0


class EvalReport(BaseModel):
    """Aggregate results for one profile run over a set of golden cases."""

    model_config = ConfigDict(frozen=True)

    profile: str
    total: int
    passed: int
    results: list[GradeResult] = Field(default_factory=list)
    by_feature: dict[str, FeatureBreakdown] = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    @property
    def mismatched_expectations(self) -> list[GradeResult]:
        return [result for result in self.results if not result.matches_expectation]


class EvalRunner:
    """Runs every :class:`GoldenCase` through a :class:`QueryPipeline` and
    grades each with a :class:`Grader`, bounded by ``max_concurrency`` (real
    pipelines call rate-limited LLM APIs)."""

    def __init__(
        self,
        pipeline: QueryPipeline,
        grader: Grader,
        *,
        is_baseline: bool,
        max_concurrency: int = 5,
    ) -> None:
        self._pipeline = pipeline
        self._grader = grader
        self._is_baseline = is_baseline
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self, profile_name: str, cases: list[GoldenCase]) -> EvalReport:
        results = list(await asyncio.gather(*(self._run_one(case) for case in cases)))
        return self._build_report(profile_name, results)

    async def _run_one(self, case: GoldenCase) -> GradeResult:
        async with self._semaphore:
            answer = await self._pipeline.answer(case.question)
        return self._grader.grade(case, answer, is_baseline=self._is_baseline)

    def _build_report(self, profile_name: str, results: list[GradeResult]) -> EvalReport:
        by_feature: dict[str, list[GradeResult]] = {}
        for result in results:
            by_feature.setdefault(result.feature.value, []).append(result)

        feature_breakdown = {
            feature: FeatureBreakdown(
                total=len(feature_results),
                passed=sum(1 for result in feature_results if result.passed),
            )
            for feature, feature_results in by_feature.items()
        }

        return EvalReport(
            profile=profile_name,
            total=len(results),
            passed=sum(1 for result in results if result.passed),
            results=results,
            by_feature=feature_breakdown,
        )
