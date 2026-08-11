from app.eval.grading import KeywordSourceGrader
from app.eval.pipeline import PipelineAnswer
from app.eval.runner import EvalRunner
from app.eval.schemas import EvalFeature, ExpectedOutcome, GoldenCase, Intent


class _FakePipeline:
    """Test double for :class:`~app.eval.pipeline.QueryPipeline`."""

    def __init__(self, canned: dict[str, PipelineAnswer]) -> None:
        self._canned = canned

    async def answer(self, question: str) -> PipelineAnswer:
        return self._canned[question]


def _case(case_id: str, question: str, feature: EvalFeature) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        question=question,
        intent=Intent.RAG,
        golden_sources=["pods.html"],
        golden_answer_keywords=["pod"],
        demonstrates_feature=feature,
        expected_baseline=ExpectedOutcome.PASS,
        expected_with_feature="pass",
        notes="test fixture",
    )


async def test_runner_produces_report_with_feature_breakdown() -> None:
    case = _case("q-991", "What is a Pod?", EvalFeature.BASELINE)
    pipeline = _FakePipeline(
        {"What is a Pod?": PipelineAnswer(answer="A pod.", sources=["pods.html"])}
    )
    runner = EvalRunner(pipeline, KeywordSourceGrader(), is_baseline=True)

    report = await runner.run("naive", [case])

    assert report.total == 1
    assert report.passed == 1
    assert report.by_feature["baseline"].passed == 1
    assert report.by_feature["baseline"].total == 1


async def test_runner_aggregates_across_mixed_pass_fail() -> None:
    cases = [
        _case("q-992", "What is a Pod?", EvalFeature.BASELINE),
        _case("q-993", "What is a Service?", EvalFeature.HYBRID),
    ]
    pipeline = _FakePipeline(
        {
            "What is a Pod?": PipelineAnswer(answer="A pod.", sources=["pods.html"]),
            "What is a Service?": PipelineAnswer(answer="No idea.", sources=[]),
        }
    )
    runner = EvalRunner(pipeline, KeywordSourceGrader(), is_baseline=False)

    report = await runner.run("all", cases)

    assert report.total == 2
    assert report.passed == 1
    assert report.pass_rate == 0.5
    assert len(report.mismatched_expectations) == 1
    assert report.mismatched_expectations[0].case_id == "q-993"
