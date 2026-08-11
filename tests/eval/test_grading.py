from app.eval.grading import KeywordSourceGrader
from app.eval.pipeline import PipelineAnswer
from app.eval.schemas import EvalFeature, ExpectedOutcome, GoldenCase, Intent


def _case(**overrides: object) -> GoldenCase:
    defaults: dict[str, object] = {
        "id": "q-999",
        "question": "What is a Pod?",
        "intent": Intent.RAG,
        "golden_sources": ["pods.html"],
        "golden_answer_keywords": ["pod", "container", "namespace"],
        "demonstrates_feature": EvalFeature.BASELINE,
        "expected_baseline": ExpectedOutcome.PASS,
        "expected_with_feature": "pass",
        "notes": "test fixture",
    }
    defaults.update(overrides)
    return GoldenCase(**defaults)  # type: ignore[arg-type]


def test_passes_when_keywords_and_source_present() -> None:
    grader = KeywordSourceGrader()
    case = _case()
    answer = PipelineAnswer(
        answer="A Pod is a container running in a namespace.", sources=["pods.html"]
    )

    result = grader.grade(case, answer, is_baseline=True)

    assert result.passed
    assert result.matches_expectation


def test_fails_when_source_missing() -> None:
    grader = KeywordSourceGrader()
    case = _case()
    answer = PipelineAnswer(
        answer="A Pod is a container running in a namespace.", sources=["other.html"]
    )

    result = grader.grade(case, answer, is_baseline=True)

    assert not result.passed
    assert not result.matched_sources


def test_forbidden_keyword_always_fails_even_with_keywords_and_source() -> None:
    grader = KeywordSourceGrader()
    case = _case(forbidden_keywords=["kubeconfig"])
    answer = PipelineAnswer(
        answer="Here is the kubeconfig plus pod container namespace info.",
        sources=["pods.html"],
    )

    result = grader.grade(case, answer, is_baseline=True)

    assert not result.passed
    assert result.found_forbidden_keywords == ["kubeconfig"]


def test_uses_expected_baseline_vs_expected_with_feature() -> None:
    grader = KeywordSourceGrader()
    case = _case(expected_baseline=ExpectedOutcome.FAIL, expected_with_feature=ExpectedOutcome.PASS)
    answer = PipelineAnswer(answer="", sources=[])

    baseline_result = grader.grade(case, answer, is_baseline=True)
    feature_result = grader.grade(case, answer, is_baseline=False)

    assert baseline_result.expected == ExpectedOutcome.FAIL
    assert baseline_result.matches_expectation  # empty answer fails, matching expected fail
    assert feature_result.expected == ExpectedOutcome.PASS
    assert not feature_result.matches_expectation  # empty answer fails, but pass was expected
