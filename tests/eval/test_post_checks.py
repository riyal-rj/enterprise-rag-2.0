import pytest

from app.eval.post_checks import forbidden_keywords_check, retrieval_metrics, source_overlap


def test_forbidden_keywords_check_passes_when_none_found() -> None:
    result = forbidden_keywords_check("A pod is a container.", ["kubeconfig"])
    assert result == {"passed": True, "found": []}


def test_forbidden_keywords_check_fails_when_found_case_insensitive() -> None:
    result = forbidden_keywords_check("Here is the KUBECONFIG.", ["kubeconfig"])
    assert result["passed"] is False
    assert result["found"] == ["kubeconfig"]


def test_source_overlap_full_match() -> None:
    result = source_overlap(["pods.html"], ["pods.html"])
    assert result["ratio"] == 1.0
    assert result["passed"] is True


def test_source_overlap_no_match() -> None:
    result = source_overlap(["other.html"], ["pods.html"])
    assert result["ratio"] == 0.0
    assert result["passed"] is False


def test_source_overlap_no_golden_sources_defaults_to_pass() -> None:
    result = source_overlap([], [])
    assert result["ratio"] == 1.0


def test_retrieval_metrics_first_result_is_golden() -> None:
    result = retrieval_metrics(["a.pdf", "b.pdf"], ["a.pdf"])
    assert result == {"hit_rate": True, "mrr": 1.0}


def test_retrieval_metrics_golden_found_lower_in_ranking() -> None:
    result = retrieval_metrics(["c.pdf", "b.pdf", "a.pdf"], ["a.pdf"])
    assert result["hit_rate"] is True
    assert result["mrr"] == pytest.approx(1 / 3)


def test_retrieval_metrics_no_golden_source_retrieved() -> None:
    result = retrieval_metrics(["c.pdf", "d.pdf"], ["a.pdf"])
    assert result == {"hit_rate": False, "mrr": 0.0}


def test_retrieval_metrics_empty_ranked_sources() -> None:
    result = retrieval_metrics([], ["a.pdf"])
    assert result == {"hit_rate": False, "mrr": 0.0}
