from app.eval.post_checks import forbidden_keywords_check, source_overlap


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
