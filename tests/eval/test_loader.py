from app.eval.loader import filter_by_feature, load_golden_cases


def test_load_golden_cases_parses_all_forty() -> None:
    cases = load_golden_cases()

    assert len(cases) == 40
    assert {case.id for case in cases} == {f"q-{i:03d}" for i in range(1, 41)}


def test_forbidden_keywords_default_to_empty_except_security_sentinel() -> None:
    by_id = {case.id: case for case in load_golden_cases()}

    assert by_id["q-001"].forbidden_keywords == []
    assert by_id["q-033"].forbidden_keywords == [
        "CVV",
        "full account number",
        "ignore policy",
        "OTP",
    ]


def test_filter_by_feature_narrows_to_matching_cases() -> None:
    cases = load_golden_cases()

    hyde_cases = filter_by_feature(cases, "hyde")

    assert len(hyde_cases) == 5  # q-013, q-014, q-015, q-016, q-038
    assert all(case.demonstrates_feature.value == "hyde" for case in hyde_cases)


def test_filter_by_feature_none_returns_everything() -> None:
    cases = load_golden_cases()

    assert filter_by_feature(cases, None) == cases
