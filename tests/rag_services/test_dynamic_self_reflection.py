"""Tests for DynamicSelfReflectionEngine's production rollout."""

from __future__ import annotations

from app.rag_services.crag import EvidenceChunk
from app.rag_services.rag_runtime_config import RagRuntimeConfig
from app.rag_services.reflection.dynamic_self_reflection import DynamicSelfReflectionEngine
from app.rag_services.reflection.reflection import ReflectionAction, SelfReflectionOutcome


class _FakeDelegate:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @property
    def cache_namespace(self) -> str:
        return "self-reflection:fake:v1"

    def reflect(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        initial_answer: str,
        augmenter: object,
    ) -> SelfReflectionOutcome:
        self.calls.append({"question": question, "initial_answer": initial_answer})
        return SelfReflectionOutcome(
            answer=initial_answer,
            evidence=evidence,
            applied=True,
            accepted=True,
            final_action=ReflectionAction.ACCEPT,
            iterations=1,
            additional_retrievals=0,
        )


def _config(
    *,
    rollout_percentage: int = 100,
    emergency_disabled: bool = False,
) -> RagRuntimeConfig:
    return RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=emergency_disabled,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
        self_reflective_enabled=rollout_percentage > 0,
        self_reflective_rollout_percentage=rollout_percentage,
    )


def test_plan_is_disabled_cohort_when_feature_off() -> None:
    engine = DynamicSelfReflectionEngine(delegate=_FakeDelegate())

    plan = engine.plan("q", _config(), enabled=False)

    assert plan.cohort == "disabled"
    assert plan.bypass_reason == "disabled"


def test_emergency_disabled_bypasses_the_delegate_entirely() -> None:
    delegate = _FakeDelegate()
    engine = DynamicSelfReflectionEngine(delegate=delegate)

    plan = engine.plan("q", _config(emergency_disabled=True), enabled=True)
    outcome = engine.execute("q", (), "initial", object(), plan)  # type: ignore[arg-type]

    assert plan.cohort == "disabled"
    assert plan.bypass_reason == "emergency_disabled"
    assert outcome.applied is False
    assert delegate.calls == []


def test_zero_percent_rollout_is_control_and_never_calls_the_delegate() -> None:
    delegate = _FakeDelegate()
    engine = DynamicSelfReflectionEngine(delegate=delegate)

    plan = engine.plan("q", _config(rollout_percentage=0), enabled=True)
    outcome = engine.execute("q", (), "initial", object(), plan)  # type: ignore[arg-type]

    assert plan.cohort == "control"
    assert plan.bypass_reason == "rollout"
    assert outcome.applied is False
    assert outcome.answer == "initial"
    assert delegate.calls == []


def test_hundred_percent_rollout_is_always_treatment_and_calls_the_delegate() -> None:
    delegate = _FakeDelegate()
    engine = DynamicSelfReflectionEngine(delegate=delegate)
    config = _config(rollout_percentage=100)

    for question in ["a", "b", "c", "d"]:
        plan = engine.plan(question, config, enabled=True)
        engine.execute(question, (), "initial", object(), plan)  # type: ignore[arg-type]

    assert [c["question"] for c in delegate.calls] == ["a", "b", "c", "d"]


def test_rollout_sampling_is_deterministic_for_the_same_normalized_question() -> None:
    engine = DynamicSelfReflectionEngine(delegate=_FakeDelegate())
    config = _config(rollout_percentage=50)

    first_run = [engine.plan(f"question {i}", config, enabled=True).cohort for i in range(20)]
    second_run = [engine.plan(f"question {i}", config, enabled=True).cohort for i in range(20)]

    assert first_run == second_run


def test_control_and_treatment_namespaces_differ() -> None:
    engine = DynamicSelfReflectionEngine(delegate=_FakeDelegate())
    config = _config(rollout_percentage=50)

    control_question = next(
        q
        for q in (f"q{i}" for i in range(50))
        if engine.plan(q, config, enabled=True).cohort == "control"
    )
    treatment_question = next(
        q
        for q in (f"q{i}" for i in range(50))
        if engine.plan(q, config, enabled=True).cohort == "treatment"
    )

    control_plan = engine.plan(control_question, config, enabled=True)
    treatment_plan = engine.plan(treatment_question, config, enabled=True)

    assert control_plan.cache_namespace != treatment_plan.cache_namespace


def test_changing_config_after_plan_does_not_affect_execute() -> None:
    """DynamicSelfReflectionEngine holds no config reference of its own -
    execute() only ever consults the plan it was given."""
    delegate = _FakeDelegate()
    engine = DynamicSelfReflectionEngine(delegate=delegate)

    plan = engine.plan("q", _config(rollout_percentage=100), enabled=True)
    later_config = _config(rollout_percentage=0, emergency_disabled=True)
    assert later_config.self_reflective_rollout_percentage == 0  # sanity: genuinely different

    outcome = engine.execute("q", (), "initial", object(), plan)  # type: ignore[arg-type]

    assert outcome.applied is True
    assert delegate.calls == [{"question": "q", "initial_answer": "initial"}]


def test_a_reflection_error_falls_back_to_the_initial_answer_not_a_5xx() -> None:
    """Wired through FailSafeSelfReflectionEngine internally - a delegate
    exception must never propagate to the caller."""

    class _RaisingDelegate:
        @property
        def cache_namespace(self) -> str:
            return "self-reflection:raising:v1"

        def reflect(
            self, question: str, evidence: object, initial_answer: str, augmenter: object
        ) -> SelfReflectionOutcome:
            raise TimeoutError("critic call timed out")

    engine = DynamicSelfReflectionEngine(delegate=_RaisingDelegate())
    plan = engine.plan("q", _config(rollout_percentage=100), enabled=True)

    outcome = engine.execute("q", (), "initial answer", object(), plan)  # type: ignore[arg-type]

    assert outcome.fallback is True
    assert outcome.accepted is False
    assert outcome.answer == "initial answer"
