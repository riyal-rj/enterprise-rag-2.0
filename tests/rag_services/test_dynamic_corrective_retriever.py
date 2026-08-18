from __future__ import annotations

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.crag.crag import CRAGDecision, CRAGOutcome
from app.rag_services.crag.dynamic_corrective_retriever import DynamicCorrectiveRetriever
from app.rag_services.dynamic_query_transformer import DynamicQueryTransformer
from app.rag_services.query_transformer import QueryTransformOutcome
from app.rag_services.rag_runtime_config import RagRuntimeConfig


class _FakeDelegate:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @property
    def cache_namespace(self) -> str:
        return "crag:fake:v1"

    def correct(
        self, question: str, chunks: list[RetrievedChunk], *, allow_web: bool
    ) -> CRAGOutcome:
        self.calls.append({"question": question, "allow_web": allow_web})
        return CRAGOutcome(evidence=(), decision=CRAGDecision.CORRECT, applied=True)


class _FakeHydeDelegate:
    @property
    def name(self) -> str:
        return "hyde"

    @property
    def cache_namespace(self) -> str:
        return "hyde:v1"

    def transform(self, query: str) -> QueryTransformOutcome:
        return QueryTransformOutcome(retrieval_texts=(query,), backend="hyde", applied=True)


def _config(
    *,
    rollout_percentage: int = 100,
    emergency_disabled: bool = False,
    crag_web_enabled: bool = False,
    hyde_rollout_percentage: int = 0,
) -> RagRuntimeConfig:
    return RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=emergency_disabled,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=hyde_rollout_percentage > 0,
        hyde_rollout_percentage=hyde_rollout_percentage,
        crag_enabled=True,
        crag_rollout_percentage=rollout_percentage,
        crag_web_enabled=crag_web_enabled and rollout_percentage > 0,
    )


def test_plan_is_disabled_cohort_when_feature_off() -> None:
    retriever = DynamicCorrectiveRetriever(delegate=_FakeDelegate())

    plan = retriever.plan("q", _config(), enabled=False)

    assert plan.cohort == "disabled"
    assert plan.bypass_reason == "disabled"
    assert plan.allow_web is False


def test_emergency_disabled_bypasses_the_delegate_entirely() -> None:
    delegate = _FakeDelegate()
    retriever = DynamicCorrectiveRetriever(delegate=delegate)

    plan = retriever.plan("q", _config(emergency_disabled=True), enabled=True)
    outcome = retriever.execute("q", [], plan)

    assert plan.cohort == "disabled"
    assert plan.bypass_reason == "emergency_disabled"
    assert outcome.applied is False
    assert delegate.calls == []


def test_zero_percent_rollout_is_control_and_never_calls_the_delegate() -> None:
    delegate = _FakeDelegate()
    retriever = DynamicCorrectiveRetriever(delegate=delegate)

    plan = retriever.plan("q", _config(rollout_percentage=0), enabled=True)
    outcome = retriever.execute("q", [], plan)

    assert plan.cohort == "control"
    assert plan.bypass_reason == "rollout"
    assert outcome.applied is False
    assert delegate.calls == []


def test_hundred_percent_rollout_is_always_treatment_and_calls_the_delegate() -> None:
    delegate = _FakeDelegate()
    retriever = DynamicCorrectiveRetriever(delegate=delegate)
    config = _config(rollout_percentage=100)

    for question in ["a", "b", "c", "d"]:
        plan = retriever.plan(question, config, enabled=True)
        retriever.execute(question, [], plan)

    assert [c["question"] for c in delegate.calls] == ["a", "b", "c", "d"]


def test_rollout_sampling_is_deterministic_for_the_same_normalized_question() -> None:
    retriever = DynamicCorrectiveRetriever(delegate=_FakeDelegate())
    config = _config(rollout_percentage=50)

    first_run = [retriever.plan(f"question {i}", config, enabled=True).cohort for i in range(20)]
    second_run = [retriever.plan(f"question {i}", config, enabled=True).cohort for i in range(20)]

    assert first_run == second_run


def test_crag_salt_differs_from_hyde_salt() -> None:
    """Both use sampled_in() but with feature-specific salts - a 50%
    rollout for CRAG must not deterministically select the exact same
    population as a 50% rollout for HyDE on the same questions."""
    crag = DynamicCorrectiveRetriever(delegate=_FakeDelegate())
    hyde = DynamicQueryTransformer(delegate=_FakeHydeDelegate())
    config = _config(rollout_percentage=50, hyde_rollout_percentage=50)

    questions = [f"question {i}" for i in range(50)]
    crag_sampled_in = {
        q for q in questions if crag.plan(q, config, enabled=True).cohort == "treatment"
    }
    hyde_sampled_in = {
        q for q in questions if hyde.plan(q, config, enabled=True).cohort == "treatment"
    }

    assert crag_sampled_in != hyde_sampled_in


def test_control_and_treatment_namespaces_differ() -> None:
    retriever = DynamicCorrectiveRetriever(delegate=_FakeDelegate())
    config = _config(rollout_percentage=50)

    control_question = next(
        q
        for q in (f"q{i}" for i in range(50))
        if retriever.plan(q, config, enabled=True).cohort == "control"
    )
    treatment_question = next(
        q
        for q in (f"q{i}" for i in range(50))
        if retriever.plan(q, config, enabled=True).cohort == "treatment"
    )

    control_plan = retriever.plan(control_question, config, enabled=True)
    treatment_plan = retriever.plan(treatment_question, config, enabled=True)

    assert control_plan.cache_namespace != treatment_plan.cache_namespace


def test_changing_config_after_plan_does_not_affect_execute() -> None:
    """DynamicCorrectiveRetriever holds no config reference of its own -
    execute() only ever consults the plan it was given."""
    delegate = _FakeDelegate()
    retriever = DynamicCorrectiveRetriever(delegate=delegate)

    plan = retriever.plan("q", _config(rollout_percentage=100), enabled=True)
    later_config = _config(rollout_percentage=0, emergency_disabled=True)
    assert later_config.crag_rollout_percentage == 0  # sanity: genuinely different

    outcome = retriever.execute("q", [], plan)

    assert outcome.applied is True
    assert delegate.calls == [{"question": "q", "allow_web": False}]


def test_planned_allow_web_remains_immutable_across_execute() -> None:
    delegate = _FakeDelegate()
    retriever = DynamicCorrectiveRetriever(delegate=delegate)

    plan = retriever.plan("q", _config(rollout_percentage=100, crag_web_enabled=True), enabled=True)
    assert plan.allow_web is True

    retriever.execute("q", [], plan)

    assert delegate.calls[0]["allow_web"] is True


def test_shadow_mode_actually_invokes_the_delegate() -> None:
    """Shadow must run the real correction (for observation), unlike
    control/disabled - see RAGService.answer, which discards the served
    outcome but still needs execute() to return the real one."""
    delegate = _FakeDelegate()
    retriever = DynamicCorrectiveRetriever(delegate=delegate)
    config = RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=False,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        crag_enabled=True,
        crag_rollout_percentage=0,  # would be "control" if shadow weren't checked first
        crag_web_enabled=False,
        crag_shadow_enabled=True,
    )

    plan = retriever.plan("q", config, enabled=True)
    outcome = retriever.execute("q", [], plan)

    assert plan.cohort == "shadow"
    assert outcome.applied is True
    assert delegate.calls == [{"question": "q", "allow_web": False}]


def test_shadow_mode_never_allows_web_even_if_crag_web_enabled_is_true() -> None:
    delegate = _FakeDelegate()
    retriever = DynamicCorrectiveRetriever(delegate=delegate)
    config = RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=False,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        crag_enabled=True,
        crag_rollout_percentage=100,
        crag_web_enabled=True,
        crag_shadow_enabled=True,
    )

    plan = retriever.plan("q", config, enabled=True)

    assert plan.cohort == "shadow"
    assert plan.allow_web is False


def test_emergency_disabled_takes_priority_over_shadow_mode() -> None:
    """Emergency-disable must still win over shadow mode - a kill switch
    must stop every CRAG code path, observation included."""
    delegate = _FakeDelegate()
    retriever = DynamicCorrectiveRetriever(delegate=delegate)
    config = RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=True,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        crag_enabled=True,
        crag_rollout_percentage=100,
        crag_web_enabled=False,
        crag_shadow_enabled=True,
    )

    plan = retriever.plan("q", config, enabled=True)
    outcome = retriever.execute("q", [], plan)

    assert plan.cohort == "disabled"
    assert plan.bypass_reason == "emergency_disabled"
    assert outcome.applied is False
    assert delegate.calls == []


def test_allow_web_false_when_config_disables_it() -> None:
    delegate = _FakeDelegate()
    retriever = DynamicCorrectiveRetriever(delegate=delegate)

    plan = retriever.plan(
        "q", _config(rollout_percentage=100, crag_web_enabled=False), enabled=True
    )
    retriever.execute("q", [], plan)

    assert plan.allow_web is False
    assert delegate.calls[0]["allow_web"] is False
