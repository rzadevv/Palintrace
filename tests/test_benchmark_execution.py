from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

import memlint.evaluation.execution as execution
import memlint.evaluation.preflight as preflight
from memlint.checkers import load_scope_policy
from memlint.evaluation import (
    BENCHMARK_SPEC_SHA256,
    BenchmarkCaseKind,
    BenchmarkCheckerIdentity,
    BenchmarkFixture,
    BenchmarkSplit,
    CleanControlBenchmarkCase,
    EvaluationInputError,
    RetrievalBenchmarkCase,
    RetrievalCondition,
    StaticMutationBenchmarkCase,
    benchmark_case_order,
    load_benchmark_spec,
    preflight_benchmark_v0_1,
    run_clean_control_case,
    run_retrieval_benchmark_case,
    run_static_benchmark_case,
)
from memlint.models import MemoryScope, NormalizedMemory, NormalizedStore
from memlint.mutations import BaseStoreStatus, DistractorFamily, MutationRequest
from memlint.retrieval import (
    RetrievalObservation,
    RetrievalSufficiencyPolicy,
)
from memlint.semantics import (
    SemanticJudgment,
    SemanticRelation,
    SemanticUsage,
)
from memlint.serialization import load_store, load_transcripts
from memlint.taxonomy import DefectClass

BENCHMARK_PATH = Path("tests/fixtures/benchmark_v0.1/benchmark.json")


class FakeJudge:
    judge_id = "fake:benchmark-dispatch"
    judge_version = "toy-1"

    def __init__(self, relation: SemanticRelation = SemanticRelation.NEUTRAL) -> None:
        self.relation = relation
        self.calls: list[tuple[str, str]] = []

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        self.calls.append((premise, hypothesis))
        return SemanticJudgment(
            relation=self.relation,
            score=1.0,
            usage=SemanticUsage(model_calls=1, input_tokens=2, output_tokens=0),
        )


def _fixture() -> BenchmarkFixture:
    return BenchmarkFixture(
        fixture_id="DEV",
        store_path="examples/mutation-store.json",
        transcripts_path="examples/mutation-transcripts.json",
        scope_policy_path="examples/scope-policy.json",
        base_store_status=BaseStoreStatus.CURATED_CLEAN,
    )


def _identity(defect_class: DefectClass) -> BenchmarkCheckerIdentity:
    return BenchmarkCheckerIdentity(
        defect_class=defect_class,
        checker_id=defect_class.value,
        checker_version="1.0",
    )


def _static_case(defect_class: DefectClass) -> StaticMutationBenchmarkCase:
    request_by_defect = {
        DefectClass.ORPHANED_PROVENANCE: MutationRequest(
            defect_class=defect_class,
            subtype="missing_transcript",
            target_memory_id="preference-python",
            base_store_status=BaseStoreStatus.CURATED_CLEAN,
        ),
        DefectClass.REDUNDANCY_BLOAT: MutationRequest(
            defect_class=defect_class,
            subtype="exact_duplicate",
            target_memory_id="preference-python",
            base_store_status=BaseStoreStatus.CURATED_CLEAN,
        ),
        DefectClass.STALE_ACTIVE: MutationRequest(
            defect_class=defect_class,
            subtype="explicit_supersession",
            target_memory_id="preference-python",
            replace_from="Python",
            replace_to="Elixir",
            base_store_status=BaseStoreStatus.CURATED_CLEAN,
        ),
        DefectClass.PRIVACY_SCOPE_VIOLATION: MutationRequest(
            defect_class=defect_class,
            subtype="cross_user_copy",
            target_memory_id="preference-python",
            destination_user_id="user-b",
            base_store_status=BaseStoreStatus.CURATED_CLEAN,
        ),
        DefectClass.UNSUPPORTED_CLAIM: MutationRequest(
            defect_class=defect_class,
            subtype="factual_substitution",
            target_memory_id="preference-python",
            replace_from="Python",
            replace_to="Elixir",
            base_store_status=BaseStoreStatus.CURATED_CLEAN,
        ),
    }
    return StaticMutationBenchmarkCase(
        case_id=f"TOY-{defect_class.value}",
        kind=BenchmarkCaseKind.STATIC_MUTATION,
        split=BenchmarkSplit.HELD_OUT,
        defect_class=defect_class,
        subtype=request_by_defect[defect_class].subtype or "",
        base_fixture_id="DEV",
        transcript_fixture_id="DEV",
        mutation_request=request_by_defect[defect_class],
        semantic_domain="toy"
        if defect_class is DefectClass.UNSUPPORTED_CLAIM
        else None,
    )


@pytest.mark.parametrize(
    "defect_class",
    (
        DefectClass.ORPHANED_PROVENANCE,
        DefectClass.REDUNDANCY_BLOAT,
        DefectClass.STALE_ACTIVE,
        DefectClass.PRIVACY_SCOPE_VIOLATION,
        DefectClass.UNSUPPORTED_CLAIM,
    ),
)
def test_static_dispatch_uses_exact_checker_and_injected_dependencies(
    defect_class: DefectClass,
) -> None:
    store = load_store("examples/mutation-store.json")
    transcripts = load_transcripts("examples/mutation-transcripts.json")
    judge = FakeJudge()
    scope_policy = load_scope_policy("examples/scope-policy.json")

    result = run_static_benchmark_case(
        case=_static_case(defect_class),
        fixture=_fixture(),
        store=store,
        transcripts=transcripts,
        expected_checker=_identity(defect_class),
        semantic_judge=judge,
        scope_policy=scope_policy,
    )
    assert result.checker_result.checker_id == defect_class.value
    assert result.checker_result.checker_version == "1.0"
    assert result.checker_result.defect_class is defect_class
    assert result.trial_evaluation.injected_positive_detected is True
    assert bool(judge.calls) is (defect_class is DefectClass.UNSUPPORTED_CLAIM)


def test_static_dispatch_rejects_checker_identity_mismatch() -> None:
    store = load_store("examples/mutation-store.json")
    transcripts = load_transcripts("examples/mutation-transcripts.json")
    defect = DefectClass.ORPHANED_PROVENANCE
    wrong = _identity(defect).model_copy(update={"checker_version": "2.0"})
    with pytest.raises(EvaluationInputError, match="identity"):
        run_static_benchmark_case(
            case=_static_case(defect),
            fixture=_fixture(),
            store=store,
            transcripts=transcripts,
            expected_checker=wrong,
        )


def test_clean_control_dispatch_uses_unmutated_store_and_explicit_policy() -> None:
    store = load_store("examples/mutation-store.json")
    policy = load_scope_policy("examples/scope-policy.json")
    case = CleanControlBenchmarkCase(
        case_id="TOY-CLEAN-PRIVACY",
        kind=BenchmarkCaseKind.CLEAN_CONTROL,
        split=BenchmarkSplit.HELD_OUT,
        defect_class=DefectClass.PRIVACY_SCOPE_VIOLATION,
        base_fixture_id="DEV",
        scope_policy_fixture_id="DEV",
    )
    execution_result = run_clean_control_case(
        case=case,
        fixture=_fixture(),
        store=store,
        transcripts=None,
        expected_checker=_identity(DefectClass.PRIVACY_SCOPE_VIOLATION),
        scope_policy=policy,
    )
    assert execution_result.checker_result.findings == ()
    assert execution_result.clean_control_evaluation.alert_present is False


def _toy_retrieval_store() -> NormalizedStore:
    return NormalizedStore(
        schema_version="0.1",
        adapter="toy-retrieval",
        memories=(
            NormalizedMemory(
                id="target",
                content="Mira drafts field notes in QuartzPad.",
                scope=MemoryScope(user_id="mira", agent_id="helper"),
                active=True,
            ),
            NormalizedMemory(
                id="other",
                content="Mira drinks mint tea after lunch.",
                scope=MemoryScope(user_id="mira", agent_id="helper"),
                active=True,
            ),
        ),
    )


def _toy_retrieval_case() -> RetrievalBenchmarkCase:
    return RetrievalBenchmarkCase(
        case_id="TOY-RS-01",
        kind=BenchmarkCaseKind.RETRIEVAL_CHALLENGE,
        split=BenchmarkSplit.HELD_OUT,
        base_fixture_id="TOY",
        mutation_request=MutationRequest(
            defect_class=DefectClass.RETRIEVAL_SHADOWING,
            subtype="distractor_crowding",
            target_memory_id="target",
            query="Where does Mira draft field notes?",
            distractor_family=DistractorFamily.EDITOR,
            distractor_count=3,
            base_store_status=BaseStoreStatus.CURATED_CLEAN,
        ),
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
        top_k=3,
        retrieval_condition_id="lexical-baseline-k3",
    )


def _condition() -> RetrievalCondition:
    return RetrievalCondition(
        condition_id="lexical-baseline-k3",
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
        top_k=3,
        retriever_kind="experimental_lexical",
        retriever_config_version="0.1",
    )


def test_retrieval_orchestration_reuses_part5_contracts_on_toy_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_calls: list[str] = []
    paired_calls: list[str] = []
    original_audit = execution.run_retrieval_audit
    original_paired = execution.assess_paired_retrieval_challenge

    def recording_audit(**kwargs: object) -> RetrievalObservation:
        request = kwargs["request"]
        assert hasattr(request, "request_id")
        audit_calls.append(request.request_id)  # type: ignore[attr-defined]
        return original_audit(**kwargs)  # type: ignore[arg-type]

    def recording_paired(*args: object, **kwargs: object) -> object:
        paired_calls.append(str(kwargs["case_id"]))
        return original_paired(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(execution, "run_retrieval_audit", recording_audit)
    monkeypatch.setattr(
        execution,
        "assess_paired_retrieval_challenge",
        recording_paired,
    )
    result = run_retrieval_benchmark_case(
        case=_toy_retrieval_case(),
        fixture=BenchmarkFixture(
            fixture_id="TOY",
            store_path="tests/toy.json",
            base_store_status=BaseStoreStatus.CURATED_CLEAN,
        ),
        store=_toy_retrieval_store(),
        transcripts=None,
        condition=_condition(),
    )
    assert audit_calls == ["TOY-RS-01:baseline", "TOY-RS-01:mutated"]
    assert paired_calls == ["TOY-RS-01"]
    assert result.baseline_observation.expected_memory_ids == ("target",)
    assert result.mutated_observation.expected_memory_ids == ("target",)
    assert result.baseline_observation.query_sha256 == hashlib.sha256(
        b"Where does Mira draft field notes?"
    ).hexdigest()
    assert result.paired_assessment.policy is RetrievalSufficiencyPolicy.ALL_EXPECTED


def test_retrieval_condition_identity_is_fail_closed() -> None:
    wrong = _condition().model_copy(update={"retriever_config_version": "0.2"})
    with pytest.raises(EvaluationInputError, match="frozen"):
        run_retrieval_benchmark_case(
            case=_toy_retrieval_case(),
            fixture=BenchmarkFixture(
                fixture_id="TOY",
                store_path="tests/toy.json",
                base_store_status=BaseStoreStatus.CURATED_CLEAN,
            ),
            store=_toy_retrieval_store(),
            transcripts=None,
            condition=wrong,
        )


def test_frozen_case_order_is_explicit_without_execution() -> None:
    spec = load_benchmark_spec(BENCHMARK_PATH)
    static_ids, clean_ids, retrieval_ids = benchmark_case_order(spec)
    assert static_ids == tuple(sorted(static_ids))
    assert clean_ids == tuple(sorted(clean_ids))
    assert retrieval_ids == tuple(sorted(retrieval_ids))
    source = inspect.getsource(execution.benchmark_case_order)
    assert "sorted" in source


def test_preflight_verifies_canonical_and_fixture_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = preflight_benchmark_v0_1(
        repository_root=Path("."),
        benchmark_path=BENCHMARK_PATH,
    )
    assert hashlib.sha256(spec.to_json(indent=None).encode()).hexdigest() == (
        BENCHMARK_SPEC_SHA256
    )

    monkeypatch.setattr(
        "memlint.evaluation.preflight._sha256_bytes",
        lambda path: (
            preflight.FROZEN_FIXTURE_HASH_MANIFEST_SHA256
            if path.name == "benchmark_v0.1.sha256.json"
            else "0" * 64
        ),
    )
    with pytest.raises(EvaluationInputError, match="SHA mismatch"):
        preflight_benchmark_v0_1(
            repository_root=Path("."),
            benchmark_path=BENCHMARK_PATH,
        )
