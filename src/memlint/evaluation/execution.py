"""Deterministic benchmark v0.1 execution orchestration."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from memlint.checkers import (
    Checker,
    CheckerResult,
    OrphanedProvenanceChecker,
    PrivacyScopeViolationChecker,
    RedundancyBloatChecker,
    ScopeIsolationPolicy,
    StaleActiveChecker,
    UnsupportedClaimChecker,
    load_scope_policy,
)
from memlint.evaluation.benchmark import (
    BENCHMARK_ID,
    BENCHMARK_SPEC_SHA256,
    BenchmarkCheckerIdentity,
    BenchmarkFixture,
    BenchmarkSpec,
    CleanControlBenchmarkCase,
    RetrievalBenchmarkCase,
    RetrievalCondition,
    StaticMutationBenchmarkCase,
)
from memlint.evaluation.clean_control import (
    evaluate_clean_control,
    summarize_static_defect_benchmark,
)
from memlint.evaluation.execution_models import (
    BenchmarkExecutionProvenance,
    BenchmarkExecutionResult,
    CleanControlCaseExecution,
    RetrievalCaseExecution,
    StaticCaseExecution,
)
from memlint.evaluation.experimental_lexical import ExperimentalLexicalRetriever
from memlint.evaluation.models import EvaluationInputError
from memlint.evaluation.mutation import evaluate_mutation_trial
from memlint.evaluation.retrieval import summarize_retrieval_challenges
from memlint.models import NormalizedStore, TranscriptSet
from memlint.mutations import BaseStoreStatus, mutate
from memlint.mutations.base import semantic_store_digest, transcript_set_digest
from memlint.retrieval import (
    RetrievalAuditRequest,
    RetrievalSufficiencyPolicy,
    assess_paired_retrieval_challenge,
    retriever_identity,
    run_retrieval_audit,
)
from memlint.semantics import SemanticJudge
from memlint.serialization import load_store, load_transcripts
from memlint.taxonomy import DefectClass

UNSUPPORTED_MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
UNSUPPORTED_MODEL_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"


def validate_checker_result_identity(
    *,
    result: CheckerResult,
    expected: BenchmarkCheckerIdentity,
) -> None:
    """Reject results from any method other than the frozen benchmark checker."""

    if not isinstance(result, CheckerResult):
        raise EvaluationInputError("benchmark checker must return a CheckerResult")
    if not isinstance(expected, BenchmarkCheckerIdentity):
        raise EvaluationInputError("expected checker identity must be benchmark metadata")
    actual = (result.defect_class, result.checker_id, result.checker_version)
    intended = (expected.defect_class, expected.checker_id, expected.checker_version)
    if actual != intended:
        raise EvaluationInputError("checker result identity does not match benchmark v0.1")


def _build_checker(
    defect_class: DefectClass,
    *,
    semantic_judge: SemanticJudge | None,
    scope_policy: ScopeIsolationPolicy | None,
) -> Checker:
    if defect_class is DefectClass.ORPHANED_PROVENANCE:
        return OrphanedProvenanceChecker()
    if defect_class is DefectClass.REDUNDANCY_BLOAT:
        return RedundancyBloatChecker()
    if defect_class is DefectClass.STALE_ACTIVE:
        return StaleActiveChecker()
    if defect_class is DefectClass.PRIVACY_SCOPE_VIOLATION:
        if scope_policy is None:
            raise EvaluationInputError("privacy benchmark cases require a scope policy")
        return PrivacyScopeViolationChecker(scope_policy)
    if defect_class is DefectClass.UNSUPPORTED_CLAIM:
        if semantic_judge is None:
            raise EvaluationInputError("unsupported benchmark cases require a semantic judge")
        return UnsupportedClaimChecker(semantic_judge)
    raise EvaluationInputError("defect class has no benchmark v0.1 static checker")


def run_static_benchmark_case(
    *,
    case: StaticMutationBenchmarkCase,
    fixture: BenchmarkFixture,
    store: NormalizedStore,
    transcripts: TranscriptSet | None,
    expected_checker: BenchmarkCheckerIdentity,
    semantic_judge: SemanticJudge | None = None,
    scope_policy: ScopeIsolationPolicy | None = None,
) -> StaticCaseExecution:
    """Execute one frozen static mutation case with an explicitly selected method."""

    if not isinstance(case, StaticMutationBenchmarkCase):
        raise EvaluationInputError("case must be a StaticMutationBenchmarkCase")
    if not isinstance(fixture, BenchmarkFixture) or fixture.fixture_id != case.base_fixture_id:
        raise EvaluationInputError("static case fixture does not match its specification")
    if fixture.base_store_status is not BaseStoreStatus.CURATED_CLEAN:
        raise EvaluationInputError("static benchmark execution requires a curated-clean fixture")
    if not isinstance(store, NormalizedStore):
        raise EvaluationInputError("static case store must be a NormalizedStore")
    if transcripts is not None and not isinstance(transcripts, TranscriptSet):
        raise EvaluationInputError("static case transcripts must be a TranscriptSet or null")
    if expected_checker.defect_class is not case.defect_class:
        raise EvaluationInputError("static case checker identity has the wrong defect class")

    mutation_result = mutate(store, case.mutation_request, transcripts)
    checker = _build_checker(
        case.defect_class,
        semantic_judge=semantic_judge,
        scope_policy=scope_policy,
    )
    result = checker.check(mutation_result.mutated_store, transcripts=transcripts)
    validate_checker_result_identity(result=result, expected=expected_checker)
    trial = evaluate_mutation_trial(
        base_store=store,
        mutated_store=mutation_result.mutated_store,
        manifest=mutation_result.manifest,
        result=result,
        transcripts=transcripts,
    )
    return StaticCaseExecution(
        case_id=case.case_id,
        mutation_id=mutation_result.manifest.mutation_id,
        checker_result=result,
        trial_evaluation=trial,
    )


def run_clean_control_case(
    *,
    case: CleanControlBenchmarkCase,
    fixture: BenchmarkFixture,
    store: NormalizedStore,
    transcripts: TranscriptSet | None,
    expected_checker: BenchmarkCheckerIdentity,
    semantic_judge: SemanticJudge | None = None,
    scope_policy: ScopeIsolationPolicy | None = None,
) -> CleanControlCaseExecution:
    """Execute one frozen checker against an unmutated curated-clean control."""

    if not isinstance(case, CleanControlBenchmarkCase):
        raise EvaluationInputError("case must be a CleanControlBenchmarkCase")
    if not isinstance(fixture, BenchmarkFixture) or fixture.fixture_id != case.base_fixture_id:
        raise EvaluationInputError("clean-control fixture does not match its specification")
    if not isinstance(store, NormalizedStore):
        raise EvaluationInputError("clean-control store must be a NormalizedStore")
    if transcripts is not None and not isinstance(transcripts, TranscriptSet):
        raise EvaluationInputError("clean-control transcripts must be a TranscriptSet or null")
    if expected_checker.defect_class is not case.defect_class:
        raise EvaluationInputError("clean-control checker identity has the wrong defect class")

    checker = _build_checker(
        case.defect_class,
        semantic_judge=semantic_judge,
        scope_policy=scope_policy,
    )
    result = checker.check(store, transcripts=transcripts)
    validate_checker_result_identity(result=result, expected=expected_checker)
    evaluation = evaluate_clean_control(
        case=case,
        fixture=fixture,
        store=store,
        result=result,
    )
    return CleanControlCaseExecution(
        case_id=case.case_id,
        checker_result=result,
        clean_control_evaluation=evaluation,
    )


def _validate_retrieval_condition(
    case: RetrievalBenchmarkCase,
    condition: RetrievalCondition,
) -> None:
    if case.retrieval_condition_id != condition.condition_id:
        raise EvaluationInputError("retrieval case references a different condition")
    if case.policy is not condition.policy or case.top_k != condition.top_k:
        raise EvaluationInputError("retrieval case does not match its condition")
    if (
        condition.condition_id != "lexical-baseline-k3"
        or condition.policy is not RetrievalSufficiencyPolicy.ALL_EXPECTED
        or condition.top_k != 3
        or condition.retriever_kind != ExperimentalLexicalRetriever.retriever_id
        or condition.retriever_config_version
        != ExperimentalLexicalRetriever.retriever_version
    ):
        raise EvaluationInputError("retrieval condition does not match frozen benchmark v0.1")


def run_retrieval_benchmark_case(
    *,
    case: RetrievalBenchmarkCase,
    fixture: BenchmarkFixture,
    store: NormalizedStore,
    transcripts: TranscriptSet | None,
    condition: RetrievalCondition,
) -> RetrievalCaseExecution:
    """Execute one target-blind paired lexical challenge through retrieval contracts."""

    if not isinstance(case, RetrievalBenchmarkCase):
        raise EvaluationInputError("case must be a RetrievalBenchmarkCase")
    if not isinstance(fixture, BenchmarkFixture) or fixture.fixture_id != case.base_fixture_id:
        raise EvaluationInputError("retrieval case fixture does not match its specification")
    if fixture.base_store_status is not BaseStoreStatus.CURATED_CLEAN:
        raise EvaluationInputError("retrieval benchmark execution requires a curated-clean fixture")
    if not isinstance(store, NormalizedStore):
        raise EvaluationInputError("retrieval case store must be a NormalizedStore")
    if transcripts is not None and not isinstance(transcripts, TranscriptSet):
        raise EvaluationInputError("retrieval transcripts must be a TranscriptSet or null")
    if not isinstance(condition, RetrievalCondition):
        raise EvaluationInputError("condition must be a RetrievalCondition")
    _validate_retrieval_condition(case, condition)

    mutation_result = mutate(store, case.mutation_request, transcripts)
    manifest = mutation_result.manifest
    if semantic_store_digest(store) != manifest.base_store_digest:
        raise EvaluationInputError("retrieval manifest base-store digest mismatch")
    if semantic_store_digest(mutation_result.mutated_store) != manifest.mutated_store_digest:
        raise EvaluationInputError("retrieval manifest mutated-store digest mismatch")
    if transcript_set_digest(transcripts) != manifest.transcript_digest:
        raise EvaluationInputError("retrieval manifest transcript digest mismatch")

    probe = manifest.retrieval_probe
    target_id = case.mutation_request.target_memory_id
    query = case.mutation_request.query
    if probe is None or target_id is None or query is None:
        raise EvaluationInputError("retrieval mutation did not produce the required probe")
    if probe.query != query or probe.expected_memory_ids != (target_id,):
        raise EvaluationInputError("retrieval probe does not match the frozen case query/target")
    if store.get(target_id) is None or mutation_result.mutated_store.get(target_id) is None:
        raise EvaluationInputError("retrieval target must exist in both paired stores")

    baseline_retriever = ExperimentalLexicalRetriever(store)
    mutated_retriever = ExperimentalLexicalRetriever(mutation_result.mutated_store)
    expected_identity = (
        condition.retriever_kind,
        condition.retriever_config_version,
    )
    if retriever_identity(baseline_retriever) != expected_identity:
        raise EvaluationInputError("baseline retriever identity does not match the condition")
    if retriever_identity(mutated_retriever) != expected_identity:
        raise EvaluationInputError("mutated retriever identity does not match the condition")

    baseline_request = RetrievalAuditRequest(
        request_id=f"{case.case_id}:baseline",
        query=query,
        expected_memory_ids=probe.expected_memory_ids,
        top_k=condition.top_k,
    )
    mutated_request = RetrievalAuditRequest(
        request_id=f"{case.case_id}:mutated",
        query=query,
        expected_memory_ids=probe.expected_memory_ids,
        top_k=condition.top_k,
    )
    baseline_observation = run_retrieval_audit(
        store=store,
        request=baseline_request,
        retriever=baseline_retriever,
    )
    mutated_observation = run_retrieval_audit(
        store=mutation_result.mutated_store,
        request=mutated_request,
        retriever=mutated_retriever,
    )
    paired = assess_paired_retrieval_challenge(
        baseline_observation,
        mutated_observation,
        policy=condition.policy,
        case_id=case.case_id,
    )
    return RetrievalCaseExecution(
        case_id=case.case_id,
        baseline_observation=baseline_observation,
        mutated_observation=mutated_observation,
        paired_assessment=paired,
    )


def benchmark_case_order(
    spec: BenchmarkSpec,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Expose the frozen deterministic execution order without running a detector."""

    if not isinstance(spec, BenchmarkSpec):
        raise EvaluationInputError("spec must be a BenchmarkSpec")
    return (
        tuple(sorted(case.case_id for case in spec.static_mutation_cases)),
        tuple(sorted(case.case_id for case in spec.clean_control_cases)),
        tuple(sorted(case.case_id for case in spec.retrieval_cases)),
    )


def _load_fixture_inputs(
    fixture: BenchmarkFixture,
    *,
    repository_root: Path,
) -> tuple[NormalizedStore, TranscriptSet | None, ScopeIsolationPolicy | None]:
    store = load_store(repository_root / fixture.store_path)
    transcripts = (
        None
        if fixture.transcripts_path is None
        else load_transcripts(repository_root / fixture.transcripts_path)
    )
    scope_policy = (
        None
        if fixture.scope_policy_path is None
        else load_scope_policy(repository_root / fixture.scope_policy_path)
    )
    return store, transcripts, scope_policy


def execute_benchmark_v0_1(
    *,
    spec: BenchmarkSpec,
    repository_root: Path,
    semantic_judge: SemanticJudge,
) -> BenchmarkExecutionResult:
    """Execute the already-preflighted benchmark in deterministic case order."""

    if not isinstance(spec, BenchmarkSpec):
        raise EvaluationInputError("spec must be a BenchmarkSpec")
    if not isinstance(repository_root, Path):
        raise EvaluationInputError("repository_root must be a pathlib.Path")

    fixtures = {fixture.fixture_id: fixture for fixture in spec.fixtures}
    loaded = {
        fixture_id: _load_fixture_inputs(fixture, repository_root=repository_root)
        for fixture_id, fixture in sorted(fixtures.items())
    }
    checker_identities = {
        identity.defect_class: identity for identity in spec.checker_identities
    }

    static_executions = tuple(
        run_static_benchmark_case(
            case=case,
            fixture=fixtures[case.base_fixture_id],
            store=loaded[case.base_fixture_id][0],
            transcripts=loaded[case.base_fixture_id][1],
            expected_checker=checker_identities[case.defect_class],
            semantic_judge=semantic_judge,
            scope_policy=loaded[case.base_fixture_id][2],
        )
        for case in sorted(spec.static_mutation_cases, key=lambda item: item.case_id)
    )
    clean_executions = tuple(
        run_clean_control_case(
            case=case,
            fixture=fixtures[case.base_fixture_id],
            store=loaded[case.base_fixture_id][0],
            transcripts=loaded[case.base_fixture_id][1],
            expected_checker=checker_identities[case.defect_class],
            semantic_judge=semantic_judge,
            scope_policy=loaded[case.base_fixture_id][2],
        )
        for case in sorted(spec.clean_control_cases, key=lambda item: item.case_id)
    )
    condition_by_id = {
        condition.condition_id: condition for condition in spec.retrieval_conditions
    }
    retrieval_executions = tuple(
        run_retrieval_benchmark_case(
            case=case,
            fixture=fixtures[case.base_fixture_id],
            store=loaded[case.base_fixture_id][0],
            transcripts=loaded[case.base_fixture_id][1],
            condition=condition_by_id[case.retrieval_condition_id],
        )
        for case in sorted(spec.retrieval_cases, key=lambda item: item.case_id)
    )

    static_summaries = summarize_static_defect_benchmark(
        mutation_trials=tuple(item.trial_evaluation for item in static_executions),
        clean_controls=tuple(
            item.clean_control_evaluation for item in clean_executions
        ),
    )
    retrieval_summary = summarize_retrieval_challenges(
        tuple(item.paired_assessment for item in retrieval_executions)
    )
    return BenchmarkExecutionResult(
        benchmark_id=BENCHMARK_ID,
        benchmark_spec_sha256=BENCHMARK_SPEC_SHA256,
        static_cases=static_executions,
        clean_controls=clean_executions,
        static_defect_summaries=static_summaries,
        retrieval_cases=retrieval_executions,
        retrieval_summary=retrieval_summary,
    )


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as error:  # pragma: no cover - real runner dependencies
        raise EvaluationInputError(
            f"required benchmark runtime package is not installed: {distribution}"
        ) from error


def build_execution_provenance() -> BenchmarkExecutionProvenance:
    """Capture safe version-only environment provenance outside scoring."""

    platform_identity = " ".join(
        part for part in (platform.system(), platform.release(), platform.machine()) if part
    )
    return BenchmarkExecutionProvenance(
        benchmark_id=BENCHMARK_ID,
        benchmark_spec_sha256=BENCHMARK_SPEC_SHA256,
        python_version=platform.python_version(),
        platform=platform_identity,
        torch_version=_package_version("torch"),
        transformers_version=_package_version("transformers"),
        tokenizers_version=_package_version("tokenizers"),
        safetensors_version=_package_version("safetensors"),
        unsupported_model_id=UNSUPPORTED_MODEL_ID,
        unsupported_model_revision=UNSUPPORTED_MODEL_REVISION,
        device="cpu",
    )
