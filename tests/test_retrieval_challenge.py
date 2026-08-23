import ast
import hashlib
import importlib
import inspect
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import memlint.cli as cli
from memlint.models import NormalizedMemory, NormalizedStore
from memlint.mutations import DistractorFamily, MutationRequest, mutate
from memlint.retrieval import (
    PairedRetrievalChallengeAssessment,
    RetrievalAuditRequest,
    RetrievalChallengeInputError,
    RetrievalChallengeOutcome,
    RetrievalHit,
    RetrievalObservation,
    RetrievalResponse,
    RetrievalSufficiencyAssessment,
    RetrievalSufficiencyPolicy,
    RetrievalUsage,
    assess_paired_retrieval_challenge,
    run_retrieval_audit,
)
from memlint.taxonomy import DefectClass

CHALLENGE_PATH = Path("src/memlint/retrieval/challenge.py")
QUERY_TEXT = "Which target should this paired retrieval challenge return?"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _observation(
    *,
    request_id: str,
    expected_memory_ids: tuple[str, ...] = ("m1",),
    hits: tuple[RetrievalHit, ...] = (),
    query_sha256: str | None = None,
    top_k: int = 3,
    retriever_id: str = "paired-retriever",
    retriever_version: str = "1",
    usage: RetrievalUsage | None = None,
) -> RetrievalObservation:
    return RetrievalObservation(
        request_id=request_id,
        query_sha256=query_sha256 or _sha(QUERY_TEXT),
        expected_memory_ids=expected_memory_ids,
        top_k=top_k,
        retriever_id=retriever_id,
        retriever_version=retriever_version,
        hits=hits,
        usage=usage or RetrievalUsage(retrieval_calls=1, candidate_count=len(hits)),
    )


def _pair(
    *,
    baseline_hits: tuple[RetrievalHit, ...] = (),
    mutated_hits: tuple[RetrievalHit, ...] = (),
    expected_memory_ids: tuple[str, ...] = ("m1",),
    policy: RetrievalSufficiencyPolicy = RetrievalSufficiencyPolicy.ALL_EXPECTED,
) -> PairedRetrievalChallengeAssessment:
    return assess_paired_retrieval_challenge(
        _observation(
            request_id="baseline-request",
            expected_memory_ids=expected_memory_ids,
            hits=baseline_hits,
        ),
        _observation(
            request_id="mutated-request",
            expected_memory_ids=expected_memory_ids,
            hits=mutated_hits,
        ),
        policy=policy,
        case_id="paired-case-1",
    )


def test_challenge_outcome_has_exactly_three_frozen_values() -> None:
    assert tuple(RetrievalChallengeOutcome) == (
        RetrievalChallengeOutcome.INDUCED_SHADOWING,
        RetrievalChallengeOutcome.RESILIENT,
        RetrievalChallengeOutcome.BASELINE_INSUFFICIENT,
    )
    assert [outcome.value for outcome in RetrievalChallengeOutcome] == [
        "induced_shadowing",
        "resilient",
        "baseline_insufficient",
    ]


def test_pair_function_requires_explicit_policy_and_case_id_without_assessments() -> None:
    signature = inspect.signature(assess_paired_retrieval_challenge)

    assert tuple(signature.parameters) == ("baseline", "mutated", "policy", "case_id")
    assert signature.parameters["policy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["case_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["policy"].default is inspect.Parameter.empty
    assert signature.parameters["case_id"].default is inspect.Parameter.empty
    assert "baseline_assessment" not in signature.parameters
    assert "mutated_assessment" not in signature.parameters


@pytest.mark.parametrize("case_id", ["", "   ", 1, None])
def test_case_id_must_be_a_nonblank_string(case_id: object) -> None:
    with pytest.raises(RetrievalChallengeInputError, match="case_id"):
        assess_paired_retrieval_challenge(
            _observation(request_id="baseline"),
            _observation(request_id="mutated"),
            policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
            case_id=cast(str, case_id),
        )


def test_real_observation_instances_and_policy_enum_are_required() -> None:
    baseline = _observation(request_id="baseline")
    mutated = _observation(request_id="mutated")

    with pytest.raises(RetrievalChallengeInputError, match="baseline"):
        assess_paired_retrieval_challenge(
            cast(RetrievalObservation, object()),
            mutated,
            policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
            case_id="case",
        )
    with pytest.raises(RetrievalChallengeInputError, match="mutated"):
        assess_paired_retrieval_challenge(
            baseline,
            cast(RetrievalObservation, object()),
            policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
            case_id="case",
        )
    with pytest.raises(RetrievalChallengeInputError, match="policy"):
        assess_paired_retrieval_challenge(
            baseline,
            mutated,
            policy=cast(RetrievalSufficiencyPolicy, "all_expected"),
            case_id="case",
        )


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("query_sha256", _sha("different query")),
        ("expected_memory_ids", ("m1", "m2")),
        ("top_k", 4),
        ("retriever_id", "different-retriever"),
        ("retriever_version", "2"),
    ],
)
def test_pair_requires_exact_experimental_compatibility(
    field: str,
    changed_value: object,
) -> None:
    baseline = _observation(request_id="baseline")
    mutated = _observation(request_id="mutated").model_copy(update={field: changed_value})

    with pytest.raises(RetrievalChallengeInputError, match=field):
        assess_paired_retrieval_challenge(
            baseline,
            mutated,
            policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
            case_id="case",
        )


def test_request_ids_may_differ_and_are_recorded() -> None:
    assessment = assess_paired_retrieval_challenge(
        _observation(request_id="baseline-distinct"),
        _observation(request_id="mutated-distinct"),
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
        case_id="bound-pair",
    )

    assert assessment.case_id == "bound-pair"
    assert assessment.baseline_request_id == "baseline-distinct"
    assert assessment.mutated_request_id == "mutated-distinct"


@pytest.mark.parametrize(
    ("baseline_present", "mutated_present", "expected_outcome"),
    [
        (True, False, RetrievalChallengeOutcome.INDUCED_SHADOWING),
        (True, True, RetrievalChallengeOutcome.RESILIENT),
        (False, False, RetrievalChallengeOutcome.BASELINE_INSUFFICIENT),
        (False, True, RetrievalChallengeOutcome.BASELINE_INSUFFICIENT),
    ],
)
def test_exact_outcome_logic(
    baseline_present: bool,
    mutated_present: bool,
    expected_outcome: RetrievalChallengeOutcome,
) -> None:
    target_hit = (RetrievalHit(memory_id="m1", rank=1),)
    assessment = _pair(
        baseline_hits=target_hit if baseline_present else (),
        mutated_hits=target_hit if mutated_present else (),
    )

    assert assessment.baseline_sufficient is baseline_present
    assert assessment.mutated_sufficient is mutated_present
    assert assessment.outcome is expected_outcome


def test_assessment_has_only_paired_methodology_fields() -> None:
    assert set(PairedRetrievalChallengeAssessment.model_fields) == {
        "case_id",
        "policy",
        "outcome",
        "baseline_request_id",
        "mutated_request_id",
        "query_sha256",
        "expected_memory_ids",
        "top_k",
        "retriever_id",
        "retriever_version",
        "baseline_sufficient",
        "mutated_sufficient",
        "baseline_retrieved_expected_memory_ids",
        "baseline_missing_expected_memory_ids",
        "mutated_retrieved_expected_memory_ids",
        "mutated_missing_expected_memory_ids",
    }


def test_assessment_canonicalizes_expected_and_partition_ids() -> None:
    assessment = PairedRetrievalChallengeAssessment(
        case_id="case",
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
        outcome=RetrievalChallengeOutcome.RESILIENT,
        baseline_request_id="baseline",
        mutated_request_id="mutated",
        query_sha256=_sha(QUERY_TEXT),
        expected_memory_ids=("m2", "m1"),
        top_k=2,
        retriever_id="retriever",
        retriever_version="1",
        baseline_sufficient=True,
        mutated_sufficient=True,
        baseline_retrieved_expected_memory_ids=("m2", "m1"),
        baseline_missing_expected_memory_ids=(),
        mutated_retrieved_expected_memory_ids=("m2", "m1"),
        mutated_missing_expected_memory_ids=(),
    )

    assert assessment.expected_memory_ids == ("m1", "m2")
    assert assessment.baseline_retrieved_expected_memory_ids == ("m1", "m2")
    assert assessment.mutated_retrieved_expected_memory_ids == ("m1", "m2")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("case_id", " ", "must not be blank"),
        ("baseline_request_id", "", "must not be blank"),
        ("expected_memory_ids", (), "must not be empty"),
        ("expected_memory_ids", ("m1", "m1"), "must be unique"),
        (
            "baseline_retrieved_expected_memory_ids",
            (),
            "baseline sufficiency partition",
        ),
        ("baseline_sufficient", False, "baseline sufficiency partition"),
        ("outcome", RetrievalChallengeOutcome.RESILIENT, "outcome must match"),
    ],
)
def test_manually_inconsistent_assessments_fail_validation(
    field: str,
    value: object,
    message: str,
) -> None:
    valid = _pair(
        baseline_hits=(RetrievalHit(memory_id="m1", rank=1),),
        mutated_hits=(),
    )
    payload = valid.model_dump()
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        PairedRetrievalChallengeAssessment.model_validate(payload)


def test_score_and_usage_jitter_have_no_effect() -> None:
    first = assess_paired_retrieval_challenge(
        _observation(
            request_id="baseline",
            hits=(RetrievalHit(memory_id="m1", rank=1, score=0.01),),
            usage=RetrievalUsage(retrieval_calls=1, candidate_count=1),
        ),
        _observation(
            request_id="mutated",
            hits=(),
            usage=RetrievalUsage(retrieval_calls=1, candidate_count=0),
        ),
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
        case_id="case",
    )
    second = assess_paired_retrieval_challenge(
        _observation(
            request_id="baseline",
            hits=(RetrievalHit(memory_id="m1", rank=1, score=999),),
            usage=RetrievalUsage(retrieval_calls=8, candidate_count=200),
        ),
        _observation(
            request_id="mutated",
            hits=(),
            usage=RetrievalUsage(retrieval_calls=7, candidate_count=99),
        ),
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
        case_id="case",
    )

    assert first == second
    assert "score" not in PairedRetrievalChallengeAssessment.model_fields
    assert "usage" not in PairedRetrievalChallengeAssessment.model_fields


def test_non_target_result_changes_have_no_effect() -> None:
    first = assess_paired_retrieval_challenge(
        _observation(
            request_id="baseline",
            hits=(RetrievalHit(memory_id="m1", rank=1),),
        ),
        _observation(
            request_id="mutated",
            hits=(RetrievalHit(memory_id="ordinary-a", rank=1),),
        ),
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
        case_id="case",
    )
    second = assess_paired_retrieval_challenge(
        _observation(
            request_id="baseline",
            hits=(
                RetrievalHit(memory_id="ordinary-b", rank=1),
                RetrievalHit(memory_id="m1", rank=2),
            ),
        ),
        _observation(
            request_id="mutated",
            hits=(
                RetrievalHit(memory_id="ordinary-c", rank=1),
                RetrievalHit(memory_id="ordinary-d", rank=2),
            ),
        ),
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
        case_id="case",
    )

    assert first == second


def test_target_rank_within_top_k_has_no_effect() -> None:
    rank_one = _pair(
        baseline_hits=(RetrievalHit(memory_id="m1", rank=1),),
        mutated_hits=(),
    )
    rank_three = _pair(
        baseline_hits=(
            RetrievalHit(memory_id="other-1", rank=1),
            RetrievalHit(memory_id="other-2", rank=2),
            RetrievalHit(memory_id="m1", rank=3),
        ),
        mutated_hits=(),
    )

    assert rank_one == rank_three


def test_all_and_any_can_produce_different_multi_target_outcomes() -> None:
    baseline_hits = (RetrievalHit(memory_id="m1", rank=1),)
    all_assessment = _pair(
        baseline_hits=baseline_hits,
        mutated_hits=(),
        expected_memory_ids=("m1", "m2"),
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
    )
    any_assessment = _pair(
        baseline_hits=baseline_hits,
        mutated_hits=(),
        expected_memory_ids=("m1", "m2"),
        policy=RetrievalSufficiencyPolicy.ANY_EXPECTED,
    )

    assert all_assessment.outcome is RetrievalChallengeOutcome.BASELINE_INSUFFICIENT
    assert any_assessment.outcome is RetrievalChallengeOutcome.INDUCED_SHADOWING


def test_assessment_serialization_is_deterministic_and_contains_no_runtime_payloads() -> None:
    assessment = _pair(
        baseline_hits=(RetrievalHit(memory_id="m1", rank=1, score=3.14),),
        mutated_hits=(RetrievalHit(memory_id="ordinary-result", rank=1),),
    )

    first = assessment.to_json()
    second = assessment.to_json()
    assert first == second
    assert PairedRetrievalChallengeAssessment.model_validate_json(first).to_json() == first
    assert QUERY_TEXT not in first
    assert "ordinary-result" not in first
    assert "3.14" not in first
    assert '"raw"' not in first


def test_pair_reuses_frozen_sufficiency_assessor_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("memlint.retrieval.challenge")
    original = module.assess_retrieval_sufficiency
    calls: list[tuple[RetrievalObservation, RetrievalSufficiencyPolicy]] = []

    def record_assessment(
        observation: RetrievalObservation,
        *,
        policy: RetrievalSufficiencyPolicy,
    ) -> RetrievalSufficiencyAssessment:
        calls.append((observation, policy))
        return original(observation, policy=policy)

    monkeypatch.setattr(module, "assess_retrieval_sufficiency", record_assessment)
    baseline = _observation(request_id="baseline")
    mutated = _observation(request_id="mutated")

    module.assess_paired_retrieval_challenge(
        baseline,
        mutated,
        policy=RetrievalSufficiencyPolicy.ANY_EXPECTED,
        case_id="case",
    )

    assert calls == [
        (baseline, RetrievalSufficiencyPolicy.ANY_EXPECTED),
        (mutated, RetrievalSufficiencyPolicy.ANY_EXPECTED),
    ]


def test_challenge_module_has_no_store_retriever_checker_mutation_semantic_or_raw_access() -> None:
    tree = ast.parse(
        CHALLENGE_PATH.read_text(encoding="utf-8"),
        filename=str(CHALLENGE_PATH),
    )
    forbidden_modules = {
        "memlint.checkers",
        "memlint.models",
        "memlint.mutations",
        "memlint.semantics",
    }
    forbidden_names = {
        "CheckerResult",
        "Finding",
        "GoldLabel",
        "MutationManifest",
        "NormalizedStore",
        "Retriever",
        "RetrievalProbe",
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if {alias.name for alias in node.names} & forbidden_modules:
                violations.append(f"{node.lineno}:forbidden import")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") in forbidden_modules or any(
                alias.name in forbidden_names for alias in node.names
            ):
                violations.append(f"{node.lineno}:forbidden import")
        elif isinstance(node, ast.Name) and node.id in forbidden_names:
            violations.append(f"{node.lineno}:{node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in {"raw", "retrieve"}:
            violations.append(f"{node.lineno}:{node.attr}")
    assert violations == []


def test_no_paired_cli_command_or_static_checker_choice_exists() -> None:
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if action.dest == "command"
    )
    assert set(subparsers.choices) == {"dump", "mutate", "audit", "retrieval-audit"}
    assert "retrieval_shadowing" not in cli.CHECKER_NAMES


class PartTwoFakeRetriever:
    retriever_id = "part-two-paired-fake"
    retriever_version = "1"

    def __init__(self, response: RetrievalResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def retrieve(self, *, query: str, top_k: int) -> RetrievalResponse:
        self.calls.append({"query": query, "top_k": top_k})
        return self.response


@pytest.mark.parametrize(
    ("baseline_present", "mutated_present", "expected_outcome"),
    [
        (True, False, RetrievalChallengeOutcome.INDUCED_SHADOWING),
        (True, True, RetrievalChallengeOutcome.RESILIENT),
        (False, False, RetrievalChallengeOutcome.BASELINE_INSUFFICIENT),
        (False, True, RetrievalChallengeOutcome.BASELINE_INSUFFICIENT),
    ],
)
def test_part_two_single_target_paired_scenarios_are_policy_compatible(
    baseline_present: bool,
    mutated_present: bool,
    expected_outcome: RetrievalChallengeOutcome,
) -> None:
    base_store = NormalizedStore(
        adapter="test",
        memories=(
            NormalizedMemory(
                id="editor-neovim",
                content="User's favorite editor is Neovim.",
            ),
        ),
    )
    mutation = mutate(
        base_store,
        MutationRequest(
            defect_class=DefectClass.RETRIEVAL_SHADOWING,
            target_memory_id="editor-neovim",
            query="Which editor does the user prefer?",
            distractor_family=DistractorFamily.EDITOR,
        ),
    )
    probe = mutation.manifest.retrieval_probe
    assert probe is not None
    assert len(probe.expected_memory_ids) == 1
    baseline_request = RetrievalAuditRequest(
        request_id="part-two-baseline",
        query=probe.query,
        expected_memory_ids=probe.expected_memory_ids,
        top_k=3,
    )
    mutated_request = RetrievalAuditRequest(
        request_id="part-two-mutated",
        query=probe.query,
        expected_memory_ids=probe.expected_memory_ids,
        top_k=3,
    )
    target_hit = (RetrievalHit(memory_id=probe.expected_memory_ids[0], rank=1),)
    baseline_retriever = PartTwoFakeRetriever(
        RetrievalResponse(
            hits=target_hit if baseline_present else (),
            usage=RetrievalUsage(retrieval_calls=1, candidate_count=int(baseline_present)),
        )
    )
    mutated_retriever = PartTwoFakeRetriever(
        RetrievalResponse(
            hits=target_hit if mutated_present else (),
            usage=RetrievalUsage(retrieval_calls=1, candidate_count=int(mutated_present)),
        )
    )
    baseline_observation = run_retrieval_audit(
        store=base_store,
        request=baseline_request,
        retriever=baseline_retriever,
    )
    mutated_observation = run_retrieval_audit(
        store=mutation.mutated_store,
        request=mutated_request,
        retriever=mutated_retriever,
    )

    outcomes = {
        assess_paired_retrieval_challenge(
            baseline_observation,
            mutated_observation,
            policy=policy,
            case_id="part-two-paired-case",
        ).outcome
        for policy in RetrievalSufficiencyPolicy
    }
    assert outcomes == {expected_outcome}
    assert baseline_observation.request_id != mutated_observation.request_id
    assert baseline_retriever.calls == [{"query": probe.query, "top_k": 3}]
    assert mutated_retriever.calls == [{"query": probe.query, "top_k": 3}]
