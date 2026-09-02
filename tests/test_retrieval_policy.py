import ast
import hashlib
import inspect
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import palintrace.retrieval as retrieval_api
from palintrace import cli
from palintrace.checkers import CheckerCost, Finding
from palintrace.models import NormalizedMemory, NormalizedStore
from palintrace.mutations import DistractorFamily, MutationRequest, mutate
from palintrace.retrieval import (
    RetrievalAuditRequest,
    RetrievalHit,
    RetrievalObservation,
    RetrievalResponse,
    RetrievalSufficiencyAssessment,
    RetrievalSufficiencyPolicy,
    RetrievalUsage,
    assess_retrieval_sufficiency,
    run_retrieval_audit,
)
from palintrace.taxonomy import DefectClass

POLICY_PATH = Path("src/palintrace/retrieval/policy.py")
QUERY_TEXT = "Which stored memory answers this exact audit query?"


def _observation(
    *,
    expected_memory_ids: tuple[str, ...] = ("m1",),
    hits: tuple[RetrievalHit, ...] = (),
    usage: RetrievalUsage | None = None,
    top_k: int | None = None,
) -> RetrievalObservation:
    return RetrievalObservation(
        request_id="policy-case-1",
        query_sha256=hashlib.sha256(QUERY_TEXT.encode("utf-8")).hexdigest(),
        expected_memory_ids=expected_memory_ids,
        top_k=max(1, len(hits)) if top_k is None else top_k,
        retriever_id="recorded-retriever",
        retriever_version="1",
        hits=hits,
        usage=usage or RetrievalUsage(retrieval_calls=1, candidate_count=len(hits)),
    )


def _assess_both(
    observation: RetrievalObservation,
) -> tuple[RetrievalSufficiencyAssessment, RetrievalSufficiencyAssessment]:
    return (
        assess_retrieval_sufficiency(
            observation,
            policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
        ),
        assess_retrieval_sufficiency(
            observation,
            policy=RetrievalSufficiencyPolicy.ANY_EXPECTED,
        ),
    )


def test_sufficiency_policy_has_exactly_two_frozen_values_and_no_primary() -> None:
    assert tuple(RetrievalSufficiencyPolicy) == (
        RetrievalSufficiencyPolicy.ALL_EXPECTED,
        RetrievalSufficiencyPolicy.ANY_EXPECTED,
    )
    assert [policy.value for policy in RetrievalSufficiencyPolicy] == [
        "all_expected",
        "any_expected",
    ]
    assert not hasattr(retrieval_api, "PRIMARY_RETRIEVAL_SUFFICIENCY_POLICY")


def test_assessment_requires_an_explicit_keyword_only_policy_without_default() -> None:
    signature = inspect.signature(assess_retrieval_sufficiency)
    policy_parameter = signature.parameters["policy"]

    assert policy_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert policy_parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="policy"):
        assess_retrieval_sufficiency(_observation())  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "invalid_policy",
    [None, "all_expected", "all", "any", "strict", "lenient", 1],
)
def test_assessment_rejects_non_enum_policies(invalid_policy: object) -> None:
    with pytest.raises(TypeError, match="RetrievalSufficiencyPolicy"):
        assess_retrieval_sufficiency(
            _observation(),
            policy=cast(RetrievalSufficiencyPolicy, invalid_policy),
        )


def test_assessment_has_exact_minimal_fields_and_strict_boolean() -> None:
    assessment = assess_retrieval_sufficiency(
        _observation(),
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
    )

    assert set(RetrievalSufficiencyAssessment.model_fields) == {
        "request_id",
        "policy",
        "sufficient",
        "expected_memory_ids",
        "retrieved_expected_memory_ids",
        "missing_expected_memory_ids",
        "top_k",
    }
    assert not isinstance(assessment, Finding)
    payload = assessment.model_dump()
    payload["sufficient"] = 1
    with pytest.raises(ValidationError):
        RetrievalSufficiencyAssessment.model_validate(payload)


def test_assessment_canonicalizes_all_id_tuples() -> None:
    assessment = RetrievalSufficiencyAssessment(
        request_id="case",
        policy=RetrievalSufficiencyPolicy.ANY_EXPECTED,
        sufficient=True,
        expected_memory_ids=("m3", "m1", "m2"),
        retrieved_expected_memory_ids=("m2", "m1"),
        missing_expected_memory_ids=("m3",),
        top_k=3,
    )

    assert assessment.expected_memory_ids == ("m1", "m2", "m3")
    assert assessment.retrieved_expected_memory_ids == ("m1", "m2")
    assert assessment.missing_expected_memory_ids == ("m3",)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_memory_ids", ()),
        ("expected_memory_ids", ("m1", "m1")),
        ("retrieved_expected_memory_ids", ("m1", "m1")),
        ("missing_expected_memory_ids", (" ",)),
    ],
)
def test_assessment_rejects_invalid_memory_id_tuples(
    field: str,
    value: tuple[str, ...],
) -> None:
    payload: dict[str, object] = {
        "request_id": "case",
        "policy": RetrievalSufficiencyPolicy.ALL_EXPECTED,
        "sufficient": True,
        "expected_memory_ids": ("m1",),
        "retrieved_expected_memory_ids": ("m1",),
        "missing_expected_memory_ids": (),
        "top_k": 1,
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        RetrievalSufficiencyAssessment.model_validate(payload)


@pytest.mark.parametrize(
    ("retrieved", "missing", "message"),
    [
        (("m1",), ("m1", "m2"), "disjoint"),
        (("m1",), (), "partition"),
        (("m1", "outside"), ("m2",), "partition"),
    ],
)
def test_assessment_subsets_must_partition_expected_ids(
    retrieved: tuple[str, ...],
    missing: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        RetrievalSufficiencyAssessment(
            request_id="case",
            policy=RetrievalSufficiencyPolicy.ANY_EXPECTED,
            sufficient=True,
            expected_memory_ids=("m1", "m2"),
            retrieved_expected_memory_ids=retrieved,
            missing_expected_memory_ids=missing,
            top_k=2,
        )


def test_assessment_boolean_must_match_selected_policy() -> None:
    with pytest.raises(ValidationError, match="selected retrieval policy"):
        RetrievalSufficiencyAssessment(
            request_id="case",
            policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
            sufficient=True,
            expected_memory_ids=("m1", "m2"),
            retrieved_expected_memory_ids=("m1",),
            missing_expected_memory_ids=("m2",),
            top_k=2,
        )


def test_single_target_present_is_sufficient_under_both_policies() -> None:
    observation = _observation(
        hits=(
            RetrievalHit(memory_id="other", rank=1, score=100),
            RetrievalHit(memory_id="m1", rank=2, score=-5),
        ),
    )

    all_assessment, any_assessment = _assess_both(observation)
    assert all_assessment.sufficient is True
    assert any_assessment.sufficient is True
    assert all_assessment.retrieved_expected_memory_ids == ("m1",)
    assert all_assessment.missing_expected_memory_ids == ()


def test_single_target_absent_is_insufficient_under_both_policies() -> None:
    observation = _observation(
        hits=(RetrievalHit(memory_id="other", rank=1),),
    )

    all_assessment, any_assessment = _assess_both(observation)
    assert all_assessment.sufficient is False
    assert any_assessment.sufficient is False
    assert all_assessment.retrieved_expected_memory_ids == ()
    assert all_assessment.missing_expected_memory_ids == ("m1",)


def test_single_target_policy_decisions_are_always_equivalent() -> None:
    present = _observation(hits=(RetrievalHit(memory_id="m1", rank=1),))
    absent = _observation(hits=())

    for observation in (present, absent):
        all_assessment, any_assessment = _assess_both(observation)
        assert all_assessment.sufficient is any_assessment.sufficient


def test_all_expected_present_is_sufficient_under_both_with_non_target_hits() -> None:
    observation = _observation(
        expected_memory_ids=("m2", "m1"),
        hits=(
            RetrievalHit(memory_id="other", rank=1),
            RetrievalHit(memory_id="m2", rank=3),
            RetrievalHit(memory_id="m1", rank=2),
        ),
    )

    all_assessment, any_assessment = _assess_both(observation)
    assert all_assessment.sufficient is True
    assert any_assessment.sufficient is True
    assert all_assessment.retrieved_expected_memory_ids == ("m1", "m2")
    assert all_assessment.missing_expected_memory_ids == ()


def test_partial_multi_target_observation_diverges_between_all_and_any() -> None:
    observation = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(
            RetrievalHit(memory_id="m1", rank=1),
            RetrievalHit(memory_id="m3", rank=2),
        ),
    )

    all_assessment, any_assessment = _assess_both(observation)
    assert all_assessment.sufficient is False
    assert any_assessment.sufficient is True
    for assessment in (all_assessment, any_assessment):
        assert assessment.retrieved_expected_memory_ids == ("m1",)
        assert assessment.missing_expected_memory_ids == ("m2",)


def test_no_multi_targets_present_is_insufficient_under_both() -> None:
    observation = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(
            RetrievalHit(memory_id="m3", rank=1),
            RetrievalHit(memory_id="m4", rank=2),
        ),
    )

    all_assessment, any_assessment = _assess_both(observation)
    assert all_assessment.sufficient is False
    assert any_assessment.sufficient is False
    assert all_assessment.retrieved_expected_memory_ids == ()
    assert all_assessment.missing_expected_memory_ids == ("m1", "m2")


def test_empty_hits_are_insufficient_under_both_policies() -> None:
    all_assessment, any_assessment = _assess_both(
        _observation(expected_memory_ids=("m1", "m2"), hits=())
    )

    assert all_assessment.sufficient is False
    assert any_assessment.sufficient is False


def test_scores_and_usage_have_zero_effect_on_assessment() -> None:
    first = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(
            RetrievalHit(memory_id="m1", rank=1, score=0.01),
            RetrievalHit(memory_id="other", rank=2, score=0.99),
        ),
        usage=RetrievalUsage(retrieval_calls=1, candidate_count=2),
    )
    second = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(
            RetrievalHit(memory_id="m1", rank=1, score=999),
            RetrievalHit(memory_id="other", rank=2, score=-999),
        ),
        usage=RetrievalUsage(retrieval_calls=8, candidate_count=200),
    )

    first_assessment = assess_retrieval_sufficiency(
        first,
        policy=RetrievalSufficiencyPolicy.ANY_EXPECTED,
    )
    second_assessment = assess_retrieval_sufficiency(
        second,
        policy=RetrievalSufficiencyPolicy.ANY_EXPECTED,
    )

    assert first_assessment == second_assessment
    assert "score" not in RetrievalSufficiencyAssessment.model_fields
    assert "usage" not in RetrievalSufficiencyAssessment.model_fields


def test_original_hit_construction_order_has_no_effect() -> None:
    first = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(
            RetrievalHit(memory_id="m2", rank=2),
            RetrievalHit(memory_id="m1", rank=1),
        ),
    )
    second = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(
            RetrievalHit(memory_id="m1", rank=1),
            RetrievalHit(memory_id="m2", rank=2),
        ),
    )

    assert first == second
    assert assess_retrieval_sufficiency(
        first,
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
    ) == assess_retrieval_sufficiency(
        second,
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
    )


def test_assessment_json_is_deterministic_and_contains_no_query() -> None:
    observation = _observation(
        expected_memory_ids=("m2", "m1"),
        hits=(RetrievalHit(memory_id="m1", rank=1),),
    )
    first = assess_retrieval_sufficiency(
        observation,
        policy=RetrievalSufficiencyPolicy.ANY_EXPECTED,
    ).to_json()
    second = assess_retrieval_sufficiency(
        observation,
        policy=RetrievalSufficiencyPolicy.ANY_EXPECTED,
    ).to_json()

    assert first == second
    assert QUERY_TEXT not in first
    assert "query" not in RetrievalSufficiencyAssessment.model_fields
    assert RetrievalSufficiencyAssessment.model_validate_json(first).to_json() == first


def test_policy_module_has_no_execution_or_forbidden_dependencies() -> None:
    tree = ast.parse(POLICY_PATH.read_text(encoding="utf-8"), filename=str(POLICY_PATH))
    forbidden_modules = {
        "palintrace.checkers",
        "palintrace.models",
        "palintrace.mutations",
        "palintrace.semantics",
    }
    forbidden_names = {
        "Finding",
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


def test_finding_and_checker_cost_schemas_are_unchanged_and_no_checker_exists() -> None:
    assert set(Finding.model_fields) == {
        "finding_id",
        "defect_class",
        "memory_ids",
        "confidence",
        "evidence",
    }
    assert set(CheckerCost.model_fields) == {
        "model_calls",
        "input_tokens",
        "output_tokens",
    }
    assert "RetrievalShadowingChecker" not in {
        node.name
        for path in Path("src/palintrace").rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }
    assert "retrieval_shadowing" not in cli.CHECKER_NAMES


class FakeRetriever:
    retriever_id = "policy-fake"
    retriever_version = "1"

    def __init__(self, response: RetrievalResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def retrieve(self, *, query: str, top_k: int) -> RetrievalResponse:
        self.calls.append({"query": query, "top_k": top_k})
        return self.response


def test_single_target_policies_agree_for_present_and_absent_results() -> None:
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
    request = RetrievalAuditRequest(
        request_id="policy-cross-check",
        query=probe.query,
        expected_memory_ids=probe.expected_memory_ids,
        top_k=2,
    )
    target_returned_retriever = FakeRetriever(
        RetrievalResponse(
            hits=(RetrievalHit(memory_id=probe.expected_memory_ids[0], rank=1),),
            usage=RetrievalUsage(retrieval_calls=1, candidate_count=1),
        )
    )
    target_absent_retriever = FakeRetriever(
        RetrievalResponse(
            hits=(),
            usage=RetrievalUsage(retrieval_calls=1, candidate_count=0),
        )
    )
    target_returned = run_retrieval_audit(
        store=mutation.mutated_store,
        request=request,
        retriever=target_returned_retriever,
    )
    target_absent = run_retrieval_audit(
        store=mutation.mutated_store,
        request=request,
        retriever=target_absent_retriever,
    )

    returned_all, returned_any = _assess_both(target_returned)
    absent_all, absent_any = _assess_both(target_absent)
    assert returned_all.sufficient is True
    assert returned_any.sufficient is True
    assert absent_all.sufficient is False
    assert absent_any.sufficient is False
    assert target_returned_retriever.calls == [{"query": probe.query, "top_k": 2}]
    assert target_absent_retriever.calls == [{"query": probe.query, "top_k": 2}]
