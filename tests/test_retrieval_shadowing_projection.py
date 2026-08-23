import ast
import hashlib
import importlib
import inspect
from pathlib import Path
from typing import cast

import pytest

from memlint import cli
from memlint.checkers import (
    CHECKER_RESULT_SCHEMA_VERSION,
    CheckerCost,
    CheckerResult,
    Finding,
    project_retrieval_shadowing_result,
)
from memlint.models import NormalizedMemory, NormalizedStore
from memlint.mutations import DistractorFamily, GoldLabelUnit, MutationRequest, mutate
from memlint.retrieval import (
    RetrievalAuditRequest,
    RetrievalHit,
    RetrievalObservation,
    RetrievalResponse,
    RetrievalSufficiencyPolicy,
    RetrievalUsage,
    run_retrieval_audit,
)
from memlint.taxonomy import DefectClass

PROJECTION_PATH = Path("src/memlint/checkers/retrieval_shadowing.py")
QUERY_TEXT = "Which exact stored preference should retrieval return?"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _observation(
    *,
    request_id: str = "retrieval-case-1",
    query_sha256: str | None = None,
    expected_memory_ids: tuple[str, ...] = ("m1",),
    hits: tuple[RetrievalHit, ...] = (),
    top_k: int | None = None,
    retriever_id: str = "recorded-retriever",
    retriever_version: str = "1",
    usage: RetrievalUsage | None = None,
) -> RetrievalObservation:
    return RetrievalObservation(
        request_id=request_id,
        query_sha256=query_sha256 or _sha(QUERY_TEXT),
        expected_memory_ids=expected_memory_ids,
        top_k=max(1, len(hits)) if top_k is None else top_k,
        retriever_id=retriever_id,
        retriever_version=retriever_version,
        hits=hits,
        usage=usage or RetrievalUsage(retrieval_calls=1, candidate_count=len(hits)),
    )


def _project(
    observation: RetrievalObservation,
    policy: RetrievalSufficiencyPolicy = RetrievalSufficiencyPolicy.ALL_EXPECTED,
) -> CheckerResult:
    return project_retrieval_shadowing_result(observation, policy=policy)


def _only_finding(result: CheckerResult) -> Finding:
    assert len(result.findings) == 1
    return result.findings[0]


def _finding_id(
    observation: RetrievalObservation,
    policy: RetrievalSufficiencyPolicy = RetrievalSufficiencyPolicy.ALL_EXPECTED,
) -> str:
    return _only_finding(_project(observation, policy)).finding_id


def test_projection_uses_frozen_result_identity_and_schema() -> None:
    result = _project(_observation())

    assert CHECKER_RESULT_SCHEMA_VERSION == "0.2"
    assert result.schema_version == "0.2"
    assert result.checker_id == "retrieval_shadowing"
    assert result.checker_version == "1.0"
    assert result.defect_class is DefectClass.RETRIEVAL_SHADOWING


def test_projection_signature_requires_observation_and_explicit_enum_policy() -> None:
    signature = inspect.signature(project_retrieval_shadowing_result)

    assert tuple(signature.parameters) == ("observation", "policy")
    assert signature.parameters["policy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["policy"].default is inspect.Parameter.empty
    assert "assessment" not in signature.parameters
    with pytest.raises(TypeError, match="RetrievalObservation"):
        project_retrieval_shadowing_result(
            cast(RetrievalObservation, object()),
            policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
        )
    with pytest.raises(TypeError, match="RetrievalSufficiencyPolicy"):
        project_retrieval_shadowing_result(
            _observation(),
            policy=cast(RetrievalSufficiencyPolicy, "all_expected"),
        )


@pytest.mark.parametrize(
    "policy",
    [
        RetrievalSufficiencyPolicy.ALL_EXPECTED,
        RetrievalSufficiencyPolicy.ANY_EXPECTED,
    ],
)
def test_sufficient_case_emits_zero_findings(
    policy: RetrievalSufficiencyPolicy,
) -> None:
    result = _project(
        _observation(hits=(RetrievalHit(memory_id="m1", rank=1),)),
        policy,
    )

    assert result.findings == ()
    assert result.stats.findings_emitted == 0


def test_insufficient_case_emits_exactly_one_case_level_finding() -> None:
    result = _project(
        _observation(expected_memory_ids=("m1", "m2", "m3"), hits=()),
        RetrievalSufficiencyPolicy.ANY_EXPECTED,
    )

    finding = _only_finding(result)
    assert finding.memory_ids == ("m1", "m2", "m3")
    assert result.stats.findings_emitted == 1


def test_finding_memory_ids_are_only_missing_expected_targets() -> None:
    result = _project(
        _observation(
            expected_memory_ids=("m1", "m2"),
            hits=(RetrievalHit(memory_id="m1", rank=1),),
        ),
        RetrievalSufficiencyPolicy.ALL_EXPECTED,
    )

    finding = _only_finding(result)
    assert finding.memory_ids == ("m2",)
    assert "m1" not in finding.memory_ids
    evidence_data = finding.evidence[0].model_dump(mode="json")["data"]
    assert evidence_data["expected_memory_ids"] == ["m1", "m2"]
    assert evidence_data["retrieved_expected_memory_ids"] == ["m1"]
    assert evidence_data["missing_expected_memory_ids"] == ["m2"]


def test_finding_confidence_means_deterministic_policy_failure() -> None:
    finding = _only_finding(_project(_observation()))
    assert finding.confidence == 1.0


def test_finding_has_one_exact_minimal_evidence_item() -> None:
    observation = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(
            RetrievalHit(memory_id="non-target", rank=1, score=0.9),
            RetrievalHit(memory_id="m1", rank=2, score=0.1),
        ),
        top_k=3,
        usage=RetrievalUsage(retrieval_calls=2, candidate_count=40),
    )
    finding = _only_finding(_project(observation))

    assert len(finding.evidence) == 1
    evidence = finding.evidence[0]
    assert evidence.kind == "insufficient_retrieval"
    assert evidence.message == (
        "Recorded retrieval did not satisfy the declared target sufficiency policy."
    )
    assert set(evidence.data) == {
        "request_id",
        "query_sha256",
        "policy",
        "top_k",
        "retriever_id",
        "retriever_version",
        "expected_memory_ids",
        "retrieved_expected_memory_ids",
        "missing_expected_memory_ids",
    }
    data = evidence.model_dump(mode="json")["data"]
    assert data == {
        "request_id": "retrieval-case-1",
        "query_sha256": _sha(QUERY_TEXT),
        "policy": "all_expected",
        "top_k": 3,
        "retriever_id": "recorded-retriever",
        "retriever_version": "1",
        "expected_memory_ids": ["m1", "m2"],
        "retrieved_expected_memory_ids": ["m1"],
        "missing_expected_memory_ids": ["m2"],
    }


def test_evidence_excludes_query_content_scores_usage_non_targets_and_ranks() -> None:
    observation = _observation(
        hits=(RetrievalHit(memory_id="ordinary-result", rank=1, score=987.5),),
        top_k=4,
        usage=RetrievalUsage(retrieval_calls=9, candidate_count=87),
    )
    evidence_json = _only_finding(_project(observation)).evidence[0].model_dump_json()

    assert QUERY_TEXT not in evidence_json
    assert "memory content" not in evidence_json
    assert "ordinary-result" not in evidence_json
    assert "score" not in evidence_json
    assert "retrieval_calls" not in evidence_json
    assert "candidate_count" not in evidence_json
    assert '"rank"' not in evidence_json


@pytest.mark.parametrize("target_present", [False, True])
def test_result_accounting_uses_zero_model_cost_and_runtime_stats(
    target_present: bool,
) -> None:
    hits = (
        (RetrievalHit(memory_id="m1", rank=1),)
        if target_present
        else (RetrievalHit(memory_id="other", rank=1),)
    )
    result = _project(
        _observation(
            expected_memory_ids=("m1", "m2"),
            hits=hits,
            top_k=3,
            usage=RetrievalUsage(retrieval_calls=4, candidate_count=25),
        )
    )

    assert result.cost == CheckerCost()
    assert result.cost.model_calls == 0
    assert result.cost.input_tokens == 0
    assert result.cost.output_tokens == 0
    assert result.stats.memories_scanned == 0
    assert result.stats.details == {
        "retrieval_cases_assessed": 1,
        "expected_targets": 2,
        "retrieved_expected_targets": int(target_present),
        "missing_expected_targets": 2 - int(target_present),
        "retrieval_calls": 4,
        "candidate_count": 25,
        "hits_observed": 1,
    }


def test_score_jitter_does_not_change_finding_id() -> None:
    first = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(RetrievalHit(memory_id="m1", rank=1, score=0.01),),
    )
    second = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(RetrievalHit(memory_id="m1", rank=1, score=999),),
    )

    assert _finding_id(first) == _finding_id(second)


def test_usage_jitter_changes_stats_but_not_finding_id() -> None:
    first = _observation(
        usage=RetrievalUsage(retrieval_calls=1, candidate_count=2),
    )
    second = _observation(
        usage=RetrievalUsage(retrieval_calls=8, candidate_count=200),
    )

    first_result = _project(first)
    second_result = _project(second)
    assert _only_finding(first_result).finding_id == _only_finding(second_result).finding_id
    assert first_result.stats.details != second_result.stats.details


def test_non_target_result_composition_does_not_change_finding_id() -> None:
    first = _observation(
        hits=(RetrievalHit(memory_id="m2", rank=1),),
        top_k=3,
    )
    second = _observation(
        hits=(
            RetrievalHit(memory_id="m3", rank=1),
            RetrievalHit(memory_id="m4", rank=2),
        ),
        top_k=3,
    )

    assert _finding_id(first) == _finding_id(second)


def test_target_membership_change_changes_finding_id() -> None:
    retrieved_one = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(RetrievalHit(memory_id="m1", rank=1),),
    )
    retrieved_none = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(),
    )

    assert _finding_id(retrieved_one) != _finding_id(retrieved_none)


def test_policy_change_changes_id_when_both_are_insufficient() -> None:
    observation = _observation(expected_memory_ids=("m1", "m2"), hits=())

    all_id = _finding_id(observation, RetrievalSufficiencyPolicy.ALL_EXPECTED)
    any_id = _finding_id(observation, RetrievalSufficiencyPolicy.ANY_EXPECTED)
    assert all_id != any_id


def test_partial_multi_target_result_emits_for_all_but_not_any() -> None:
    observation = _observation(
        expected_memory_ids=("m1", "m2"),
        hits=(RetrievalHit(memory_id="m1", rank=1),),
    )

    assert len(_project(observation, RetrievalSufficiencyPolicy.ALL_EXPECTED).findings) == 1
    assert _project(observation, RetrievalSufficiencyPolicy.ANY_EXPECTED).findings == ()


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("retriever_version", "2"),
        ("query_sha256", _sha("different query")),
        ("top_k", 2),
        ("request_id", "retrieval-case-2"),
    ],
)
def test_operational_identity_changes_finding_id(
    changed_field: str,
    changed_value: object,
) -> None:
    baseline = _observation(top_k=1)
    changes: dict[str, object] = {changed_field: changed_value}
    changed = _observation(**changes)  # type: ignore[arg-type]

    assert _finding_id(baseline) != _finding_id(changed)


def test_checker_result_serialization_is_deterministic_and_private() -> None:
    observation = _observation(
        hits=(RetrievalHit(memory_id="ordinary-result", rank=1, score=3.14),),
        top_k=2,
    )

    first = _project(observation).to_json()
    second = _project(observation).to_json()
    assert first == second
    assert CheckerResult.model_validate_json(first).to_json() == first
    assert QUERY_TEXT not in first
    assert "ordinary-result" not in first
    assert "3.14" not in first
    assert '"raw"' not in first


def test_projection_recomputes_assessment_internally(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("memlint.checkers.retrieval_shadowing")
    original = module.assess_retrieval_sufficiency
    calls: list[tuple[RetrievalObservation, RetrievalSufficiencyPolicy]] = []

    def recording_assessment(
        observation: RetrievalObservation,
        *,
        policy: RetrievalSufficiencyPolicy,
    ) -> object:
        calls.append((observation, policy))
        return original(observation, policy=policy)

    monkeypatch.setattr(module, "assess_retrieval_sufficiency", recording_assessment)
    observation = _observation()
    result = module.project_retrieval_shadowing_result(
        observation,
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
    )

    assert len(result.findings) == 1
    assert calls == [(observation, RetrievalSufficiencyPolicy.ALL_EXPECTED)]


def test_projection_module_has_no_store_retriever_mutation_or_raw_dependency() -> None:
    tree = ast.parse(
        PROJECTION_PATH.read_text(encoding="utf-8"),
        filename=str(PROJECTION_PATH),
    )
    forbidden_modules = {"memlint.models", "memlint.mutations", "memlint.semantics"}
    forbidden_names = {
        "GoldLabel",
        "MutationManifest",
        "MutationRequest",
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


def test_no_checker_class_or_cli_choice_was_added() -> None:
    tree = ast.parse(
        PROJECTION_PATH.read_text(encoding="utf-8"),
        filename=str(PROJECTION_PATH),
    )
    assert not any(
        isinstance(node, ast.ClassDef) and node.name == "RetrievalShadowingChecker"
        for node in ast.walk(tree)
    )
    assert "retrieval_shadowing" not in cli.CHECKER_NAMES


class PartTwoFakeRetriever:
    retriever_id = "part-two-fake"
    retriever_version = "1"

    def __init__(self, response: RetrievalResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def retrieve(self, *, query: str, top_k: int) -> RetrievalResponse:
        self.calls.append({"query": query, "top_k": top_k})
        return self.response


def test_part_two_single_target_projects_to_zero_or_one_case_finding() -> None:
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
    assert mutation.manifest.gold_label.unit is GoldLabelUnit.RETRIEVAL_CASE
    request = RetrievalAuditRequest(
        request_id="part-two-projection-cross-check",
        query=probe.query,
        expected_memory_ids=probe.expected_memory_ids,
        top_k=2,
    )
    target_returned_retriever = PartTwoFakeRetriever(
        RetrievalResponse(
            hits=(RetrievalHit(memory_id=probe.expected_memory_ids[0], rank=1),),
            usage=RetrievalUsage(retrieval_calls=1, candidate_count=1),
        )
    )
    target_absent_retriever = PartTwoFakeRetriever(
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

    for policy in RetrievalSufficiencyPolicy:
        assert project_retrieval_shadowing_result(
            target_returned,
            policy=policy,
        ).findings == ()
        absent_result = project_retrieval_shadowing_result(
            target_absent,
            policy=policy,
        )
        assert len(absent_result.findings) == 1
        assert absent_result.findings[0].memory_ids == mutation.manifest.gold_label.memory_ids

    assert target_returned_retriever.calls == [{"query": probe.query, "top_k": 2}]
    assert target_absent_retriever.calls == [{"query": probe.query, "top_k": 2}]
