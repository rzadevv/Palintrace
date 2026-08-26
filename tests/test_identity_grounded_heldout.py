from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from memlint.evaluation import identity_grounded_heldout as heldout
from memlint.semantics import SemanticJudgment, SemanticRelation, SemanticUsage

FIXTURE = Path("tests/fixtures/unsupported_identity_grounded_heldout_v0.1.json")
DEVELOPMENT_FIXTURE = Path("tests/fixtures/unsupported_identity_probe_v0.1.json")


class _PreregisteredFakeJudge:
    judge_id = f"hf-nli:{heldout.HELDOUT_MODEL_ID}"
    judge_version = heldout.HELDOUT_MODEL_REVISION

    def __init__(
        self,
        spec: heldout.HeldoutSpec,
        *,
        baseline_clean: SemanticRelation = SemanticRelation.NEUTRAL,
        candidate_clean: SemanticRelation = SemanticRelation.ENTAILMENT,
        unsupported: SemanticRelation = SemanticRelation.CONTRADICTION,
        candidate_unsupported: SemanticRelation | None = None,
        all_entailment: bool = False,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.sensitive_clean = {
            case.clean_hypothesis
            for case in spec.semantic_cases
            if case.case_kind is heldout.SemanticCaseKind.IDENTITY_SENSITIVE
        }
        self.control_clean = {
            case.clean_hypothesis
            for case in spec.semantic_cases
            if case.case_kind is heldout.SemanticCaseKind.IDENTITY_FREE_CONTROL
        }
        self.unsupported = {case.unsupported_hypothesis for case in spec.semantic_cases}
        self.baseline_clean = baseline_clean
        self.candidate_clean = candidate_clean
        self.unsupported_relation = unsupported
        self.candidate_unsupported = candidate_unsupported or unsupported
        self.all_entailment = all_entailment

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        self.calls.append((premise, hypothesis))
        grounded = premise.startswith("The speaker is ")
        if self.all_entailment or hypothesis in self.control_clean:
            relation = SemanticRelation.ENTAILMENT
        elif hypothesis in self.sensitive_clean:
            relation = self.candidate_clean if grounded else self.baseline_clean
        elif hypothesis in self.unsupported:
            relation = (
                self.candidate_unsupported if grounded else self.unsupported_relation
            )
        else:
            relation = SemanticRelation.ENTAILMENT
        return SemanticJudgment(
            relation=relation,
            score=0.91,
            usage=SemanticUsage(model_calls=1, input_tokens=11, output_tokens=0),
        )


def _spec() -> heldout.HeldoutSpec:
    return heldout.preflight_heldout_fixture(FIXTURE)


def _execute(
    judge: _PreregisteredFakeJudge,
    spec: heldout.HeldoutSpec,
) -> heldout.HeldoutExecutionResult:
    return heldout.execute_heldout(
        spec=spec,
        semantic_judge=judge,
        protected_hashes_valid=True,
        candidate_nonpublic=True,
        candidate_noncli=True,
    )


def test_fixture_sha_matrix_domains_and_freshness() -> None:
    spec = _spec()
    heldout.validate_freshness(spec, DEVELOPMENT_FIXTURE)
    assert heldout.sha256_file(FIXTURE) == heldout.HELDOUT_FIXTURE_SHA256
    assert len(spec.semantic_cases) == 40
    assert len(spec.coverage_cases) == 6
    assert {case.case_id for case in spec.semantic_cases}.isdisjoint(
        {
            case["case_id"]
            for case in json.loads(DEVELOPMENT_FIXTURE.read_text())["cases"]
        }
    )
    assert {case.person_name for case in spec.semantic_cases}.isdisjoint(
        {
            case["person_name"]
            for case in json.loads(DEVELOPMENT_FIXTURE.read_text())["cases"]
        }
    )
    assert {case.domain for case in spec.semantic_cases} == heldout._REQUIRED_DOMAINS


def test_fixture_hash_is_verified_before_parsing(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_bytes(FIXTURE.read_bytes() + b"\n")
    with pytest.raises(heldout.HeldoutInputError, match="fixture SHA mismatch"):
        heldout.preflight_heldout_fixture(changed)


def test_exact_substitution_and_condition_mapping_are_frozen() -> None:
    spec = _spec()
    for case in spec.semantic_cases:
        assert case.unsupported_hypothesis == case.clean_hypothesis.replace(
            case.source_value, case.replacement_value, 1
        )
        assert case.source_text.count(case.source_value) == 1
        assert case.clean_hypothesis.count(case.source_value) == 1
    assert [condition.condition for condition in spec.conditions] == list(
        heldout.ConditionName
    )
    assert spec.conditions[0].checker_id == "unsupported_claim"
    assert spec.conditions[1].checker_id == "unsupported_claim_identity_grounded"


def test_coverage_resolution_matrix_without_semantic_execution() -> None:
    spec = _spec()
    heldout.validate_coverage_contract(spec)
    assert [case.expected_identity_status.value for case in spec.coverage_cases].count(
        "resolved"
    ) == 2
    assert [case.expected_identity_status.value for case in spec.coverage_cases].count(
        "unavailable"
    ) == 3
    assert [case.expected_identity_status.value for case in spec.coverage_cases].count(
        "conflict"
    ) == 1


def test_supportive_fake_execution_has_exact_metrics_gates_and_order() -> None:
    spec = _spec()
    judge = _PreregisteredFakeJudge(spec)
    result = _execute(judge, spec)

    sensitive = result.identity_sensitive
    assert sensitive.baseline_clean_entailments == 0
    assert sensitive.candidate_clean_entailments == 30
    assert sensitive.baseline_false_alerts == 30
    assert sensitive.candidate_false_alerts == 0
    assert sensitive.clean_rescues == 30
    assert sensitive.clean_regressions == 0
    assert sensitive.baseline_unsupported_detections == 30
    assert sensitive.candidate_unsupported_detections == 30
    assert sensitive.unsupported_detected_to_missed == 0
    assert result.identity_free_controls.clean_exact_relation_changes == 0
    assert result.identity_free_controls.unsupported_exact_relation_changes == 0
    assert result.gates.clean_selectivity is heldout.GateState.PASS
    assert result.gates.unsupported_safety is heldout.GateState.PASS
    assert result.gates.identity_free_stability is heldout.GateState.PASS
    assert result.gates.abstention_contract is heldout.GateState.PASS
    assert result.gates.regression_privacy is heldout.GateState.PASS
    assert result.interpretation is heldout.FinalInterpretation.SUPPORTS_CANDIDATE
    assert [
        (
            trial.case_id,
            trial.hypothesis_kind,
            trial.condition,
        )
        for trial in result.semantic_trials[:4]
    ] == [
        ("H6GC-C01", heldout.HypothesisKind.CLEAN, heldout.ConditionName.BASELINE_PLAIN),
        ("H6GC-C01", heldout.HypothesisKind.CLEAN, heldout.ConditionName.IDENTITY_GROUNDED),
        (
            "H6GC-C01",
            heldout.HypothesisKind.UNSUPPORTED,
            heldout.ConditionName.BASELINE_PLAIN,
        ),
        (
            "H6GC-C01",
            heldout.HypothesisKind.UNSUPPORTED,
            heldout.ConditionName.IDENTITY_GROUNDED,
        ),
    ]


def test_coverage_counts_abstentions_and_no_calls_for_unavailable_or_conflict() -> None:
    spec = _spec()
    judge = _PreregisteredFakeJudge(spec)
    result = _execute(judge, spec)

    assert len(judge.calls) == 162
    assert result.coverage.total_candidate_memories == 86
    assert result.coverage.declared_memories == 86
    assert result.coverage.evidence_resolvable_memories == 86
    assert result.coverage.identity_resolved == 82
    assert result.coverage.identity_unavailable == 3
    assert result.coverage.identity_conflict == 1
    assert result.coverage.assessed_memories == 82
    assert result.coverage.abstained_memories == 4
    assert result.coverage.semantic_model_calls == 82
    abstained = [
        trial
        for trial in result.coverage_trials
        if trial.identity_status is not heldout.SpeakerIdentityResolutionStatus.RESOLVED
    ]
    assert len(abstained) == 4
    assert all(trial.model_calls == 0 and not trial.assessed for trial in abstained)
    assert all(not trial.finding_ids and not trial.alert for trial in abstained)
    forbidden_hypotheses = {
        case.memory_content
        for case in spec.coverage_cases
        if case.expected_identity_status
        is not heldout.SpeakerIdentityResolutionStatus.RESOLVED
    }
    assert forbidden_hypotheses.isdisjoint({hypothesis for _, hypothesis in judge.calls})


def test_result_is_deterministic_and_private() -> None:
    spec = _spec()
    first = _execute(_PreregisteredFakeJudge(spec), spec)
    second = _execute(_PreregisteredFakeJudge(spec), spec)
    assert first.to_json() == second.to_json()
    serialized = first.to_json()
    assert all(literal not in serialized for literal in heldout.privacy_literals(spec))
    assert "premise_sha256" in serialized
    assert "hypothesis_sha256" in serialized
    assert "speaker_label" not in serialized
    assert "source_text" not in serialized
    assert "memory_content" not in serialized


def test_interpretation_is_inconclusive_when_baseline_failure_is_not_reproduced() -> None:
    spec = _spec()
    result = _execute(_PreregisteredFakeJudge(spec, all_entailment=True), spec)
    assert result.identity_sensitive.baseline_false_alerts == 0
    assert result.gates.baseline_failure_reproduced is False
    assert result.interpretation is (
        heldout.FinalInterpretation.INCONCLUSIVE_BASELINE_FAILURE_NOT_REPRODUCED
    )


def test_interpretation_rejects_candidate_when_safety_gate_fails() -> None:
    spec = _spec()
    result = _execute(
        _PreregisteredFakeJudge(
            spec,
            candidate_unsupported=SemanticRelation.ENTAILMENT,
        ),
        spec,
    )
    assert result.gates.baseline_failure_reproduced is True
    assert result.gates.unsupported_safety is heldout.GateState.FAIL
    assert result.interpretation is heldout.FinalInterpretation.DOES_NOT_SUPPORT_CANDIDATE


def test_result_rejects_tampered_metric_arithmetic() -> None:
    spec = _spec()
    result = _execute(_PreregisteredFakeJudge(spec), spec)
    changed_summary = result.identity_sensitive.model_copy(
        update={"clean_rescues": result.identity_sensitive.clean_rescues - 1}
    )
    with pytest.raises(ValidationError, match="metrics do not match trials"):
        heldout.HeldoutExecutionResult.model_validate(
            {
                **result.model_dump(),
                "identity_sensitive": changed_summary.model_dump(),
            }
        )


def test_frozen_repository_and_nonpromotion_are_valid() -> None:
    protected, nonpublic, noncli = heldout.validate_frozen_repository(Path.cwd())
    assert protected is True
    assert nonpublic is True
    assert noncli is True
