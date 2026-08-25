from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from memlint.evaluation.identity_probe import (
    IDENTITY_PROBE_FIXTURE_SHA256,
    IDENTITY_PROBE_ID,
    IDENTITY_PROBE_MODEL_ID,
    IDENTITY_PROBE_MODEL_REVISION,
    CleanRescueGateState,
    FailurePatternState,
    IdentityProbeCase,
    IdentityProbeCaseKind,
    IdentityProbeConditionName,
    IdentityProbeExecutionResult,
    IdentityProbeHypothesisKind,
    IdentityProbeInterpretation,
    IdentityProbeJudgment,
    IdentityProbeSpec,
    IdentityProbeTransformationKind,
    PrefixStabilityGateState,
    UnsupportedSafetyGateState,
    build_identity_probe_premise,
    evaluate_identity_probe_gates,
    execute_identity_probe,
    identity_probe_hypothesis,
    interpret_identity_probe_gates,
    preflight_identity_probe,
    summarize_identity_probe_judgments,
)
from memlint.evaluation.preflight import preflight_benchmark_v0_1
from memlint.semantics import SemanticJudgment, SemanticRelation, SemanticUsage

FIXTURE = Path("tests/fixtures/unsupported_identity_probe_v0.1.json")
PINNED_JUDGE_ID = f"hf-nli:{IDENTITY_PROBE_MODEL_ID}"


def _fixture() -> IdentityProbeSpec:
    return preflight_identity_probe(FIXTURE)


def _all_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_strings(item)


def _judgments(
    spec: IdentityProbeSpec,
    relation_for: Callable[
        [IdentityProbeCase, IdentityProbeConditionName, IdentityProbeHypothesisKind],
        SemanticRelation,
    ],
) -> tuple[IdentityProbeJudgment, ...]:
    return tuple(
        IdentityProbeJudgment(
            case_id=case.case_id,
            case_kind=case.case_kind,
            condition=condition,
            hypothesis_kind=hypothesis_kind,
            relation=relation_for(case, condition, hypothesis_kind),
            score=0.9,
            judge_id=PINNED_JUDGE_ID,
            judge_version=IDENTITY_PROBE_MODEL_REVISION,
            model_calls=1,
            input_tokens=8,
            output_tokens=0,
        )
        for case in spec.cases
        for condition in IdentityProbeConditionName
        for hypothesis_kind in IdentityProbeHypothesisKind
    )


def _gate_fixture_judgments(
    spec: IdentityProbeSpec,
    *,
    plain_clean_entailments: int,
    grounded_clean_entailments: int,
    plain_unsupported_detected: int = 18,
    grounded_unsupported_detected: int = 18,
    control_clean_relation_changes: int = 0,
    control_detected_to_missed: int = 0,
) -> tuple[IdentityProbeJudgment, ...]:
    sensitive_ids = [
        case.case_id
        for case in spec.cases
        if case.case_kind is IdentityProbeCaseKind.IDENTITY_SENSITIVE
    ]
    control_ids = [
        case.case_id
        for case in spec.cases
        if case.case_kind is IdentityProbeCaseKind.IDENTITY_FREE_CONTROL
    ]
    sensitive_index = {case_id: index for index, case_id in enumerate(sensitive_ids)}
    control_index = {case_id: index for index, case_id in enumerate(control_ids)}

    def relation_for(
        case: IdentityProbeCase,
        condition: IdentityProbeConditionName,
        hypothesis_kind: IdentityProbeHypothesisKind,
    ) -> SemanticRelation:
        if case.case_kind is IdentityProbeCaseKind.IDENTITY_SENSITIVE:
            index = sensitive_index[case.case_id]
            if hypothesis_kind is IdentityProbeHypothesisKind.CLEAN:
                limit = (
                    plain_clean_entailments
                    if condition is IdentityProbeConditionName.PLAIN
                    else grounded_clean_entailments
                )
                return (
                    SemanticRelation.ENTAILMENT
                    if index < limit
                    else SemanticRelation.NEUTRAL
                )
            limit = (
                plain_unsupported_detected
                if condition is IdentityProbeConditionName.PLAIN
                else grounded_unsupported_detected
            )
            return (
                SemanticRelation.CONTRADICTION
                if index < limit
                else SemanticRelation.ENTAILMENT
            )

        index = control_index[case.case_id]
        if hypothesis_kind is IdentityProbeHypothesisKind.CLEAN:
            if (
                condition is IdentityProbeConditionName.SPEAKER_GROUNDED
                and index < control_clean_relation_changes
            ):
                return SemanticRelation.CONTRADICTION
            return SemanticRelation.ENTAILMENT
        if (
            condition is IdentityProbeConditionName.SPEAKER_GROUNDED
            and index < control_detected_to_missed
        ):
            return SemanticRelation.ENTAILMENT
        return SemanticRelation.CONTRADICTION

    return _judgments(spec, relation_for)


class _FakePinnedJudge:
    judge_id = PINNED_JUDGE_ID
    judge_version = IDENTITY_PROBE_MODEL_REVISION

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        self.calls.append((premise, hypothesis))
        hypothesis_kind_is_clean = len(self.calls) % 2 == 1
        return SemanticJudgment(
            relation=(
                SemanticRelation.ENTAILMENT
                if hypothesis_kind_is_clean
                else SemanticRelation.CONTRADICTION
            ),
            score=0.9,
            usage=SemanticUsage(model_calls=1, input_tokens=8, output_tokens=0),
        )


def test_fixture_schema_matrix_balance_and_hash_are_frozen() -> None:
    spec = _fixture()
    assert spec.schema_version == "0.1"
    assert spec.probe_id == IDENTITY_PROBE_ID
    assert spec.split == "development"
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == IDENTITY_PROBE_FIXTURE_SHA256
    assert len(spec.cases) == 24
    assert [case.case_id for case in spec.cases] == sorted(case.case_id for case in spec.cases)
    assert Counter(case.case_kind for case in spec.cases) == Counter(
        {
            IdentityProbeCaseKind.IDENTITY_SENSITIVE: 18,
            IdentityProbeCaseKind.IDENTITY_FREE_CONTROL: 6,
        }
    )
    assert Counter(case.transformation_kind for case in spec.cases) == Counter(
        {
            IdentityProbeTransformationKind.FIRST_PERSON_SUBJECT: 12,
            IdentityProbeTransformationKind.FIRST_PERSON_POSSESSIVE: 6,
            IdentityProbeTransformationKind.ALREADY_NAMED_SUBJECT: 6,
        }
    )
    assert {case.person_name for case in spec.cases} == {"Mireya", "Tomasz", "Yuna"}
    sensitive = [
        case
        for case in spec.cases
        if case.case_kind is IdentityProbeCaseKind.IDENTITY_SENSITIVE
    ]
    assert Counter(case.person_name for case in sensitive) == Counter(
        {"Mireya": 6, "Tomasz": 6, "Yuna": 6}
    )
    assert Counter(case.domain for case in sensitive) == Counter(
        {
            "tool_or_software": 3,
            "location": 3,
            "schedule": 3,
            "device_or_hardware": 3,
            "project_or_employment": 3,
            "preference_or_subscription": 3,
        }
    )
    assert len(spec.cases) * len(spec.conditions) * 2 == 96


def test_every_case_preserves_clean_value_and_changes_one_unsupported_value() -> None:
    spec = _fixture()
    for case in spec.cases:
        assert case.source_text.count(case.source_value) == 1
        assert case.clean_hypothesis.count(case.source_value) == 1
        assert case.replacement_value not in case.source_text
        assert case.replacement_value not in case.clean_hypothesis
        assert case.unsupported_hypothesis == case.clean_hypothesis.replace(
            case.source_value,
            case.replacement_value,
            1,
        )
        assert case.person_name in case.clean_hypothesis
        assert case.person_name in case.unsupported_hypothesis


def test_case_contract_rejects_a_second_or_undeclared_change() -> None:
    case_payload = _fixture().cases[0].model_dump(mode="json")
    case_payload["unsupported_hypothesis"] += " Extra fact."
    with pytest.raises(ValidationError, match="exactly the declared factual value"):
        IdentityProbeCase.model_validate(case_payload)

    case_payload = _fixture().cases[0].model_dump(mode="json")
    case_payload["replacement_value"] = case_payload["source_value"]
    with pytest.raises(ValidationError, match="must differ"):
        IdentityProbeCase.model_validate(case_payload)


def test_fixture_has_no_exact_reuse_or_held_out_people() -> None:
    probe = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = probe["cases"]
    existing_strings: set[str] = set()
    existing_text_parts: list[str] = []
    for fixture_path in Path("tests/fixtures").rglob("*.json"):
        if fixture_path == FIXTURE:
            continue
        text = fixture_path.read_text(encoding="utf-8")
        existing_text_parts.append(text)
        try:
            existing_strings.update(_all_strings(json.loads(text)))
        except json.JSONDecodeError:
            continue
    for field in ("case_id", "source_text", "clean_hypothesis", "unsupported_hypothesis"):
        assert not {case[field] for case in cases} & existing_strings
    existing_text = "\n".join(existing_text_parts).casefold()
    assert all(name.casefold() not in existing_text for name in ("Mireya", "Tomasz", "Yuna"))
    probe_text = FIXTURE.read_text(encoding="utf-8")
    assert all(name not in probe_text for name in ("Ava", "Kenji", "Lina"))


def test_plain_and_grounded_premises_are_exact_for_both_case_kinds() -> None:
    spec = _fixture()
    for case in spec.cases:
        assert (
            build_identity_probe_premise(case, IdentityProbeConditionName.PLAIN)
            == case.source_text
        )
        assert build_identity_probe_premise(
            case,
            IdentityProbeConditionName.SPEAKER_GROUNDED,
        ) == f"The speaker is {case.person_name}.\n{case.source_text}"
        assert (
            identity_probe_hypothesis(case, IdentityProbeHypothesisKind.CLEAN)
            == case.clean_hypothesis
        )
        assert (
            identity_probe_hypothesis(case, IdentityProbeHypothesisKind.UNSUPPORTED)
            == case.unsupported_hypothesis
        )


def test_gate_boundaries_and_supports_h1_path_are_frozen() -> None:
    summary = summarize_identity_probe_judgments(
        _gate_fixture_judgments(
            _fixture(),
            plain_clean_entailments=12,
            grounded_clean_entailments=16,
            plain_unsupported_detected=18,
            grounded_unsupported_detected=17,
            control_clean_relation_changes=1,
            control_detected_to_missed=1,
        )
    )
    gates = evaluate_identity_probe_gates(summary)
    assert gates.failure_pattern is FailurePatternState.REPRODUCED
    assert gates.clean_rescue is CleanRescueGateState.PASS
    assert gates.unsupported_safety is UnsupportedSafetyGateState.PASS
    assert gates.prefix_stability is PrefixStabilityGateState.PASS
    assert interpret_identity_probe_gates(gates) is IdentityProbeInterpretation.SUPPORTS_H1


def test_does_not_support_h1_when_reproduced_failure_misses_any_gate() -> None:
    rescue_failure = summarize_identity_probe_judgments(
        _gate_fixture_judgments(
            _fixture(),
            plain_clean_entailments=12,
            grounded_clean_entailments=15,
        )
    )
    gates = evaluate_identity_probe_gates(rescue_failure)
    assert gates.failure_pattern is FailurePatternState.REPRODUCED
    assert gates.clean_rescue is CleanRescueGateState.FAIL
    assert (
        interpret_identity_probe_gates(gates)
        is IdentityProbeInterpretation.DOES_NOT_SUPPORT_H1
    )

    safety_failure = summarize_identity_probe_judgments(
        _gate_fixture_judgments(
            _fixture(),
            plain_clean_entailments=12,
            grounded_clean_entailments=16,
            grounded_unsupported_detected=16,
        )
    )
    assert (
        evaluate_identity_probe_gates(safety_failure).unsupported_safety
        is UnsupportedSafetyGateState.FAIL
    )

    prefix_failure = summarize_identity_probe_judgments(
        _gate_fixture_judgments(
            _fixture(),
            plain_clean_entailments=12,
            grounded_clean_entailments=16,
            control_clean_relation_changes=2,
        )
    )
    assert (
        evaluate_identity_probe_gates(prefix_failure).prefix_stability
        is PrefixStabilityGateState.FAIL
    )


def test_inconclusive_path_takes_precedence_when_failure_is_not_reproduced() -> None:
    summary = summarize_identity_probe_judgments(
        _gate_fixture_judgments(
            _fixture(),
            plain_clean_entailments=13,
            grounded_clean_entailments=17,
        )
    )
    gates = evaluate_identity_probe_gates(summary)
    assert gates.failure_pattern is FailurePatternState.WEAK_OR_NOT_REPRODUCED
    assert (
        interpret_identity_probe_gates(gates)
        is IdentityProbeInterpretation.INCONCLUSIVE_FAILURE_NOT_REPRODUCED
    )


def test_fake_judge_execution_requires_96_and_serializes_deterministically() -> None:
    judge = _FakePinnedJudge()
    result = execute_identity_probe(spec=_fixture(), semantic_judge=judge)
    assert len(judge.calls) == 96
    assert len(result.judgments) == 96
    assert result.fixture_sha256 == IDENTITY_PROBE_FIXTURE_SHA256
    assert result.interpretation is IdentityProbeInterpretation.INCONCLUSIVE_FAILURE_NOT_REPRODUCED
    assert result.to_json() == IdentityProbeExecutionResult.model_validate_json(
        result.to_json()
    ).to_json()
    keys = [
        (item.case_id, item.condition.value, item.hypothesis_kind.value)
        for item in result.judgments
    ]
    assert keys == sorted(keys)
    result_text = result.to_json().lower()
    assert "rationale" not in result_text
    assert "chain_of_thought" not in result_text
    assert "threshold" not in result_text

    incomplete = result.model_dump(mode="json")
    incomplete["judgments"] = incomplete["judgments"][:-1]
    with pytest.raises(ValidationError, match="96 judgments"):
        IdentityProbeExecutionResult.model_validate(incomplete)


def test_v01_preflight_and_manifest_hash_remain_frozen() -> None:
    benchmark = preflight_benchmark_v0_1(repository_root=Path.cwd())
    assert benchmark.benchmark_id == "memlint-controlled-v0.1"
    canonical_sha = hashlib.sha256(benchmark.to_json(indent=None).encode("utf-8")).hexdigest()
    assert canonical_sha == "fd11b0d547197495d51684f005ac17c861392891e464d818815e04eb6f37dad0"
    manifest_sha = hashlib.sha256(
        Path("tests/fixtures/benchmark_v0.1.sha256.json").read_bytes()
    ).hexdigest()
    assert manifest_sha == "de4bb8c2076a2c89b7e2df95518ef5588934644b711119fccc8727e0e9ac73fb"


def test_identity_probe_is_evaluation_only_and_has_no_production_or_cli_dependency() -> None:
    for root in (
        Path("src/memlint/checkers"),
        Path("src/memlint/semantics"),
        Path("src/memlint/mutations"),
        Path("src/memlint/retrieval"),
    ):
        for source_path in root.rglob("*.py"):
            assert "identity_probe" not in source_path.read_text(encoding="utf-8")
    assert "identity" not in Path("src/memlint/cli.py").read_text(encoding="utf-8")
    assert not any("threshold" in field for field in IdentityProbeJudgment.model_fields)
    assert not any("threshold" in field for field in IdentityProbeExecutionResult.model_fields)


def test_evaluation_docs_do_not_restore_stale_current_benchmark_status() -> None:
    text = Path("docs/evaluation.md").read_text(encoding="utf-8")
    assert "No final research benchmark has been run." not in text
    assert "Benchmark v0.1 remains `NOT_RUN`." not in text
    assert "execution status at specification freeze time: `NOT_RUN`" in text
