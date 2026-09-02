from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from palintrace.evaluation.semantic_selectivity import (
    SEMANTIC_SELECTIVITY_FIXTURE_SHA256,
    SEMANTIC_SELECTIVITY_MODEL_ID,
    SEMANTIC_SELECTIVITY_MODEL_REVISION,
    SEMANTIC_SELECTIVITY_THRESHOLD_GRID,
    CalibrationSelection,
    CalibrationSelectionStatus,
    CalibrationThresholdEvaluation,
    SemanticSelectivityDecision,
    SemanticSelectivityDomain,
    SemanticSelectivityExecutionResult,
    SemanticSelectivityGate,
    SemanticSelectivityHypothesisKind,
    SemanticSelectivityInterpretation,
    SemanticSelectivityJudgment,
    SemanticSelectivityScenario,
    SemanticSelectivitySpec,
    SemanticSelectivitySplit,
    baseline_decision,
    preflight_semantic_selectivity,
    select_calibration_threshold,
    selective_decision,
    summarize_semantic_selectivity,
)
from palintrace.semantics import SemanticRelation, SemanticUsage

FIXTURE = Path("tests/fixtures/semantic_selectivity_probe_v0.1.json")
PINNED_JUDGE_ID = f"hf-nli:{SEMANTIC_SELECTIVITY_MODEL_ID}"
FIRST_PERSON = re.compile(r"\b(?:i|my|we|our)\b", re.IGNORECASE)


def _fixture() -> SemanticSelectivitySpec:
    return preflight_semantic_selectivity(FIXTURE)


def _all_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_strings(item)


def _rows(
    splits: tuple[SemanticSelectivitySplit, ...],
    relation_score_for: Callable[
        [SemanticSelectivityScenario, SemanticSelectivityHypothesisKind],
        tuple[SemanticRelation, float],
    ],
    *,
    threshold: float | None,
) -> tuple[SemanticSelectivityJudgment, ...]:
    return tuple(
        SemanticSelectivityJudgment(
            scenario_id=scenario.scenario_id,
            hypothesis_kind=kind,
            relation=relation,
            score=score,
            usage=SemanticUsage(model_calls=1, input_tokens=12, output_tokens=0),
            baseline_decision=baseline_decision(relation),
            selected_threshold=threshold,
            selective_decision=(
                None if threshold is None else selective_decision(relation, score, threshold)
            ),
        )
        for scenario in _fixture().scenarios
        if scenario.split in splits
        for kind in SemanticSelectivityHypothesisKind
        for relation, score in (relation_score_for(scenario, kind),)
    )


def _calibration_for_threshold_half(
    *,
    threshold: float | None,
) -> tuple[SemanticSelectivityJudgment, ...]:
    return _rows(
        (SemanticSelectivitySplit.CALIBRATION,),
        lambda _scenario, kind: (
            (SemanticRelation.ENTAILMENT, 0.96)
            if kind is SemanticSelectivityHypothesisKind.CLEAN
            else (SemanticRelation.CONTRADICTION, 0.96)
        ),
        threshold=threshold,
    )


def _selected_half() -> CalibrationSelection:
    return select_calibration_threshold(_calibration_for_threshold_half(threshold=None))


def _completed_rows(
    *,
    baseline_clean_alerts: int,
    selective_clean_alerts: int,
    baseline_unsupported_alerts: int = 24,
    selective_unsupported_alerts: int = 23,
) -> tuple[SemanticSelectivityJudgment, ...]:
    clean_index = 0
    unsupported_index = 0

    def relation_score(
        _scenario: SemanticSelectivityScenario,
        kind: SemanticSelectivityHypothesisKind,
    ) -> tuple[SemanticRelation, float]:
        nonlocal clean_index, unsupported_index
        if kind is SemanticSelectivityHypothesisKind.CLEAN:
            index = clean_index
            clean_index += 1
            if index < selective_clean_alerts:
                return SemanticRelation.NEUTRAL, 0.9
            if index < baseline_clean_alerts:
                return SemanticRelation.NEUTRAL, 0.4
            return SemanticRelation.ENTAILMENT, 0.9
        index = unsupported_index
        unsupported_index += 1
        if index < selective_unsupported_alerts:
            return SemanticRelation.CONTRADICTION, 0.9
        if index < baseline_unsupported_alerts:
            return SemanticRelation.CONTRADICTION, 0.4
        return SemanticRelation.ENTAILMENT, 0.9

    confirmatory = _rows(
        (SemanticSelectivitySplit.CONFIRMATORY,),
        relation_score,
        threshold=0.5,
    )
    return (*_calibration_for_threshold_half(threshold=0.5), *confirmatory)


def test_fixture_inventory_balance_hash_and_future_judgment_count_are_frozen() -> None:
    spec = _fixture()
    assert spec.schema_version == "0.1"
    assert spec.probe_id == "semantic-selectivity-probe-v0.1"
    assert spec.hypothesis_id == "H3"
    assert spec.experiment_type == "fresh_calibration_and_confirmatory_synthetic"
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SEMANTIC_SELECTIVITY_FIXTURE_SHA256
    assert len(spec.scenarios) == 48
    assert Counter(item.split for item in spec.scenarios) == Counter(
        {SemanticSelectivitySplit.CALIBRATION: 24, SemanticSelectivitySplit.CONFIRMATORY: 24}
    )
    assert Counter((item.split, item.domain) for item in spec.scenarios) == Counter(
        {
            (split, domain): 4
            for split in SemanticSelectivitySplit
            for domain in SemanticSelectivityDomain
        }
    )
    assert [item.scenario_id for item in spec.scenarios] == [
        *(f"H3-CAL-{index:02d}" for index in range(1, 25)),
        *(f"H3-CONF-{index:02d}" for index in range(1, 25)),
    ]
    assert len(spec.scenarios) * 2 == 96


def test_every_case_is_third_person_and_one_value_substitution() -> None:
    for scenario in _fixture().scenarios:
        texts = (
            scenario.premise,
            scenario.clean_hypothesis,
            scenario.unsupported_hypothesis,
        )
        assert all(FIRST_PERSON.search(text) is None for text in texts)
        assert all(text.count(scenario.subject_name) == 1 for text in texts)
        assert scenario.premise.count(scenario.source_value) == 1
        assert scenario.clean_hypothesis.count(scenario.source_value) == 1
        assert scenario.replacement_value not in scenario.premise
        assert scenario.replacement_value not in scenario.clean_hypothesis
        assert scenario.unsupported_hypothesis == scenario.clean_hypothesis.replace(
            scenario.source_value,
            scenario.replacement_value,
            1,
        )


def test_probe_text_has_zero_exact_reuse_from_existing_fixtures_or_examples() -> None:
    existing_strings: set[str] = set()
    paths = [*Path("tests/fixtures").rglob("*.json"), *Path("examples").rglob("*.json")]
    for path in paths:
        if path == FIXTURE:
            continue
        try:
            existing_strings.update(_all_strings(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    for scenario in _fixture().scenarios:
        assert scenario.premise not in existing_strings
        assert scenario.clean_hypothesis not in existing_strings
        assert scenario.unsupported_hypothesis not in existing_strings


def test_case_contract_rejects_first_person_and_extra_unsupported_change() -> None:
    payload = _fixture().scenarios[0].model_dump(mode="json")
    payload["premise"] = "I use CobaltDesk."
    with pytest.raises(ValidationError, match="first-person"):
        SemanticSelectivityScenario.model_validate(payload)

    payload = _fixture().scenarios[0].model_dump(mode="json")
    payload["unsupported_hypothesis"] += " Extra assertion."
    with pytest.raises(ValidationError, match="replace exactly one factual value"):
        SemanticSelectivityScenario.model_validate(payload)


def test_threshold_grid_and_both_decision_policies_are_exact() -> None:
    assert SEMANTIC_SELECTIVITY_THRESHOLD_GRID == (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    assert baseline_decision(SemanticRelation.ENTAILMENT) is SemanticSelectivityDecision.NO_ALERT
    assert baseline_decision(SemanticRelation.NEUTRAL) is SemanticSelectivityDecision.ALERT
    assert baseline_decision(SemanticRelation.CONTRADICTION) is SemanticSelectivityDecision.ALERT
    assert (
        selective_decision(SemanticRelation.ENTAILMENT, 0.4, 0.8)
        is SemanticSelectivityDecision.NO_ALERT
    )
    assert (
        selective_decision(SemanticRelation.NEUTRAL, 0.8, 0.8)
        is SemanticSelectivityDecision.ALERT
    )
    assert (
        selective_decision(SemanticRelation.CONTRADICTION, 0.79, 0.8)
        is SemanticSelectivityDecision.ABSTAIN
    )


def test_calibration_eligibility_and_lowest_threshold_tie_break() -> None:
    selection = _selected_half()
    assert selection.status is CalibrationSelectionStatus.SELECTED
    assert selection.selected_threshold == 0.5
    assert all(item.eligible for item in selection.evaluations)
    assert all(item.unsupported_alerts == 24 for item in selection.evaluations)


def test_calibration_tie_break_considers_all_four_frozen_keys() -> None:
    evaluations = tuple(
        CalibrationThresholdEvaluation(
            threshold=threshold,
            clean_alerts=(0 if threshold >= 0.7 else 1),
            unsupported_alerts=(24 if threshold in (0.7, 0.8) else 22),
            total_abstains=(3 if threshold == 0.8 else 4),
            eligible=True,
        )
        for threshold in SEMANTIC_SELECTIVITY_THRESHOLD_GRID
    )
    selection = CalibrationSelection(
        status=CalibrationSelectionStatus.SELECTED,
        selected_threshold=0.8,
        evaluations=evaluations,
    )
    assert selection.selected_threshold == 0.8


def test_no_eligible_calibration_threshold_fails_without_confirmation() -> None:
    unsupported_seen = 0

    def relation_score(
        _scenario: SemanticSelectivityScenario,
        kind: SemanticSelectivityHypothesisKind,
    ) -> tuple[SemanticRelation, float]:
        nonlocal unsupported_seen
        if kind is SemanticSelectivityHypothesisKind.CLEAN:
            return SemanticRelation.ENTAILMENT, 0.96
        index = unsupported_seen
        unsupported_seen += 1
        if index < 21:
            return SemanticRelation.CONTRADICTION, 0.96
        return SemanticRelation.ENTAILMENT, 0.96

    calibration = _rows(
        (SemanticSelectivitySplit.CALIBRATION,),
        relation_score,
        threshold=None,
    )
    selection = select_calibration_threshold(calibration)
    assert selection.status is CalibrationSelectionStatus.FAILED
    assert selection.selected_threshold is None
    summary = summarize_semantic_selectivity(calibration, selection)
    assert summary.confirmatory_scenarios == 0
    assert summary.interpretation is SemanticSelectivityInterpretation.DOES_NOT_SUPPORT_H3


@pytest.mark.parametrize(
    ("rows", "expected_interpretation", "expected_failed_gate"),
    [
        (
            _completed_rows(baseline_clean_alerts=6, selective_clean_alerts=3),
            SemanticSelectivityInterpretation.SUPPORTS_H3,
            None,
        ),
        (
            _completed_rows(baseline_clean_alerts=3, selective_clean_alerts=0),
            SemanticSelectivityInterpretation.INCONCLUSIVE_BASELINE_TOO_EASY,
            SemanticSelectivityGate.BASELINE_SELECTIVITY_CHALLENGE_FAIL,
        ),
        (
            _completed_rows(
                baseline_clean_alerts=6,
                selective_clean_alerts=3,
                selective_unsupported_alerts=22,
            ),
            SemanticSelectivityInterpretation.DOES_NOT_SUPPORT_H3,
            SemanticSelectivityGate.UNSUPPORTED_SAFETY_GATE_FAIL,
        ),
        (
            _completed_rows(baseline_clean_alerts=6, selective_clean_alerts=4),
            SemanticSelectivityInterpretation.DOES_NOT_SUPPORT_H3,
            SemanticSelectivityGate.CLEAN_ALERT_REDUCTION_GATE_FAIL,
        ),
        (
            _completed_rows(baseline_clean_alerts=10, selective_clean_alerts=2),
            SemanticSelectivityInterpretation.DOES_NOT_SUPPORT_H3,
            SemanticSelectivityGate.COVERAGE_GATE_FAIL,
        ),
    ],
)
def test_all_gate_paths_and_three_interpretations_are_frozen(
    rows: tuple[SemanticSelectivityJudgment, ...],
    expected_interpretation: SemanticSelectivityInterpretation,
    expected_failed_gate: SemanticSelectivityGate | None,
) -> None:
    summary = summarize_semantic_selectivity(rows, _selected_half())
    assert summary.interpretation is expected_interpretation
    if expected_failed_gate is not None:
        assert expected_failed_gate in (
            summary.baseline_selectivity_challenge_gate,
            summary.unsupported_safety_gate,
            summary.clean_alert_reduction_gate,
            summary.coverage_gate,
        )


def test_result_recomputes_selection_decisions_summary_and_deterministic_json() -> None:
    selection = _selected_half()
    rows = _completed_rows(baseline_clean_alerts=6, selective_clean_alerts=3)
    summary = summarize_semantic_selectivity(rows, selection)
    result = SemanticSelectivityExecutionResult(
        schema_version="0.1",
        probe_id="semantic-selectivity-probe-v0.1",
        hypothesis_id="H3",
        fixture_sha256=SEMANTIC_SELECTIVITY_FIXTURE_SHA256,
        judge_id=PINNED_JUDGE_ID,
        judge_version=SEMANTIC_SELECTIVITY_MODEL_REVISION,
        calibration_selection=selection,
        judgments=rows,
        summary=summary,
    )
    text = result.to_json()
    assert text == result.to_json()
    assert SemanticSelectivityExecutionResult.model_validate_json(text) == result
    assert "premise" not in text
    assert "clean_hypothesis" not in text
    assert "unsupported_hypothesis" not in text

    payload = result.model_dump(mode="json")
    payload["summary"]["baseline_clean_alerts"] = 7
    with pytest.raises(ValidationError, match="summary does not recompute"):
        SemanticSelectivityExecutionResult.model_validate(payload)


def test_stored_decision_and_selected_threshold_tampering_fail_closed() -> None:
    row = _completed_rows(baseline_clean_alerts=6, selective_clean_alerts=3)[48]
    payload = row.model_dump(mode="json")
    payload["selective_decision"] = "NO_ALERT"
    with pytest.raises(ValidationError, match="selective decision"):
        SemanticSelectivityJudgment.model_validate(payload)

    selection = _selected_half().model_dump(mode="json")
    selection["selected_threshold"] = 0.6
    with pytest.raises(ValidationError, match="tie-break"):
        CalibrationSelection.model_validate(selection)


def test_fixture_hash_preflight_rejects_tampering(tmp_path: Path) -> None:
    tampered = tmp_path / "semantic-selectivity.json"
    tampered.write_bytes(FIXTURE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fixture SHA mismatch"):
        preflight_semantic_selectivity(tampered)
