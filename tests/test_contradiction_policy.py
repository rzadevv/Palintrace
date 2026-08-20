from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import pytest

from memlint.semantics import SemanticJudgment, SemanticRelation, SemanticUsage
from tools.evaluate_contradiction_policy import (
    ContradictionAggregationPolicy,
    DirectionalResult,
    PairClass,
    PairResult,
    PolicyReport,
    aggregate_pair_relations,
    asymmetric_results,
    evaluate_timed_pass,
    is_order_invariant,
    load_cases,
    policy_report,
    select_primary_policy,
)

CONTRADICTION_PROBE_PATH = Path("tests/fixtures/contradiction_pair_probe_v0.1.json")


def _direction(relation: SemanticRelation, *, score: float = 0.5) -> DirectionalResult:
    return DirectionalResult(relation=relation, score=score, input_tokens=5)


def _pair(
    case_id: str,
    expected: PairClass,
    a_to_b: SemanticRelation,
    b_to_a: SemanticRelation,
) -> PairResult:
    return PairResult(
        case_id=case_id,
        expected_pair_class=expected,
        a_to_b=_direction(a_to_b),
        b_to_a=_direction(b_to_a),
    )


@pytest.mark.parametrize(
    ("a_to_b", "b_to_a", "any_expected", "both_expected"),
    [
        (
            SemanticRelation.CONTRADICTION,
            SemanticRelation.CONTRADICTION,
            PairClass.CONTRADICTION,
            PairClass.CONTRADICTION,
        ),
        (
            SemanticRelation.CONTRADICTION,
            SemanticRelation.NEUTRAL,
            PairClass.CONTRADICTION,
            PairClass.COMPATIBLE,
        ),
        (
            SemanticRelation.NEUTRAL,
            SemanticRelation.CONTRADICTION,
            PairClass.CONTRADICTION,
            PairClass.COMPATIBLE,
        ),
        (
            SemanticRelation.NEUTRAL,
            SemanticRelation.NEUTRAL,
            PairClass.COMPATIBLE,
            PairClass.COMPATIBLE,
        ),
        (
            SemanticRelation.ENTAILMENT,
            SemanticRelation.CONTRADICTION,
            PairClass.CONTRADICTION,
            PairClass.COMPATIBLE,
        ),
    ],
)
def test_exact_two_policy_relation_table(
    a_to_b: SemanticRelation,
    b_to_a: SemanticRelation,
    any_expected: PairClass,
    both_expected: PairClass,
) -> None:
    assert (
        aggregate_pair_relations(
            ContradictionAggregationPolicy.ANY_DIRECTION,
            a_to_b,
            b_to_a,
        )
        is any_expected
    )
    assert (
        aggregate_pair_relations(
            ContradictionAggregationPolicy.BOTH_DIRECTIONS,
            a_to_b,
            b_to_a,
        )
        is both_expected
    )


def test_only_the_two_frozen_symmetric_policies_exist() -> None:
    assert tuple(ContradictionAggregationPolicy) == (
        ContradictionAggregationPolicy.ANY_DIRECTION,
        ContradictionAggregationPolicy.BOTH_DIRECTIONS,
    )


@pytest.mark.parametrize(
    ("a_to_b", "b_to_a"),
    list(itertools.product(SemanticRelation, repeat=2)),
)
def test_swapping_directional_results_preserves_both_policy_decisions(
    a_to_b: SemanticRelation,
    b_to_a: SemanticRelation,
) -> None:
    for policy in ContradictionAggregationPolicy:
        assert aggregate_pair_relations(policy, a_to_b, b_to_a) is aggregate_pair_relations(
            policy,
            b_to_a,
            a_to_b,
        )


def test_contradiction_pair_probe_v0_1_has_frozen_content_and_shape() -> None:
    assert hashlib.sha256(CONTRADICTION_PROBE_PATH.read_bytes()).hexdigest() == (
        "0744755a747164a9ff646a094b78fdf132e2b89de09556cf17f0189054d72744"
    )
    cases = json.loads(CONTRADICTION_PROBE_PATH.read_text(encoding="utf-8"))
    assert len(cases) == 18
    assert [case["id"] for case in cases] == [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
    ]
    assert Counter(case["expected_pair_class"] for case in cases) == {
        "contradiction": 6,
        "compatible": 12,
    }
    assert all(
        set(case) == {"id", "memory_a", "memory_b", "expected_pair_class"}
        for case in cases
    )


def test_existing_probe_hashes_remain_frozen() -> None:
    expected_hashes = {
        Path("tests/fixtures/semantic_probe_v0.1.json"): (
            "e277c04b9b18d5717f94b524e65467b0240ec515961abed49398132dc8777fb4"
        ),
        Path("tests/fixtures/evidence_composition_probe_v0.1.json"): (
            "84f824548b1ae2ee2d75fc04e5069bb1d8e45580092515a6c1aaa5d656675237"
        ),
    }
    for path, expected_hash in expected_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash


def test_loader_returns_the_exact_fixed_case_groups() -> None:
    cases = load_cases()

    assert [case.case_id for case in cases] == [
        *(f"C{index}" for index in range(1, 7)),
        *(f"N{index}" for index in range(1, 7)),
        *(f"T{index}" for index in range(1, 7)),
    ]
    assert all(case.expected_pair_class is PairClass.CONTRADICTION for case in cases[:6])
    assert all(case.expected_pair_class is PairClass.COMPATIBLE for case in cases[6:])


def test_policy_reports_keep_normal_and_temporal_false_positives_separate() -> None:
    results = (
        _pair(
            "C1",
            PairClass.CONTRADICTION,
            SemanticRelation.CONTRADICTION,
            SemanticRelation.CONTRADICTION,
        ),
        _pair(
            "C2",
            PairClass.CONTRADICTION,
            SemanticRelation.CONTRADICTION,
            SemanticRelation.NEUTRAL,
        ),
        _pair(
            "N1",
            PairClass.COMPATIBLE,
            SemanticRelation.NEUTRAL,
            SemanticRelation.CONTRADICTION,
        ),
        _pair(
            "N2",
            PairClass.COMPATIBLE,
            SemanticRelation.NEUTRAL,
            SemanticRelation.NEUTRAL,
        ),
        _pair(
            "T1",
            PairClass.COMPATIBLE,
            SemanticRelation.CONTRADICTION,
            SemanticRelation.NEUTRAL,
        ),
        _pair(
            "T2",
            PairClass.COMPATIBLE,
            SemanticRelation.ENTAILMENT,
            SemanticRelation.ENTAILMENT,
        ),
    )

    any_report = policy_report(results, ContradictionAggregationPolicy.ANY_DIRECTION)
    both_report = policy_report(results, ContradictionAggregationPolicy.BOTH_DIRECTIONS)

    assert any_report.correct_ids == ("C1", "C2", "N2", "T2")
    assert any_report.contradiction_ids == ("C1", "C2")
    assert any_report.missed_contradiction_ids == ()
    assert any_report.normal_false_positive_ids == ("N1",)
    assert any_report.temporal_false_positive_ids == ("T1",)
    assert any_report.total_false_positives == 2
    assert any_report.asymmetric_contradiction_ids == ("C2", "N1", "T1")
    assert both_report.correct_ids == ("C1", "N1", "N2", "T1", "T2")
    assert both_report.contradiction_ids == ("C1",)
    assert both_report.missed_contradiction_ids == ("C2",)
    assert both_report.normal_false_positive_ids == ()
    assert both_report.temporal_false_positive_ids == ()
    assert both_report.total_false_positives == 0
    assert both_report.asymmetric_contradiction_ids == ()
    assert is_order_invariant(results, ContradictionAggregationPolicy.ANY_DIRECTION)
    assert is_order_invariant(results, ContradictionAggregationPolicy.BOTH_DIRECTIONS)
    assert [result.case_id for result in asymmetric_results(results)] == ["C2", "N1", "T1"]
    assert select_primary_policy((any_report, both_report)) == (
        ContradictionAggregationPolicy.BOTH_DIRECTIONS,
        "fewer false positives across the 12 compatible cases",
    )


def _report(
    policy: ContradictionAggregationPolicy,
    *,
    detected: tuple[str, ...] = (),
    normal_false_positives: tuple[str, ...] = (),
    temporal_false_positives: tuple[str, ...] = (),
    asymmetric_contradictions: tuple[str, ...] = (),
) -> PolicyReport:
    return PolicyReport(
        policy=policy,
        correct_ids=(),
        contradiction_ids=detected,
        missed_contradiction_ids=(),
        normal_false_positive_ids=normal_false_positives,
        temporal_false_positive_ids=temporal_false_positives,
        asymmetric_contradiction_ids=asymmetric_contradictions,
    )


def test_selection_prefers_detection_only_after_false_positive_criteria_tie() -> None:
    any_report = _report(
        ContradictionAggregationPolicy.ANY_DIRECTION,
        detected=("C1", "C2"),
    )
    both_report = _report(
        ContradictionAggregationPolicy.BOTH_DIRECTIONS,
        detected=("C1",),
    )

    assert select_primary_policy((any_report, both_report)) == (
        ContradictionAggregationPolicy.ANY_DIRECTION,
        "more correctly detected clear contradiction cases",
    )


def test_selection_prefers_fewer_temporal_false_positives_after_total_ties() -> None:
    any_report = _report(
        ContradictionAggregationPolicy.ANY_DIRECTION,
        normal_false_positives=("N1", "N2"),
    )
    both_report = _report(
        ContradictionAggregationPolicy.BOTH_DIRECTIONS,
        normal_false_positives=("N1",),
        temporal_false_positives=("T1",),
    )

    assert select_primary_policy((any_report, both_report)) == (
        ContradictionAggregationPolicy.ANY_DIRECTION,
        "fewer false positives across the six temporal-compatible cases",
    )


def test_selection_prefers_fewer_asymmetric_decisions_after_higher_priorities_tie() -> None:
    any_report = _report(
        ContradictionAggregationPolicy.ANY_DIRECTION,
        asymmetric_contradictions=("C1",),
    )
    both_report = _report(ContradictionAggregationPolicy.BOTH_DIRECTIONS)

    assert select_primary_policy((any_report, both_report)) == (
        ContradictionAggregationPolicy.BOTH_DIRECTIONS,
        "fewer pair decisions resting on a directional asymmetry",
    )


def test_selection_uses_conservative_both_directions_tie_break() -> None:
    any_report = _report(ContradictionAggregationPolicy.ANY_DIRECTION)
    both_report = _report(ContradictionAggregationPolicy.BOTH_DIRECTIONS)

    assert select_primary_policy((any_report, both_report)) == (
        ContradictionAggregationPolicy.BOTH_DIRECTIONS,
        "conservative BOTH_DIRECTIONS tie-break",
    )


class _RecordingJudge:
    judge_id = "test:recording"
    judge_version = "1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        self.calls.append((premise, hypothesis))
        return SemanticJudgment(
            relation=SemanticRelation.NEUTRAL,
            score=0.6,
            usage=SemanticUsage(model_calls=1, input_tokens=4, output_tokens=0),
        )


def test_complete_pass_judges_every_pair_in_both_directions() -> None:
    cases = load_cases()
    judge = _RecordingJudge()

    timed_pass = evaluate_timed_pass(judge, cases)

    assert len(timed_pass.results) == 18
    assert len(judge.calls) == 36
    assert timed_pass.model_calls == 36
    assert timed_pass.input_tokens == 144
    assert len(timed_pass.latencies_ms) == 36
    for index, case in enumerate(cases):
        assert judge.calls[index * 2] == (case.memory_a, case.memory_b)
        assert judge.calls[index * 2 + 1] == (case.memory_b, case.memory_a)
