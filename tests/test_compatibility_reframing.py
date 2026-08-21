from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import pytest

from memlint.semantics import SemanticJudgment, SemanticRelation, SemanticUsage
from tools.evaluate_compatibility_reframing import (
    CANDIDATES,
    COMPATIBILITY_HYPOTHESIS,
    CONTRADICTION_PROBE_SHA256,
    MAXIMUM_COMPATIBLE_FALSE_INCOMPATIBLE,
    MAXIMUM_TEMPORAL_FALSE_INCOMPATIBLE,
    MINIMUM_COMPATIBLE_CORRECT,
    MINIMUM_INCOMPATIBLE_DETECTED,
    MINIMUM_TEMPORAL_COMPATIBLE_CORRECT,
    CompatibilityClass,
    CompatibilityReport,
    ModelEvaluation,
    aggregate_compatibility,
    evaluate_timed_pass,
    nli_relation_to_compatibility,
    passes_readiness_gate,
    render_compatibility_premise,
    select_ready_model,
    validate_candidate_definitions,
)
from tools.evaluate_contradiction_policy import load_cases


def _report(
    *,
    incompatible_detected: int = 5,
    normal_correct: int = 5,
    normal_false_incompatible: int = 0,
    temporal_correct: int = 5,
    temporal_false_incompatible: int = 0,
) -> CompatibilityReport:
    normal_used = normal_correct + normal_false_incompatible
    temporal_used = temporal_correct + temporal_false_incompatible
    return CompatibilityReport(
        incompatible_detected_ids=tuple(
            f"C{index}" for index in range(1, incompatible_detected + 1)
        ),
        missed_incompatible_ids=tuple(
            f"C{index}" for index in range(incompatible_detected + 1, 7)
        ),
        normal_compatible_correct_ids=tuple(
            f"N{index}" for index in range(1, normal_correct + 1)
        ),
        normal_false_incompatible_ids=tuple(
            f"N{index}"
            for index in range(normal_correct + 1, normal_used + 1)
        ),
        normal_uncertain_ids=tuple(
            f"N{index}" for index in range(normal_used + 1, 7)
        ),
        temporal_compatible_correct_ids=tuple(
            f"T{index}" for index in range(1, temporal_correct + 1)
        ),
        temporal_false_incompatible_ids=tuple(
            f"T{index}"
            for index in range(temporal_correct + 1, temporal_used + 1)
        ),
        temporal_uncertain_ids=tuple(
            f"T{index}" for index in range(temporal_used + 1, 7)
        ),
        order_disagreement_ids=(),
    )


def _evaluation(
    candidate_index: int,
    report: CompatibilityReport,
    *,
    latency_ms: float = 1.0,
) -> ModelEvaluation:
    return ModelEvaluation(
        candidate=CANDIDATES[candidate_index],
        results=(),
        report=report,
        median_directional_cpu_latency_ms=latency_ms,
        timed_passes=3,
        timed_model_calls=108,
        timed_input_tokens=1,
        warmup_model_calls=1,
    )


def test_exact_two_candidate_models_and_pinned_revisions() -> None:
    assert tuple((candidate.model_id, candidate.revision) for candidate in CANDIDATES) == (
        (
            "cross-encoder/nli-MiniLM2-L6-H768",
            "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
        ),
        (
            "tasksource/deberta-small-long-nli",
            "9a77395d4d3751be9e2a69c4ae318491d9b3fffb",
        ),
    )
    validate_candidate_definitions(CANDIDATES)


def test_exact_fixed_hypothesis_and_premise_formatting() -> None:
    assert COMPATIBILITY_HYPOTHESIS == (
        "These two memory claims can both be true as stated."
    )
    assert render_compatibility_premise(
        "The user uses Python.",
        "The user uses Rust.",
    ) == (
        "Memory claim 1: The user uses Python.\n"
        "Memory claim 2: The user uses Rust."
    )


def test_ab_ba_rendering_swaps_only_the_claim_positions() -> None:
    ab = render_compatibility_premise("A", "B")
    ba = render_compatibility_premise("B", "A")

    assert ab == "Memory claim 1: A\nMemory claim 2: B"
    assert ba == "Memory claim 1: B\nMemory claim 2: A"


class _RecordingJudge:
    judge_id = "test:compatibility-recording"
    judge_version = "1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        self.calls.append((premise, hypothesis))
        return SemanticJudgment(
            relation=SemanticRelation.NEUTRAL,
            score=0.5,
            usage=SemanticUsage(model_calls=1, input_tokens=20, output_tokens=0),
        )


def test_timed_pass_calls_exact_ab_and_ba_renderings_with_fixed_hypothesis() -> None:
    case = load_cases()[6]
    judge = _RecordingJudge()

    timed = evaluate_timed_pass(judge, (case,))

    assert judge.calls == [
        (
            "Memory claim 1: The user uses Python.\n"
            "Memory claim 2: The user uses Rust.",
            COMPATIBILITY_HYPOTHESIS,
        ),
        (
            "Memory claim 1: The user uses Rust.\n"
            "Memory claim 2: The user uses Python.",
            COMPATIBILITY_HYPOTHESIS,
        ),
    ]
    assert timed.model_calls == 2
    assert timed.input_tokens == 40
    assert timed.results[0].final is CompatibilityClass.UNCERTAIN


@pytest.mark.parametrize(
    ("relation", "expected"),
    [
        (SemanticRelation.ENTAILMENT, CompatibilityClass.COMPATIBLE),
        (SemanticRelation.CONTRADICTION, CompatibilityClass.INCOMPATIBLE),
        (SemanticRelation.NEUTRAL, CompatibilityClass.UNCERTAIN),
    ],
)
def test_frozen_nli_relation_mapping(
    relation: SemanticRelation,
    expected: CompatibilityClass,
) -> None:
    assert nli_relation_to_compatibility(relation) is expected


@pytest.mark.parametrize(
    ("ab", "ba"),
    list(itertools.product(CompatibilityClass, repeat=2)),
)
def test_conservative_pair_aggregation(
    ab: CompatibilityClass,
    ba: CompatibilityClass,
) -> None:
    if ab is CompatibilityClass.INCOMPATIBLE and ba is CompatibilityClass.INCOMPATIBLE:
        expected = CompatibilityClass.INCOMPATIBLE
    elif ab is CompatibilityClass.COMPATIBLE and ba is CompatibilityClass.COMPATIBLE:
        expected = CompatibilityClass.COMPATIBLE
    else:
        expected = CompatibilityClass.UNCERTAIN

    assert aggregate_compatibility(ab, ba) is expected
    assert aggregate_compatibility(ab, ba) is aggregate_compatibility(ba, ab)


def test_readiness_gate_is_pre_frozen() -> None:
    assert MINIMUM_INCOMPATIBLE_DETECTED == 5
    assert MAXIMUM_COMPATIBLE_FALSE_INCOMPATIBLE == 0
    assert MINIMUM_COMPATIBLE_CORRECT == 10
    assert MAXIMUM_TEMPORAL_FALSE_INCOMPATIBLE == 0
    assert MINIMUM_TEMPORAL_COMPATIBLE_CORRECT == 5
    assert passes_readiness_gate(_report())


def test_zero_compatible_false_incompatible_is_required() -> None:
    assert not passes_readiness_gate(
        _report(normal_correct=5, normal_false_incompatible=1)
    )


def test_zero_temporal_false_incompatible_is_required() -> None:
    assert not passes_readiness_gate(
        _report(temporal_correct=5, temporal_false_incompatible=1)
    )


def test_ten_of_twelve_compatible_coverage_is_required() -> None:
    assert passes_readiness_gate(_report(normal_correct=5, temporal_correct=5))
    assert not passes_readiness_gate(_report(normal_correct=4, temporal_correct=5))


def test_five_of_six_temporal_compatible_coverage_is_required() -> None:
    assert passes_readiness_gate(_report(normal_correct=5, temporal_correct=5))
    assert not passes_readiness_gate(_report(normal_correct=6, temporal_correct=4))


def test_five_of_six_incompatible_coverage_is_required() -> None:
    assert passes_readiness_gate(_report(incompatible_detected=5))
    assert not passes_readiness_gate(_report(incompatible_detected=4))


def test_selection_considers_only_gate_passing_models() -> None:
    failing = _evaluation(0, _report(incompatible_detected=4), latency_ms=0.1)
    passing = _evaluation(1, _report(), latency_ms=100.0)

    assert select_ready_model((failing, passing)) is passing


def test_selection_returns_none_when_no_model_passes() -> None:
    evaluations = (
        _evaluation(0, _report(incompatible_detected=4)),
        _evaluation(1, _report(temporal_correct=4, normal_correct=6)),
    )

    assert select_ready_model(evaluations) is None


def test_frozen_fixture_hashes_are_unchanged() -> None:
    expected = {
        Path("tests/fixtures/contradiction_pair_probe_v0.1.json"): (
            CONTRADICTION_PROBE_SHA256
        ),
        Path("tests/fixtures/semantic_probe_v0.1.json"): (
            "e277c04b9b18d5717f94b524e65467b0240ec515961abed49398132dc8777fb4"
        ),
        Path("tests/fixtures/evidence_composition_probe_v0.1.json"): (
            "84f824548b1ae2ee2d75fc04e5069bb1d8e45580092515a6c1aaa5d656675237"
        ),
    }

    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected
