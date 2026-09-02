from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from memlint.cli import build_parser
from tools.evaluate_contradiction_models import (
    CANDIDATES,
    MAXIMUM_TEMPORAL_FALSE_POSITIVES,
    MAXIMUM_TOTAL_COMPATIBLE_FALSE_POSITIVES,
    MINIMUM_CONTRADICTIONS_DETECTED,
    ModelCandidate,
    ModelPolicyCombination,
    passes_readiness_gate,
    select_ready_combination,
    validate_candidate_definitions,
)
from tools.evaluate_contradiction_policy import (
    ContradictionAggregationPolicy,
    PolicyReport,
)
from tools.evaluate_evidence_composition import (
    MODEL_ID as UNSUPPORTED_DEVELOPMENT_MODEL_ID,
)
from tools.evaluate_evidence_composition import (
    MODEL_REVISION as UNSUPPORTED_DEVELOPMENT_MODEL_REVISION,
)


def _report(
    *,
    detected: int = 5,
    normal_false_positives: int = 0,
    temporal_false_positives: int = 0,
    policy: ContradictionAggregationPolicy = (
        ContradictionAggregationPolicy.BOTH_DIRECTIONS
    ),
) -> PolicyReport:
    contradiction_ids = tuple(f"C{index}" for index in range(1, detected + 1))
    return PolicyReport(
        policy=policy,
        correct_ids=(),
        contradiction_ids=contradiction_ids,
        missed_contradiction_ids=tuple(
            f"C{index}" for index in range(detected + 1, 7)
        ),
        normal_false_positive_ids=tuple(
            f"N{index}" for index in range(1, normal_false_positives + 1)
        ),
        temporal_false_positive_ids=tuple(
            f"T{index}" for index in range(1, temporal_false_positives + 1)
        ),
        asymmetric_contradiction_ids=(),
    )


def _combination(
    candidate: ModelCandidate,
    report: PolicyReport,
    *,
    latency_ms: float = 1.0,
) -> ModelPolicyCombination:
    return ModelPolicyCombination(
        candidate=candidate,
        report=report,
        median_directional_cpu_latency_ms=latency_ms,
    )


def test_exact_four_candidate_definitions_are_frozen() -> None:
    assert tuple(
        (
            candidate.display_name,
            candidate.model_id,
            candidate.revision,
            candidate.license_id,
            candidate.safetensors_bytes,
        )
        for candidate in CANDIDATES
    ) == (
        (
            "MiniLM",
            "cross-encoder/nli-MiniLM2-L6-H768",
            "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
            "apache-2.0",
            328_499_560,
        ),
        (
            "DeBERTa v3 small",
            "cross-encoder/nli-deberta-v3-small",
            "fa2804872c3b4bd748f38c0185cc85775361e735",
            "apache-2.0",
            567_605_820,
        ),
        (
            "DeBERTa v3 base",
            "cross-encoder/nli-deberta-v3-base",
            "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
            "apache-2.0",
            737_726_552,
        ),
        (
            "Tasksource DeBERTa small",
            "tasksource/deberta-small-long-nli",
            "9a77395d4d3751be9e2a69c4ae318491d9b3fffb",
            "apache-2.0",
            567_601_628,
        ),
    )
    validate_candidate_definitions(CANDIDATES)


def test_mutable_main_revision_is_forbidden_for_recorded_configuration() -> None:
    candidates = (*CANDIDATES[:3], ModelCandidate("bad", "example/nli", "main", None, 1))

    with pytest.raises(ValueError, match="mutable 'main'"):
        validate_candidate_definitions(candidates)


def test_all_three_probe_hashes_remain_frozen() -> None:
    expected = {
        Path("tests/fixtures/contradiction_pair_probe_v0.1.json"): (
            "0744755a747164a9ff646a094b78fdf132e2b89de09556cf17f0189054d72744"
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


def test_readiness_gate_is_pre_frozen() -> None:
    assert MINIMUM_CONTRADICTIONS_DETECTED == 5
    assert MAXIMUM_TEMPORAL_FALSE_POSITIVES == 0
    assert MAXIMUM_TOTAL_COMPATIBLE_FALSE_POSITIVES == 1


def test_five_of_six_contradictions_passes_recall_gate() -> None:
    assert passes_readiness_gate(_report(detected=5))


def test_four_of_six_contradictions_fails_recall_gate() -> None:
    assert not passes_readiness_gate(_report(detected=4))


def test_zero_temporal_false_positives_are_required() -> None:
    assert passes_readiness_gate(_report(temporal_false_positives=0))
    assert not passes_readiness_gate(_report(temporal_false_positives=1))


def test_one_total_compatible_false_positive_is_allowed() -> None:
    assert passes_readiness_gate(_report(normal_false_positives=1))


def test_two_total_compatible_false_positives_fails() -> None:
    assert not passes_readiness_gate(_report(normal_false_positives=2))


def test_selection_considers_only_gate_passing_combinations() -> None:
    failing_but_superficially_better = _combination(
        CANDIDATES[0],
        _report(detected=4),
        latency_ms=0.1,
    )
    passing = _combination(
        CANDIDATES[2],
        _report(detected=5, normal_false_positives=1),
        latency_ms=100.0,
    )

    assert select_ready_combination((failing_but_superficially_better, passing)) is passing


def test_no_pass_condition_returns_no_model_or_policy() -> None:
    combinations = tuple(
        _combination(candidate, _report(temporal_false_positives=1))
        for candidate in CANDIDATES
    )

    assert select_ready_combination(combinations) is None


def test_unsupported_checker_configuration_remains_unchanged() -> None:
    assert UNSUPPORTED_DEVELOPMENT_MODEL_ID == "cross-encoder/nli-MiniLM2-L6-H768"
    assert UNSUPPORTED_DEVELOPMENT_MODEL_REVISION == (
        "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
    )
    args = build_parser().parse_args(
        [
            "audit",
            "--store",
            "store.json",
            "--checker",
            "unsupported_claim",
        ]
    )
    assert args.semantic_model_id is None
    assert args.semantic_model_revision is None
