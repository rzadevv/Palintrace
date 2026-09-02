from __future__ import annotations

import ast
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

import palintrace.evaluation.retrieval_strong_probe as probe
from palintrace.retrieval import (
    PairedRetrievalChallengeAssessment,
    RetrievalChallengeOutcome,
    RetrievalHit,
    RetrievalObservation,
    RetrievalUsage,
    assess_paired_retrieval_challenge,
)

FIXTURE_PATH = Path("tests/fixtures/retrieval_shadowing_strong_probe_v0.1.json")
BENCHMARK_ROOT = Path("tests/fixtures/benchmark_v0.1")
SIMPLE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@pytest.fixture(scope="module")
def spec() -> probe.RetrievalStrongProbeSpec:
    return probe.load_retrieval_strong_probe(FIXTURE_PATH)


def _fake_observation(
    case: probe.RetrievalStrongProbeCase,
    *,
    baseline_sufficient: bool,
    mutated_sufficient: bool,
) -> probe.RetrievalStrongProbeObservation:
    query_sha256 = hashlib.sha256(case.query.encode("utf-8")).hexdigest()
    target_id = case.target_memory.id
    non_target_id = case.baseline_other_memories[0].id

    def observation(
        *,
        condition: str,
        sufficient: bool,
        candidate_count: int,
    ) -> RetrievalObservation:
        hit_id = target_id if sufficient else non_target_id
        return RetrievalObservation(
            request_id=f"{case.case_id}:{condition}",
            query_sha256=query_sha256,
            expected_memory_ids=case.expected_memory_ids,
            top_k=case.top_k,
            retriever_id=probe.FROZEN_RETRIEVER_KIND,
            retriever_version=probe.FROZEN_RETRIEVER_VERSION,
            hits=(RetrievalHit(memory_id=hit_id, rank=1, score=None),),
            usage=RetrievalUsage(
                retrieval_calls=1,
                candidate_count=candidate_count,
            ),
        )

    baseline = observation(
        condition="baseline",
        sufficient=baseline_sufficient,
        candidate_count=4,
    )
    mutated = observation(
        condition="mutated",
        sufficient=mutated_sufficient,
        candidate_count=12,
    )
    paired = assess_paired_retrieval_challenge(
        baseline,
        mutated,
        policy=case.policy,
        case_id=case.case_id,
    )
    return probe.RetrievalStrongProbeObservation(
        case_id=case.case_id,
        case_kind=case.case_kind,
        challenge_family=case.challenge_family,
        domain=case.domain,
        baseline_observation=baseline,
        mutated_observation=mutated,
        paired_assessment=paired,
    )


def _fake_matrix(
    spec: probe.RetrievalStrongProbeSpec,
    *,
    induced: set[str] | None = None,
    baseline_insufficient: set[str] | None = None,
) -> tuple[probe.RetrievalStrongProbeObservation, ...]:
    induced = induced or set()
    baseline_insufficient = baseline_insufficient or set()
    assert induced.isdisjoint(baseline_insufficient)
    return tuple(
        _fake_observation(
            case,
            baseline_sufficient=case.case_id not in baseline_insufficient,
            mutated_sufficient=case.case_id not in induced,
        )
        for case in spec.cases
    )


def _forge_paired_assessment(
    observation: probe.RetrievalStrongProbeObservation,
    *,
    outcome: RetrievalChallengeOutcome,
    mutated_sufficient: bool,
) -> PairedRetrievalChallengeAssessment:
    expected_ids = observation.mutated_observation.expected_memory_ids
    payload = observation.paired_assessment.model_dump(mode="python")
    payload.update(
        outcome=outcome,
        mutated_sufficient=mutated_sufficient,
        mutated_retrieved_expected_memory_ids=(expected_ids if mutated_sufficient else ()),
        mutated_missing_expected_memory_ids=(() if mutated_sufficient else expected_ids),
    )
    return PairedRetrievalChallengeAssessment.model_validate(payload)


def _simple_tokens(text: str) -> frozenset[str]:
    return frozenset(match.group(0).lower() for match in SIMPLE_TOKEN_RE.finditer(text))


def _jaccard(left: str, right: str) -> float:
    left_tokens = _simple_tokens(left)
    right_tokens = _simple_tokens(right)
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def test_exact_probe_identity_counts_and_order(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    assert spec.schema_version == "0.1"
    assert spec.probe_id == "retrieval-shadowing-strong-probe-v0.1"
    assert spec.split == "development"
    assert tuple(case.case_id for case in spec.cases) == probe.EXPECTED_CASE_IDS
    assert len(spec.cases) == 30
    assert Counter(case.case_kind.value for case in spec.cases) == {
        "strong_challenge": 24,
        "resilience_control": 6,
    }
    assert Counter(case.challenge_family.value for case in spec.cases) == {
        "query_term_crowding": 8,
        "negated_value_decoys": 8,
        "contextual_mention_decoys": 8,
        "low_overlap_control": 6,
    }


def test_domains_are_exactly_balanced(spec: probe.RetrievalStrongProbeSpec) -> None:
    assert {case.domain for case in spec.cases} == set(probe.RetrievalStrongProbeDomain)
    for domain in probe.RetrievalStrongProbeDomain:
        assert sum(
            case.domain is domain
            and case.case_kind is probe.RetrievalStrongProbeCaseKind.STRONG_CHALLENGE
            for case in spec.cases
        ) == 4
        assert sum(
            case.domain is domain
            and case.case_kind is probe.RetrievalStrongProbeCaseKind.RESILIENCE_CONTROL
            for case in spec.cases
        ) == 1


def test_every_case_freezes_the_same_four_plus_eight_structure(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    for case in spec.cases:
        assert len(case.baseline_other_memories) == 3
        assert len(case.distractor_memories) == 8
        assert len(case.expected_memory_ids) == 1
        assert case.expected_memory_ids == (case.target_memory.id,)
        assert case.top_k == 3
        assert case.policy.value == "all_expected"
        baseline = (case.target_memory, *case.baseline_other_memories)
        mutated = (*baseline, *case.distractor_memories)
        assert len(baseline) == 4
        assert len(mutated) == 12
        assert len({memory.id for memory in mutated}) == 12
        assert len({memory.content for memory in mutated}) == 12
        assert case.target_memory.content not in {
            memory.content for memory in case.distractor_memories
        }


def test_negated_value_cases_record_distinct_values_and_unambiguous_negation(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    negative_markers = ("not ", "does not ", "rejected ", "ruled out ")
    negated_cases = tuple(
        case
        for case in spec.cases
        if case.challenge_family
        is probe.RetrievalStrongProbeChallengeFamily.NEGATED_VALUE_DECOYS
    )
    assert len(negated_cases) == 8
    for case in negated_cases:
        assert case.target_value is not None
        assert len(case.distractor_values) == 8
        assert case.target_value.casefold() not in {
            value.casefold() for value in case.distractor_values
        }
        for memory, value in zip(
            case.distractor_memories,
            case.distractor_values,
            strict=True,
        ):
            content = memory.content.casefold()
            assert value.casefold() in content
            assert case.target_value.casefold() not in content
            assert any(marker in content for marker in negative_markers)


def test_fixture_reuses_no_exact_prior_retrieval_query_or_memory_text(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    benchmark = json.loads((BENCHMARK_ROOT / "benchmark.json").read_text())
    previous_queries = {
        case["mutation_request"]["query"] for case in benchmark["retrieval_cases"]
    }
    previous_contents: set[str] = set()
    for path in sorted(BENCHMARK_ROOT.glob("fixture_*_store.json")):
        payload = json.loads(path.read_text())
        previous_contents.update(memory["content"] for memory in payload["memories"])
    example = json.loads(Path("examples/mutation-store.json").read_text())
    previous_contents.update(memory["content"] for memory in example["memories"])

    for source_path in (
        Path("src/palintrace/mutations/shadowing.py"),
        Path("tests/test_retrieval_challenge.py"),
    ):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        previous_contents.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )

    new_queries = {case.query for case in spec.cases}
    new_contents = {
        memory.content
        for case in spec.cases
        for memory in (
            case.target_memory,
            *case.baseline_other_memories,
            *case.distractor_memories,
        )
    }
    assert new_queries.isdisjoint(previous_queries | previous_contents)
    assert new_contents.isdisjoint(previous_queries | previous_contents)


def test_simple_model_free_overlap_separates_strong_cases_from_controls(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    # Descriptive only: this local regex tokenizer does not import or call BM25.
    strong_overlap = [
        _jaccard(case.query, memory.content)
        for case in spec.cases
        if case.case_kind is probe.RetrievalStrongProbeCaseKind.STRONG_CHALLENGE
        for memory in case.distractor_memories
    ]
    control_overlap = [
        _jaccard(case.query, memory.content)
        for case in spec.cases
        if case.case_kind is probe.RetrievalStrongProbeCaseKind.RESILIENCE_CONTROL
        for memory in case.distractor_memories
    ]
    assert statistics.mean(strong_overlap) > statistics.mean(control_overlap)


def test_spec_and_models_serialize_deterministically(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    first = spec.to_json()
    second = probe.RetrievalStrongProbeSpec.model_validate_json(first).to_json()
    assert first == second


def test_fixture_hash_is_frozen_and_preflights_before_parsing(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    assert probe.sha256_file(FIXTURE_PATH) == (
        "98c2a6e1f1f5a38f34691918dd99a1eeb9096888ca2c02d63469595820c87748"
    )
    assert probe.preflight_retrieval_strong_probe(FIXTURE_PATH) == spec


def test_existing_paired_assessment_produces_only_three_frozen_outcomes(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    case = spec.cases[0]
    outcomes = {
        (True, False): "induced_shadowing",
        (True, True): "resilient",
        (False, False): "baseline_insufficient",
        (False, True): "baseline_insufficient",
    }
    for (baseline, mutated), expected in outcomes.items():
        observation = _fake_observation(
            case,
            baseline_sufficient=baseline,
            mutated_sufficient=mutated,
        )
        assert observation.paired_assessment.outcome.value == expected


def test_observation_accepts_exact_recomputed_paired_assessment(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    observation = _fake_observation(
        spec.cases[0],
        baseline_sufficient=True,
        mutated_sufficient=True,
    )

    assert (
        probe.RetrievalStrongProbeObservation.model_validate(
            observation.model_dump(mode="python")
        )
        == observation
    )


def test_observation_rejects_forged_induced_assessment(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    observation = _fake_observation(
        spec.cases[0],
        baseline_sufficient=True,
        mutated_sufficient=True,
    )
    forged = _forge_paired_assessment(
        observation,
        outcome=RetrievalChallengeOutcome.INDUCED_SHADOWING,
        mutated_sufficient=False,
    )

    with pytest.raises(ValidationError, match="recomputed from raw observations"):
        probe.RetrievalStrongProbeObservation(
            case_id=observation.case_id,
            case_kind=observation.case_kind,
            challenge_family=observation.challenge_family,
            domain=observation.domain,
            baseline_observation=observation.baseline_observation,
            mutated_observation=observation.mutated_observation,
            paired_assessment=forged,
        )


def test_observation_rejects_forged_resilient_assessment(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    observation = _fake_observation(
        spec.cases[0],
        baseline_sufficient=True,
        mutated_sufficient=False,
    )
    forged = _forge_paired_assessment(
        observation,
        outcome=RetrievalChallengeOutcome.RESILIENT,
        mutated_sufficient=True,
    )

    with pytest.raises(ValidationError, match="recomputed from raw observations"):
        probe.RetrievalStrongProbeObservation(
            case_id=observation.case_id,
            case_kind=observation.case_kind,
            challenge_family=observation.challenge_family,
            domain=observation.domain,
            baseline_observation=observation.baseline_observation,
            mutated_observation=observation.mutated_observation,
            paired_assessment=forged,
        )


def test_all_four_gates_pass_at_preregistered_boundaries(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    induced = {f"RS-Q{index:02d}" for index in range(1, 5)} | {
        f"RS-N{index:02d}" for index in range(1, 5)
    }
    baseline_insufficient = {f"RS-C{index:02d}" for index in range(5, 9)} | {
        "RS-R06"
    }
    induced.add("RS-R01")
    summary = probe.summarize_retrieval_strong_probe(
        _fake_matrix(
            spec,
            induced=induced,
            baseline_insufficient=baseline_insufficient,
        )
    )
    assert summary.strong_baseline_eligible_cases == 20
    assert summary.strong_induced_shadowing_cases == 8
    assert summary.strong_induced_shadowing_rate == 0.4
    assert summary.families_with_induced_shadowing == 2
    assert summary.control_baseline_eligible_cases == 5
    assert summary.control_induced_shadowing_cases == 1
    assert summary.baseline_eligibility_gate.value == "BASELINE_ELIGIBILITY_GATE_PASS"
    assert summary.strong_shadowing_gate.value == "STRONG_SHADOWING_GATE_PASS"
    assert summary.family_breadth_gate.value == "FAMILY_BREADTH_GATE_PASS"
    assert summary.control_stability_gate.value == "CONTROL_STABILITY_GATE_PASS"
    assert summary.interpretation is probe.RetrievalStrongProbeInterpretation.SUPPORTS_H4


def test_supports_h4_with_more_than_boundary_counts(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    induced = {f"RS-Q{index:02d}" for index in range(1, 6)} | {
        f"RS-N{index:02d}" for index in range(1, 6)
    }
    summary = probe.summarize_retrieval_strong_probe(
        _fake_matrix(spec, induced=induced)
    )
    assert summary.strong_induced_shadowing_cases == 10
    assert summary.interpretation is probe.RetrievalStrongProbeInterpretation.SUPPORTS_H4


def test_inadequate_baseline_is_inconclusive_even_when_other_gates_pass(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    induced = {f"RS-Q{index:02d}" for index in range(1, 5)} | {
        f"RS-N{index:02d}" for index in range(1, 5)
    }
    baseline_insufficient = {
        "RS-C04",
        "RS-C05",
        "RS-C06",
        "RS-C07",
        "RS-C08",
    }
    summary = probe.summarize_retrieval_strong_probe(
        _fake_matrix(
            spec,
            induced=induced,
            baseline_insufficient=baseline_insufficient,
        )
    )
    assert summary.strong_baseline_eligible_cases == 19
    assert summary.strong_shadowing_gate.value == "STRONG_SHADOWING_GATE_PASS"
    assert summary.family_breadth_gate.value == "FAMILY_BREADTH_GATE_PASS"
    assert summary.baseline_eligibility_gate.value == "BASELINE_ELIGIBILITY_GATE_FAIL"
    assert (
        summary.interpretation
        is probe.RetrievalStrongProbeInterpretation.INCONCLUSIVE_BASELINE_CONSTRUCTION
    )


@pytest.mark.parametrize(
    ("induced", "baseline_insufficient", "expected_failed_gate"),
    [
        (
            {f"RS-Q{index:02d}" for index in range(1, 6)}
            | {f"RS-N{index:02d}" for index in range(1, 5)},
            set(),
            "STRONG_SHADOWING_GATE_FAIL",
        ),
        (
            {f"RS-Q{index:02d}" for index in range(1, 9)},
            {"RS-N07", "RS-N08", "RS-C07", "RS-C08"},
            "FAMILY_BREADTH_GATE_FAIL",
        ),
        (
            {
                "RS-Q01",
                "RS-Q02",
                "RS-Q03",
                "RS-Q04",
                "RS-Q05",
                "RS-N01",
                "RS-N02",
                "RS-N03",
                "RS-N04",
                "RS-N05",
                "RS-R01",
                "RS-R02",
            },
            set(),
            "CONTROL_STABILITY_GATE_FAIL",
        ),
        (
            {
                "RS-Q01",
                "RS-Q02",
                "RS-Q03",
                "RS-Q04",
                "RS-Q05",
                "RS-N01",
                "RS-N02",
                "RS-N03",
                "RS-N04",
                "RS-N05",
            },
            {"RS-R05", "RS-R06"},
            "CONTROL_STABILITY_GATE_FAIL",
        ),
    ],
)
def test_nonbaseline_gate_failures_preserve_does_not_support(
    spec: probe.RetrievalStrongProbeSpec,
    induced: set[str],
    baseline_insufficient: set[str],
    expected_failed_gate: str,
) -> None:
    summary = probe.summarize_retrieval_strong_probe(
        _fake_matrix(
            spec,
            induced=induced,
            baseline_insufficient=baseline_insufficient,
        )
    )
    assert summary.baseline_eligibility_gate.value == "BASELINE_ELIGIBILITY_GATE_PASS"
    assert expected_failed_gate in {
        summary.strong_shadowing_gate.value,
        summary.family_breadth_gate.value,
        summary.control_stability_gate.value,
    }
    assert (
        summary.interpretation
        is probe.RetrievalStrongProbeInterpretation.DOES_NOT_SUPPORT_H4
    )


def test_execution_result_recomputes_summary_and_json(
    spec: probe.RetrievalStrongProbeSpec,
) -> None:
    induced = {f"RS-Q{index:02d}" for index in range(1, 6)} | {
        f"RS-N{index:02d}" for index in range(1, 6)
    }
    observations = _fake_matrix(spec, induced=induced)
    result = probe.RetrievalStrongProbeExecutionResult(
        schema_version=probe.RETRIEVAL_STRONG_PROBE_SCHEMA_VERSION,
        probe_id=probe.RETRIEVAL_STRONG_PROBE_ID,
        fixture_sha256=probe.RETRIEVAL_STRONG_PROBE_FIXTURE_SHA256,
        observations=observations,
        summary=probe.summarize_retrieval_strong_probe(observations),
    )
    first = result.to_json()
    assert probe.RetrievalStrongProbeExecutionResult.model_validate_json(first).to_json() == first
    assert "Which" not in first
    assert "Alacritty" not in first


def test_exactly_three_final_interpretations_are_frozen() -> None:
    assert [item.value for item in probe.RetrievalStrongProbeInterpretation] == [
        "SUPPORTS_H4",
        "DOES_NOT_SUPPORT_H4",
        "INCONCLUSIVE_BASELINE_CONSTRUCTION",
    ]
