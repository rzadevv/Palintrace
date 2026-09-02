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

import palintrace.evaluation.retrieval_negation_confirmatory as probe
from palintrace.retrieval import (
    PairedRetrievalChallengeAssessment,
    RetrievalChallengeOutcome,
    RetrievalHit,
    RetrievalObservation,
    RetrievalUsage,
    assess_paired_retrieval_challenge,
)

FIXTURE_PATH = Path("tests/fixtures/retrieval_negation_confirmatory_v0.1.json")
OLD_PROBE_PATH = Path("tests/fixtures/retrieval_shadowing_strong_probe_v0.1.json")
BENCHMARK_ROOT = Path("tests/fixtures/benchmark_v0.1")
SIMPLE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

SUPPORT_NEGATED = {
    "H4N-01",
    "H4N-02",
    "H4N-04",
    "H4N-05",
    "H4N-07",
    "H4N-08",
    "H4N-10",
    "H4N-11",
    "H4N-13",
    "H4N-14",
    "H4N-16",
    "H4N-17",
}
SUPPORT_CONTEXTUAL = {"H4N-01", "H4N-04", "H4N-07"}
SUPPORT_LOW = {"H4N-02"}


@pytest.fixture(scope="module")
def spec() -> probe.RetrievalNegationConfirmatorySpec:
    return probe.load_retrieval_negation_confirmatory(FIXTURE_PATH)


def _simple_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in SIMPLE_TOKEN_RE.finditer(text))


def _simple_jaccard(left: str, right: str) -> float:
    left_tokens = set(_simple_tokens(left))
    right_tokens = set(_simple_tokens(right))
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _runtime_observation(
    scenario: probe.RetrievalNegationScenario,
    *,
    condition: str,
    sufficient: bool,
    candidate_count: int,
) -> RetrievalObservation:
    target_id = scenario.target_memory.id
    hit_id = target_id if sufficient else scenario.baseline_other_memories[0].id
    return RetrievalObservation(
        request_id=f"{scenario.scenario_id}:{condition}",
        query_sha256=hashlib.sha256(scenario.query.encode("utf-8")).hexdigest(),
        expected_memory_ids=scenario.expected_memory_ids,
        top_k=probe.FROZEN_TOP_K,
        retriever_id=probe.FROZEN_RETRIEVER_KIND,
        retriever_version=probe.FROZEN_RETRIEVER_VERSION,
        hits=(RetrievalHit(memory_id=hit_id, rank=1, score=None),),
        usage=RetrievalUsage(
            retrieval_calls=1,
            candidate_count=candidate_count,
        ),
    )


def _fake_scenario_observation(
    scenario: probe.RetrievalNegationScenario,
    *,
    baseline_sufficient: bool = True,
    induced_conditions: set[probe.RetrievalNegationCondition] | None = None,
) -> probe.RetrievalNegationScenarioObservation:
    induced_conditions = induced_conditions or set()
    baseline = _runtime_observation(
        scenario,
        condition="baseline",
        sufficient=baseline_sufficient,
        candidate_count=4,
    )
    conditions: list[probe.RetrievalNegationConditionObservation] = []
    for condition in probe.CONDITION_ORDER:
        mutated = _runtime_observation(
            scenario,
            condition=condition.value,
            sufficient=condition not in induced_conditions,
            candidate_count=12,
        )
        paired = assess_paired_retrieval_challenge(
            baseline,
            mutated,
            policy=probe.FROZEN_POLICY,
            case_id=f"{scenario.scenario_id}:{condition.value}",
        )
        conditions.append(
            probe.RetrievalNegationConditionObservation(
                scenario_id=scenario.scenario_id,
                condition=condition,
                mutated_observation=mutated,
                paired_assessment=paired,
            )
        )
    return probe.RetrievalNegationScenarioObservation(
        scenario_id=scenario.scenario_id,
        domain=scenario.domain,
        baseline_observation=baseline,
        conditions=tuple(conditions),
    )


def _fake_matrix(
    spec: probe.RetrievalNegationConfirmatorySpec,
    *,
    negated_induced: set[str] | None = None,
    contextual_induced: set[str] | None = None,
    low_induced: set[str] | None = None,
    baseline_insufficient: set[str] | None = None,
) -> tuple[probe.RetrievalNegationScenarioObservation, ...]:
    negated_induced = negated_induced or set()
    contextual_induced = contextual_induced or set()
    low_induced = low_induced or set()
    baseline_insufficient = baseline_insufficient or set()
    return tuple(
        _fake_scenario_observation(
            scenario,
            baseline_sufficient=scenario.scenario_id not in baseline_insufficient,
            induced_conditions={
                condition
                for condition, scenario_ids in (
                    (
                        probe.RetrievalNegationCondition.NEGATED_COMPETING_VALUE,
                        negated_induced,
                    ),
                    (
                        probe.RetrievalNegationCondition.CONTEXTUAL_COMPETING_VALUE_CONTROL,
                        contextual_induced,
                    ),
                    (
                        probe.RetrievalNegationCondition.LOW_OVERLAP_CONTROL,
                        low_induced,
                    ),
                )
                if scenario.scenario_id in scenario_ids
            },
        )
        for scenario in spec.scenarios
    )


def test_exact_probe_identity_inventory_and_balance(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    assert spec.schema_version == "0.1"
    assert spec.probe_id == "retrieval-negation-confirmatory-v0.1"
    assert spec.hypothesis_id == "H4-N"
    assert spec.split == "fresh_confirmatory_development"
    assert tuple(scenario.scenario_id for scenario in spec.scenarios) == (
        probe.EXPECTED_SCENARIO_IDS
    )
    assert len(spec.scenarios) == 18
    assert Counter(scenario.domain for scenario in spec.scenarios) == {
        domain: 3 for domain in probe.RetrievalNegationDomain
    }


def test_each_scenario_has_one_shared_four_plus_eight_design(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    raw_fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert all(
        tuple(scenario["conditions"])
        == tuple(condition.value for condition in probe.CONDITION_ORDER)
        for scenario in raw_fixture["scenarios"]
    )
    for scenario in spec.scenarios:
        assert len(scenario.baseline_other_memories) == 3
        assert len(scenario.baseline_memories) == 4
        assert len(scenario.competing_values) == 8
        assert scenario.expected_memory_ids == (scenario.target_memory.id,)
        assert tuple(type(scenario.conditions).model_fields) == tuple(
            condition.value for condition in probe.CONDITION_ORDER
        )
        for condition in probe.CONDITION_ORDER:
            assert len(scenario.conditions.for_condition(condition)) == 8
            mutated = scenario.memories_for_condition(condition)
            assert len(mutated) == 12
            assert mutated[:4] == scenario.baseline_memories
        assert spec.retriever.top_k == 3
        assert spec.retriever.policy.value == "all_expected"
        assert spec.retriever.kind == "experimental_lexical"
        assert spec.retriever.version == "0.1"


def test_all_144_matched_pairs_satisfy_frozen_lexical_invariants(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    pair_count = 0
    same_value_count = 0
    same_overlap_count = 0
    matched_length_count = 0
    rejection_tokens = {"not", "no", "never", "without", "rejected", "ruled"}
    for scenario in spec.scenarios:
        query_tokens = set(_simple_tokens(scenario.query))
        for value, negated, contextual in zip(
            scenario.competing_values,
            scenario.conditions.negated_competing_value,
            scenario.conditions.contextual_competing_value_control,
            strict=True,
        ):
            pair_count += 1
            value_present = (
                negated.content.casefold().count(value.casefold()) == 1
                and contextual.content.casefold().count(value.casefold()) == 1
            )
            same_value_count += value_present
            negated_overlap = len(query_tokens & set(_simple_tokens(negated.content)))
            contextual_overlap = len(query_tokens & set(_simple_tokens(contextual.content)))
            same_overlap_count += negated_overlap == contextual_overlap
            length_difference = abs(
                len(_simple_tokens(negated.content))
                - len(_simple_tokens(contextual.content))
            )
            matched_length_count += length_difference <= 3
            assert "not" in _simple_tokens(negated.content)
            assert not rejection_tokens & set(_simple_tokens(contextual.content))
            assert negated.content != scenario.target_memory.content
            assert contextual.content != scenario.target_memory.content
    assert pair_count == 144
    assert same_value_count == 144
    assert same_overlap_count == 144
    assert matched_length_count == 144


def test_targets_values_and_nonanswers_are_structurally_disjoint(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    for scenario in spec.scenarios:
        non_target_memories = (
            *scenario.baseline_other_memories,
            *scenario.conditions.negated_competing_value,
            *scenario.conditions.contextual_competing_value_control,
            *scenario.conditions.low_overlap_control,
        )
        assert scenario.target_value.casefold() in scenario.target_memory.content.casefold()
        assert all(
            scenario.target_value.casefold() not in memory.content.casefold()
            for memory in non_target_memories
        )
        for index, value in enumerate(scenario.competing_values):
            assert value.casefold() not in scenario.target_memory.content.casefold()
            assert all(
                value.casefold() not in memory.content.casefold()
                for memory in (
                    *scenario.baseline_other_memories,
                    *scenario.conditions.low_overlap_control,
                )
            )
            assert value.casefold() in (
                scenario.conditions.negated_competing_value[index].content.casefold()
            )
            assert value.casefold() in (
                scenario.conditions.contextual_competing_value_control[index]
                .content.casefold()
            )


def test_low_overlap_medians_are_below_both_matched_conditions(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    for scenario in spec.scenarios:
        low_median = statistics.median(
            _simple_jaccard(scenario.query, memory.content)
            for memory in scenario.conditions.low_overlap_control
        )
        negated_median = statistics.median(
            _simple_jaccard(scenario.query, memory.content)
            for memory in scenario.conditions.negated_competing_value
        )
        contextual_median = statistics.median(
            _simple_jaccard(scenario.query, memory.content)
            for memory in scenario.conditions.contextual_competing_value_control
        )
        assert low_median < negated_median
        assert low_median < contextual_median


def test_fixture_reuses_no_exact_prior_retrieval_text_or_competing_value(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    old_probe = json.loads(OLD_PROBE_PATH.read_text(encoding="utf-8"))
    old_queries = {case["query"] for case in old_probe["cases"]}
    old_texts = {
        memory["content"]
        for case in old_probe["cases"]
        for memory in (
            case["target_memory"],
            *case["baseline_other_memories"],
            *case["distractor_memories"],
        )
    }
    old_competing_values = {
        value for case in old_probe["cases"] for value in case.get("distractor_values", ())
    }
    benchmark = json.loads((BENCHMARK_ROOT / "benchmark.json").read_text())
    old_queries.update(
        case["mutation_request"]["query"] for case in benchmark["retrieval_cases"]
    )
    for path in sorted(BENCHMARK_ROOT.glob("fixture_*_store.json")):
        old_texts.update(
            memory["content"]
            for memory in json.loads(path.read_text(encoding="utf-8"))["memories"]
        )
    example = json.loads(Path("examples/mutation-store.json").read_text())
    old_texts.update(memory["content"] for memory in example["memories"])
    for source_path in (
        Path("src/palintrace/mutations/shadowing.py"),
        Path("tests/test_retrieval_challenge.py"),
    ):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        old_texts.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )

    new_queries = {scenario.query for scenario in spec.scenarios}
    new_texts = {
        memory.content
        for scenario in spec.scenarios
        for memory in (
            scenario.target_memory,
            *scenario.baseline_other_memories,
            *scenario.conditions.negated_competing_value,
            *scenario.conditions.contextual_competing_value_control,
            *scenario.conditions.low_overlap_control,
        )
    }
    new_competing_values = {
        value for scenario in spec.scenarios for value in scenario.competing_values
    }
    assert new_queries.isdisjoint(old_queries | old_texts)
    assert new_texts.isdisjoint(old_queries | old_texts)
    assert new_competing_values.isdisjoint(old_competing_values)
    old_inventories = {
        frozenset(case.get("distractor_values", ())) for case in old_probe["cases"]
    }
    assert all(
        frozenset(scenario.competing_values) not in old_inventories
        for scenario in spec.scenarios
    )


def test_fixture_hash_preflight_and_deterministic_models(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    assert probe.sha256_file(FIXTURE_PATH) == (
        probe.RETRIEVAL_NEGATION_CONFIRMATORY_FIXTURE_SHA256
    )
    assert probe.preflight_retrieval_negation_confirmatory(FIXTURE_PATH) == spec
    first = spec.to_json()
    assert probe.RetrievalNegationConfirmatorySpec.model_validate_json(first).to_json() == first


def test_valid_stored_paired_assessments_recompute(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    observation = _fake_scenario_observation(spec.scenarios[0])
    assert (
        probe.RetrievalNegationScenarioObservation.model_validate(
            observation.model_dump(mode="python")
        )
        == observation
    )


def test_forged_paired_assessment_is_rejected(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    observation = _fake_scenario_observation(spec.scenarios[0])
    first_condition = observation.conditions[0]
    expected_ids = first_condition.mutated_observation.expected_memory_ids
    payload = first_condition.paired_assessment.model_dump(mode="python")
    payload.update(
        outcome=RetrievalChallengeOutcome.INDUCED_SHADOWING,
        mutated_sufficient=False,
        mutated_retrieved_expected_memory_ids=(),
        mutated_missing_expected_memory_ids=expected_ids,
    )
    forged = PairedRetrievalChallengeAssessment.model_validate(payload)
    forged_condition = probe.RetrievalNegationConditionObservation(
        scenario_id=first_condition.scenario_id,
        condition=first_condition.condition,
        mutated_observation=first_condition.mutated_observation,
        paired_assessment=forged,
    )
    with pytest.raises(ValidationError, match="recomputed from stored observations"):
        probe.RetrievalNegationScenarioObservation(
            scenario_id=observation.scenario_id,
            domain=observation.domain,
            baseline_observation=observation.baseline_observation,
            conditions=(forged_condition, *observation.conditions[1:]),
        )


def test_all_gates_pass_for_scenario_level_matched_boundary(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    observations = _fake_matrix(
        spec,
        negated_induced=SUPPORT_NEGATED,
        contextual_induced=SUPPORT_CONTEXTUAL,
        low_induced=SUPPORT_LOW,
    )
    summary = probe.summarize_retrieval_negation_confirmatory(observations)
    assert summary.scenario_total == 18
    assert summary.baseline_eligible_scenarios == 18
    assert summary.condition_summaries[0].induced_shadowing_scenarios == 12
    assert summary.negated_induced_rate == 2 / 3
    assert summary.negation_specific_scenarios == 9
    assert summary.reverse_specific_scenarios == 0
    assert len(summary.domains_with_negated_induced) == 6
    assert summary.baseline_eligibility_gate.value == "BASELINE_ELIGIBILITY_GATE_PASS"
    assert summary.negation_replication_gate.value == "NEGATION_REPLICATION_GATE_PASS"
    assert summary.matched_specificity_gate.value == "MATCHED_SPECIFICITY_GATE_PASS"
    assert summary.contextual_control_gate.value == "CONTEXTUAL_CONTROL_GATE_PASS"
    assert summary.low_overlap_control_gate.value == "LOW_OVERLAP_CONTROL_GATE_PASS"
    assert summary.domain_breadth_gate.value == "DOMAIN_BREADTH_GATE_PASS"
    assert summary.interpretation is probe.RetrievalNegationInterpretation.SUPPORTS_H4_N


def test_seventeen_baselines_can_remain_eligible_and_support(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    summary = probe.summarize_retrieval_negation_confirmatory(
        _fake_matrix(
            spec,
            negated_induced=SUPPORT_NEGATED,
            contextual_induced=SUPPORT_CONTEXTUAL,
            low_induced=SUPPORT_LOW,
            baseline_insufficient={"H4N-18"},
        )
    )
    assert summary.baseline_eligible_scenarios == 17
    assert summary.condition_summaries[0].induced_shadowing_scenarios == 12
    assert summary.interpretation is probe.RetrievalNegationInterpretation.SUPPORTS_H4_N


def test_failed_baseline_gate_is_inconclusive(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    summary = probe.summarize_retrieval_negation_confirmatory(
        _fake_matrix(
            spec,
            negated_induced=SUPPORT_NEGATED,
            contextual_induced=SUPPORT_CONTEXTUAL,
            low_induced=SUPPORT_LOW,
            baseline_insufficient={"H4N-15", "H4N-18"},
        )
    )
    assert summary.baseline_eligible_scenarios == 16
    assert summary.baseline_eligibility_gate.value == "BASELINE_ELIGIBILITY_GATE_FAIL"
    assert (
        summary.interpretation
        is probe.RetrievalNegationInterpretation.INCONCLUSIVE_BASELINE_CONSTRUCTION
    )


@pytest.mark.parametrize(
    ("negated", "contextual", "low", "failed_gate"),
    [
        (
            SUPPORT_NEGATED - {"H4N-02"},
            SUPPORT_CONTEXTUAL,
            SUPPORT_LOW,
            "NEGATION_REPLICATION_GATE_FAIL",
        ),
        (
            SUPPORT_NEGATED,
            {"H4N-01", "H4N-04", "H4N-07", "H4N-10", "H4N-13"},
            SUPPORT_LOW,
            "MATCHED_SPECIFICITY_GATE_FAIL",
        ),
        (
            SUPPORT_NEGATED,
            {"H4N-03", "H4N-06", "H4N-09"},
            SUPPORT_LOW,
            "MATCHED_SPECIFICITY_GATE_FAIL",
        ),
        (
            SUPPORT_NEGATED,
            {"H4N-01", "H4N-02", "H4N-04", "H4N-05", "H4N-07", "H4N-08"},
            SUPPORT_LOW,
            "CONTEXTUAL_CONTROL_GATE_FAIL",
        ),
        (
            SUPPORT_NEGATED,
            SUPPORT_CONTEXTUAL,
            {"H4N-02", "H4N-05"},
            "LOW_OVERLAP_CONTROL_GATE_FAIL",
        ),
        (
            {f"H4N-{index:02d}" for index in range(1, 13)},
            {"H4N-01", "H4N-04", "H4N-07"},
            SUPPORT_LOW,
            "DOMAIN_BREADTH_GATE_FAIL",
        ),
    ],
)
def test_each_confirmatory_gate_failure_preserves_negative_interpretation(
    spec: probe.RetrievalNegationConfirmatorySpec,
    negated: set[str],
    contextual: set[str],
    low: set[str],
    failed_gate: str,
) -> None:
    summary = probe.summarize_retrieval_negation_confirmatory(
        _fake_matrix(
            spec,
            negated_induced=negated,
            contextual_induced=contextual,
            low_induced=low,
        )
    )
    actual_gates = {
        summary.negation_replication_gate.value,
        summary.matched_specificity_gate.value,
        summary.contextual_control_gate.value,
        summary.low_overlap_control_gate.value,
        summary.domain_breadth_gate.value,
    }
    assert failed_gate in actual_gates
    assert summary.baseline_eligibility_gate.value == "BASELINE_ELIGIBILITY_GATE_PASS"
    assert (
        summary.interpretation
        is probe.RetrievalNegationInterpretation.DOES_NOT_SUPPORT_H4_N
    )


def test_execution_result_recomputes_summary_and_serializes_deterministically(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> None:
    observations = _fake_matrix(
        spec,
        negated_induced=SUPPORT_NEGATED,
        contextual_induced=SUPPORT_CONTEXTUAL,
        low_induced=SUPPORT_LOW,
    )
    summary = probe.summarize_retrieval_negation_confirmatory(observations)
    result = probe.RetrievalNegationExecutionResult(
        schema_version=probe.RETRIEVAL_NEGATION_CONFIRMATORY_SCHEMA_VERSION,
        probe_id=probe.RETRIEVAL_NEGATION_CONFIRMATORY_ID,
        hypothesis_id=probe.RETRIEVAL_NEGATION_CONFIRMATORY_HYPOTHESIS_ID,
        fixture_sha256=probe.RETRIEVAL_NEGATION_CONFIRMATORY_FIXTURE_SHA256,
        scenarios=observations,
        summary=summary,
    )
    first = result.to_json()
    assert probe.RetrievalNegationExecutionResult.model_validate_json(first).to_json() == first
    assert "Which" not in first
    assert "CedarTrack" not in first

    mismatched_summary = probe.summarize_retrieval_negation_confirmatory(
        _fake_matrix(spec, negated_induced=SUPPORT_NEGATED - {"H4N-02"})
    )
    with pytest.raises(ValidationError, match="summary must be recomputed"):
        probe.RetrievalNegationExecutionResult(
            schema_version=probe.RETRIEVAL_NEGATION_CONFIRMATORY_SCHEMA_VERSION,
            probe_id=probe.RETRIEVAL_NEGATION_CONFIRMATORY_ID,
            hypothesis_id=probe.RETRIEVAL_NEGATION_CONFIRMATORY_HYPOTHESIS_ID,
            fixture_sha256=probe.RETRIEVAL_NEGATION_CONFIRMATORY_FIXTURE_SHA256,
            scenarios=observations,
            summary=mismatched_summary,
        )


def test_exactly_three_final_interpretations_are_frozen() -> None:
    assert [item.value for item in probe.RetrievalNegationInterpretation] == [
        "SUPPORTS_H4_N",
        "DOES_NOT_SUPPORT_H4_N",
        "INCONCLUSIVE_BASELINE_CONSTRUCTION",
    ]
