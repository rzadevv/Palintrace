"""Frozen models and accounting for the H4-N confirmatory retrieval probe."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from palintrace.retrieval import (
    PairedRetrievalChallengeAssessment,
    RetrievalChallengeOutcome,
    RetrievalObservation,
    RetrievalSufficiencyPolicy,
    assess_paired_retrieval_challenge,
)

RETRIEVAL_NEGATION_CONFIRMATORY_SCHEMA_VERSION = "0.1"
RETRIEVAL_NEGATION_CONFIRMATORY_ID = "retrieval-negation-confirmatory-v0.1"
RETRIEVAL_NEGATION_CONFIRMATORY_HYPOTHESIS_ID = "H4-N"
RETRIEVAL_NEGATION_CONFIRMATORY_SPLIT = "fresh_confirmatory_development"
RETRIEVAL_NEGATION_CONFIRMATORY_FIXTURE_PATH = Path(
    "tests/fixtures/retrieval_negation_confirmatory_v0.1.json"
)
RETRIEVAL_NEGATION_CONFIRMATORY_FIXTURE_SHA256 = (
    "fc5e1c442fb522d2c506b10cd7c05d42a99d3fbafc8493fd01c56fe1c88b1307"
)

FROZEN_RETRIEVER_KIND = "experimental_lexical"
FROZEN_RETRIEVER_VERSION = "0.1"
FROZEN_TOP_K = 3
FROZEN_POLICY = RetrievalSufficiencyPolicy.ALL_EXPECTED

BASELINE_ELIGIBILITY_MINIMUM = 17
NEGATION_INDUCED_MINIMUM = 12
NEGATION_SPECIFIC_MINIMUM = 8
REVERSE_SPECIFIC_MAXIMUM = 2
CONTEXTUAL_INDUCED_MAXIMUM = 5
LOW_OVERLAP_INDUCED_MAXIMUM = 1
DOMAIN_BREADTH_MINIMUM = 5

StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]
StrictUnitRatio = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]

_SIMPLE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_CONTEXTUAL_REJECTION_TOKENS = frozenset(
    {
        "not",
        "no",
        "never",
        "without",
        "reject",
        "rejected",
        "rejects",
        "avoid",
        "avoided",
        "avoids",
        "decline",
        "declined",
        "declines",
        "ruled",
    }
)


class RetrievalNegationConfirmatoryInputError(ValueError):
    """The frozen H4-N probe cannot be loaded or summarized safely."""


class RetrievalNegationCondition(StrEnum):
    """The three matched mutation conditions in canonical order."""

    NEGATED_COMPETING_VALUE = "negated_competing_value"
    CONTEXTUAL_COMPETING_VALUE_CONTROL = "contextual_competing_value_control"
    LOW_OVERLAP_CONTROL = "low_overlap_control"


class RetrievalNegationDomain(StrEnum):
    """The six organizational domains balanced across scenarios."""

    SOFTWARE_TOOL = "software_tool"
    LOCATION = "location"
    SCHEDULE = "schedule"
    DEVICE = "device"
    PROJECT_OR_WORK = "project_or_work"
    PREFERENCE_OR_SUBSCRIPTION = "preference_or_subscription"


class RetrievalNegationGateResult(StrEnum):
    """Named outcomes for the six preregistered confirmatory gates."""

    BASELINE_ELIGIBILITY_GATE_PASS = "BASELINE_ELIGIBILITY_GATE_PASS"
    BASELINE_ELIGIBILITY_GATE_FAIL = "BASELINE_ELIGIBILITY_GATE_FAIL"
    NEGATION_REPLICATION_GATE_PASS = "NEGATION_REPLICATION_GATE_PASS"
    NEGATION_REPLICATION_GATE_FAIL = "NEGATION_REPLICATION_GATE_FAIL"
    MATCHED_SPECIFICITY_GATE_PASS = "MATCHED_SPECIFICITY_GATE_PASS"
    MATCHED_SPECIFICITY_GATE_FAIL = "MATCHED_SPECIFICITY_GATE_FAIL"
    CONTEXTUAL_CONTROL_GATE_PASS = "CONTEXTUAL_CONTROL_GATE_PASS"
    CONTEXTUAL_CONTROL_GATE_FAIL = "CONTEXTUAL_CONTROL_GATE_FAIL"
    LOW_OVERLAP_CONTROL_GATE_PASS = "LOW_OVERLAP_CONTROL_GATE_PASS"
    LOW_OVERLAP_CONTROL_GATE_FAIL = "LOW_OVERLAP_CONTROL_GATE_FAIL"
    DOMAIN_BREADTH_GATE_PASS = "DOMAIN_BREADTH_GATE_PASS"
    DOMAIN_BREADTH_GATE_FAIL = "DOMAIN_BREADTH_GATE_FAIL"


class RetrievalNegationInterpretation(StrEnum):
    """The only allowed future H4-N interpretations."""

    SUPPORTS_H4_N = "SUPPORTS_H4_N"
    DOES_NOT_SUPPORT_H4_N = "DOES_NOT_SUPPORT_H4_N"
    INCONCLUSIVE_BASELINE_CONSTRUCTION = "INCONCLUSIVE_BASELINE_CONSTRUCTION"


CONDITION_ORDER = tuple(RetrievalNegationCondition)
DOMAIN_ORDER = tuple(RetrievalNegationDomain)
EXPECTED_SCENARIO_IDS = tuple(f"H4N-{index:02d}" for index in range(1, 19))


def _nonblank(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    if value != value.strip():
        raise ValueError(f"{field_name} must not have surrounding whitespace")
    return value


def simple_tokens(text: str) -> tuple[str, ...]:
    """Apply the preregistered model-free descriptive tokenizer."""

    return tuple(match.group(0).lower() for match in _SIMPLE_TOKEN_PATTERN.finditer(text))


def _query_intersection_count(query: str, content: str) -> int:
    return len(set(simple_tokens(query)) & set(simple_tokens(content)))


def _query_jaccard(query: str, content: str) -> float:
    query_tokens = set(simple_tokens(query))
    content_tokens = set(simple_tokens(content))
    return len(query_tokens & content_tokens) / len(query_tokens | content_tokens)


def _expected_domain(scenario_id: str) -> RetrievalNegationDomain:
    try:
        index = EXPECTED_SCENARIO_IDS.index(scenario_id)
    except ValueError as error:
        raise ValueError("scenario_id is outside the frozen inventory") from error
    return DOMAIN_ORDER[index // 3]


class _DeterministicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically without nonfinite values."""

        text = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )
        if indent is not None:
            text += "\n"
        return text


class RetrievalNegationRetrieverSpec(_DeterministicModel):
    """The exact non-tunable future execution setup."""

    kind: StrictStr
    version: StrictStr
    top_k: StrictPositiveInt
    policy: RetrievalSufficiencyPolicy

    @model_validator(mode="after")
    def condition_is_frozen(self) -> RetrievalNegationRetrieverSpec:
        if (
            self.kind != FROZEN_RETRIEVER_KIND
            or self.version != FROZEN_RETRIEVER_VERSION
            or self.top_k != FROZEN_TOP_K
            or self.policy is not FROZEN_POLICY
        ):
            raise ValueError("confirmatory retriever must match the frozen lexical setup")
        return self


class RetrievalNegationMemory(_DeterministicModel):
    """One synthetic memory without runtime retrieval data."""

    id: StrictStr
    content: StrictStr

    @field_validator("id", "content")
    @classmethod
    def strings_are_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="confirmatory memory string")


class RetrievalNegationConditions(_DeterministicModel):
    """Exactly eight distractors for each canonical condition."""

    negated_competing_value: tuple[RetrievalNegationMemory, ...]
    contextual_competing_value_control: tuple[RetrievalNegationMemory, ...]
    low_overlap_control: tuple[RetrievalNegationMemory, ...]

    @field_validator(
        "negated_competing_value",
        "contextual_competing_value_control",
        "low_overlap_control",
    )
    @classmethod
    def exactly_eight(
        cls,
        value: tuple[RetrievalNegationMemory, ...],
    ) -> tuple[RetrievalNegationMemory, ...]:
        if len(value) != 8:
            raise ValueError("each confirmatory condition requires exactly eight distractors")
        return value

    def for_condition(
        self,
        condition: RetrievalNegationCondition,
    ) -> tuple[RetrievalNegationMemory, ...]:
        """Return memories for one frozen condition without accepting free-form names."""

        if condition is RetrievalNegationCondition.NEGATED_COMPETING_VALUE:
            return self.negated_competing_value
        if condition is RetrievalNegationCondition.CONTEXTUAL_COMPETING_VALUE_CONTROL:
            return self.contextual_competing_value_control
        return self.low_overlap_control


class RetrievalNegationScenario(_DeterministicModel):
    """One shared four-memory baseline and three matched mutations."""

    scenario_id: StrictStr
    domain: RetrievalNegationDomain
    scope_user_id: StrictStr
    query: StrictStr
    target_value: StrictStr
    target_memory: RetrievalNegationMemory
    baseline_other_memories: tuple[RetrievalNegationMemory, ...]
    competing_values: tuple[StrictStr, ...]
    conditions: RetrievalNegationConditions

    @field_validator("scenario_id", "scope_user_id", "query", "target_value")
    @classmethod
    def scenario_strings_are_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="confirmatory scenario string")

    @field_validator("baseline_other_memories")
    @classmethod
    def exactly_three_baseline_non_targets(
        cls,
        value: tuple[RetrievalNegationMemory, ...],
    ) -> tuple[RetrievalNegationMemory, ...]:
        if len(value) != 3:
            raise ValueError("each scenario requires exactly three baseline non-targets")
        return value

    @field_validator("competing_values")
    @classmethod
    def exactly_eight_unique_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 8:
            raise ValueError("each scenario requires exactly eight competing values")
        for item in value:
            _nonblank(item, field_name="competing value")
        if len({item.casefold() for item in value}) != 8:
            raise ValueError("competing values must be unique")
        return value

    @property
    def expected_memory_ids(self) -> tuple[str, ...]:
        return (self.target_memory.id,)

    @property
    def baseline_memories(self) -> tuple[RetrievalNegationMemory, ...]:
        return (self.target_memory, *self.baseline_other_memories)

    def memories_for_condition(
        self,
        condition: RetrievalNegationCondition,
    ) -> tuple[RetrievalNegationMemory, ...]:
        return (*self.baseline_memories, *self.conditions.for_condition(condition))

    @model_validator(mode="after")
    def scenario_is_matched_and_unambiguous(self) -> RetrievalNegationScenario:
        if self.scenario_id not in EXPECTED_SCENARIO_IDS:
            raise ValueError("scenario_id is outside the frozen inventory")
        if self.domain is not _expected_domain(self.scenario_id):
            raise ValueError("scenario domain must match the frozen balanced order")
        prefix = self.scenario_id.lower()
        if self.target_memory.id != f"{prefix}-target":
            raise ValueError("target memory ID must be derived from scenario_id")
        if self.target_value.casefold() not in self.target_memory.content.casefold():
            raise ValueError("target memory must contain the declared target value")
        if self.target_value.casefold() in {
            value.casefold() for value in self.competing_values
        }:
            raise ValueError("target value and competing values must be disjoint")

        for index, memory in enumerate(self.baseline_other_memories, 1):
            if memory.id != f"{prefix}-base-{index:02d}":
                raise ValueError("baseline memory IDs must use canonical scenario order")

        all_memories = [
            self.target_memory,
            *self.baseline_other_memories,
            *self.conditions.negated_competing_value,
            *self.conditions.contextual_competing_value_control,
            *self.conditions.low_overlap_control,
        ]
        if len({memory.id for memory in all_memories}) != 28:
            raise ValueError("all 28 scenario memory IDs must be unique")
        if len({memory.content.casefold() for memory in all_memories}) != 28:
            raise ValueError("all 28 scenario memory contents must be unique")

        for index, (value, negated, contextual, low_overlap) in enumerate(
            zip(
                self.competing_values,
                self.conditions.negated_competing_value,
                self.conditions.contextual_competing_value_control,
                self.conditions.low_overlap_control,
                strict=True,
            ),
            1,
        ):
            if negated.id != f"{prefix}-neg-{index:02d}":
                raise ValueError("negated distractor IDs must use canonical pair order")
            if contextual.id != f"{prefix}-ctx-{index:02d}":
                raise ValueError("contextual distractor IDs must use canonical pair order")
            if low_overlap.id != f"{prefix}-low-{index:02d}":
                raise ValueError("low-overlap distractor IDs must use canonical order")
            value_folded = value.casefold()
            if (
                negated.content.casefold().count(value_folded) != 1
                or contextual.content.casefold().count(value_folded) != 1
            ):
                raise ValueError("paired distractors must contain the same value exactly once")
            if value_folded in low_overlap.content.casefold():
                raise ValueError("low-overlap controls cannot contain competing values")
            if "not" not in simple_tokens(negated.content):
                raise ValueError("negated distractors must explicitly contain not")
            if set(simple_tokens(contextual.content)) & _CONTEXTUAL_REJECTION_TOKENS:
                raise ValueError("contextual distractors cannot reject the competing value")
            if _query_intersection_count(self.query, negated.content) != (
                _query_intersection_count(self.query, contextual.content)
            ):
                raise ValueError("paired distractors must have equal simple query overlap")
            token_length_difference = abs(
                len(simple_tokens(negated.content))
                - len(simple_tokens(contextual.content))
            )
            if token_length_difference > 3:
                raise ValueError("paired distractor token lengths may differ by at most three")
            if (
                negated.content == self.target_memory.content
                or contextual.content == self.target_memory.content
            ):
                raise ValueError("paired distractors cannot duplicate the target")
            if negated.content.count(".") != 1 or contextual.content.count(".") != 1:
                raise ValueError("paired distractors must contain one simple proposition")

        competing_values_folded = tuple(value.casefold() for value in self.competing_values)
        for memory in self.conditions.low_overlap_control:
            if any(value in memory.content.casefold() for value in competing_values_folded):
                raise ValueError("low-overlap controls cannot contain competing values")

        low_median = median(
            _query_jaccard(self.query, memory.content)
            for memory in self.conditions.low_overlap_control
        )
        negated_median = median(
            _query_jaccard(self.query, memory.content)
            for memory in self.conditions.negated_competing_value
        )
        contextual_median = median(
            _query_jaccard(self.query, memory.content)
            for memory in self.conditions.contextual_competing_value_control
        )
        if not low_median < negated_median or not low_median < contextual_median:
            raise ValueError("low-overlap median must be below both matched conditions")
        return self


class RetrievalNegationConfirmatorySpec(_DeterministicModel):
    """The complete frozen 18-scenario confirmatory design."""

    schema_version: StrictStr
    probe_id: StrictStr
    hypothesis_id: StrictStr
    split: StrictStr
    retriever: RetrievalNegationRetrieverSpec
    scenarios: tuple[RetrievalNegationScenario, ...]

    @model_validator(mode="after")
    def inventory_is_exact(self) -> RetrievalNegationConfirmatorySpec:
        if (
            self.schema_version != RETRIEVAL_NEGATION_CONFIRMATORY_SCHEMA_VERSION
            or self.probe_id != RETRIEVAL_NEGATION_CONFIRMATORY_ID
            or self.hypothesis_id != RETRIEVAL_NEGATION_CONFIRMATORY_HYPOTHESIS_ID
            or self.split != RETRIEVAL_NEGATION_CONFIRMATORY_SPLIT
        ):
            raise ValueError("confirmatory probe identity must match the frozen design")
        if tuple(scenario.scenario_id for scenario in self.scenarios) != EXPECTED_SCENARIO_IDS:
            raise ValueError("confirmatory scenarios must use canonical IDs and order")
        if Counter(scenario.domain for scenario in self.scenarios) != {
            domain: 3 for domain in DOMAIN_ORDER
        }:
            raise ValueError("confirmatory probe requires three scenarios per domain")
        if len({scenario.query for scenario in self.scenarios}) != 18:
            raise ValueError("confirmatory queries must be globally unique")
        memories = [
            memory
            for scenario in self.scenarios
            for memory in (
                scenario.target_memory,
                *scenario.baseline_other_memories,
                *scenario.conditions.negated_competing_value,
                *scenario.conditions.contextual_competing_value_control,
                *scenario.conditions.low_overlap_control,
            )
        ]
        if len(memories) != 504:
            raise ValueError("confirmatory fixture must contain 504 scenario memories")
        if len({memory.id for memory in memories}) != 504:
            raise ValueError("confirmatory memory IDs must be globally unique")
        if len({memory.content.casefold() for memory in memories}) != 504:
            raise ValueError("confirmatory memory contents must be globally unique")
        answer_values = [
            value.casefold()
            for scenario in self.scenarios
            for value in (scenario.target_value, *scenario.competing_values)
        ]
        if len(answer_values) != 162 or len(set(answer_values)) != 162:
            raise ValueError("target and competing values must be globally unique")
        return self


def _validate_runtime_condition(
    observation: RetrievalObservation,
    *,
    expected_request_id: str,
    expected_memory_ids: tuple[str, ...],
    expected_candidate_count: int,
) -> None:
    if observation.request_id != expected_request_id:
        raise ValueError("runtime request ID must match scenario and condition")
    if (
        observation.expected_memory_ids != expected_memory_ids
        or observation.top_k != FROZEN_TOP_K
        or observation.retriever_id != FROZEN_RETRIEVER_KIND
        or observation.retriever_version != FROZEN_RETRIEVER_VERSION
    ):
        raise ValueError("runtime observation must match the frozen retrieval condition")
    if (
        observation.usage.retrieval_calls != 1
        or observation.usage.candidate_count != expected_candidate_count
    ):
        raise ValueError("runtime usage must record one call and the frozen corpus size")


class RetrievalNegationConditionObservation(_DeterministicModel):
    """One mutated observation and its stored paired assessment."""

    scenario_id: StrictStr
    condition: RetrievalNegationCondition
    mutated_observation: RetrievalObservation
    paired_assessment: PairedRetrievalChallengeAssessment


class RetrievalNegationScenarioObservation(_DeterministicModel):
    """One shared baseline paired with all three condition observations."""

    scenario_id: StrictStr
    domain: RetrievalNegationDomain
    baseline_observation: RetrievalObservation
    conditions: tuple[RetrievalNegationConditionObservation, ...]

    @model_validator(mode="after")
    def stored_pairs_recompute_exactly(self) -> RetrievalNegationScenarioObservation:
        if self.scenario_id not in EXPECTED_SCENARIO_IDS:
            raise ValueError("runtime scenario is outside the frozen inventory")
        if self.domain is not _expected_domain(self.scenario_id):
            raise ValueError("runtime scenario domain must match its ID")
        if tuple(item.condition for item in self.conditions) != CONDITION_ORDER:
            raise ValueError("runtime conditions must use canonical matched order")
        if any(item.scenario_id != self.scenario_id for item in self.conditions):
            raise ValueError("runtime condition scenario IDs must match their parent")

        target_ids = (f"{self.scenario_id.lower()}-target",)
        _validate_runtime_condition(
            self.baseline_observation,
            expected_request_id=f"{self.scenario_id}:baseline",
            expected_memory_ids=target_ids,
            expected_candidate_count=4,
        )
        for item in self.conditions:
            request_id = f"{self.scenario_id}:{item.condition.value}"
            _validate_runtime_condition(
                item.mutated_observation,
                expected_request_id=request_id,
                expected_memory_ids=target_ids,
                expected_candidate_count=12,
            )
            paired_case_id = request_id
            recomputed = assess_paired_retrieval_challenge(
                self.baseline_observation,
                item.mutated_observation,
                policy=FROZEN_POLICY,
                case_id=paired_case_id,
            )
            if item.paired_assessment != recomputed:
                raise ValueError("paired assessment must be recomputed from stored observations")
        return self


class RetrievalNegationConditionSummary(_DeterministicModel):
    """Outcome counts for one matched condition across 18 scenarios."""

    condition: RetrievalNegationCondition
    total_scenarios: StrictPositiveInt
    baseline_eligible_scenarios: StrictNonNegativeInt
    baseline_insufficient_scenarios: StrictNonNegativeInt
    induced_shadowing_scenarios: StrictNonNegativeInt
    resilient_scenarios: StrictNonNegativeInt

    @model_validator(mode="after")
    def counts_partition_scenarios(self) -> RetrievalNegationConditionSummary:
        if self.total_scenarios != 18:
            raise ValueError("condition summary must cover 18 matched scenarios")
        if self.baseline_eligible_scenarios + self.baseline_insufficient_scenarios != 18:
            raise ValueError("condition baseline counts must partition 18 scenarios")
        if (
            self.induced_shadowing_scenarios + self.resilient_scenarios
            != self.baseline_eligible_scenarios
        ):
            raise ValueError("eligible condition outcomes must partition eligible scenarios")
        return self


class RetrievalNegationConfirmatorySummary(_DeterministicModel):
    """Scenario-level matched counts, gates, and frozen interpretation."""

    scenario_total: StrictPositiveInt
    baseline_eligible_scenarios: StrictNonNegativeInt
    baseline_insufficient_scenarios: StrictNonNegativeInt
    condition_summaries: tuple[RetrievalNegationConditionSummary, ...]
    negated_induced_rate: StrictUnitRatio | None
    negation_specific_scenarios: StrictNonNegativeInt
    reverse_specific_scenarios: StrictNonNegativeInt
    domains_with_negated_induced: tuple[RetrievalNegationDomain, ...]
    baseline_eligibility_gate: RetrievalNegationGateResult
    negation_replication_gate: RetrievalNegationGateResult
    matched_specificity_gate: RetrievalNegationGateResult
    contextual_control_gate: RetrievalNegationGateResult
    low_overlap_control_gate: RetrievalNegationGateResult
    domain_breadth_gate: RetrievalNegationGateResult
    interpretation: RetrievalNegationInterpretation

    @field_validator("negated_induced_rate", mode="before")
    @classmethod
    def rate_is_python_float(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("negated induced rate must be a Python float or null")
        return value

    @model_validator(mode="after")
    def gates_are_mechanical(self) -> RetrievalNegationConfirmatorySummary:
        if self.scenario_total != 18:
            raise ValueError("confirmatory summary must cover 18 scenarios")
        if self.baseline_eligible_scenarios + self.baseline_insufficient_scenarios != 18:
            raise ValueError("summary baseline counts must partition 18 scenarios")
        if tuple(item.condition for item in self.condition_summaries) != CONDITION_ORDER:
            raise ValueError("condition summaries must use canonical order")
        if any(
            item.baseline_eligible_scenarios != self.baseline_eligible_scenarios
            or item.baseline_insufficient_scenarios != self.baseline_insufficient_scenarios
            for item in self.condition_summaries
        ):
            raise ValueError("all conditions must share scenario baseline eligibility")
        if tuple(dict.fromkeys(self.domains_with_negated_induced)) != (
            self.domains_with_negated_induced
        ):
            raise ValueError("affected domains must be unique and canonical")
        if tuple(
            domain for domain in DOMAIN_ORDER if domain in self.domains_with_negated_induced
        ) != self.domains_with_negated_induced:
            raise ValueError("affected domains must follow canonical domain order")
        if (
            self.negation_specific_scenarios + self.reverse_specific_scenarios
            > self.baseline_eligible_scenarios
        ):
            raise ValueError("matched discordances cannot exceed eligible scenarios")

        summaries = {item.condition: item for item in self.condition_summaries}
        negated_induced = summaries[
            RetrievalNegationCondition.NEGATED_COMPETING_VALUE
        ].induced_shadowing_scenarios
        contextual_induced = summaries[
            RetrievalNegationCondition.CONTEXTUAL_COMPETING_VALUE_CONTROL
        ].induced_shadowing_scenarios
        low_induced = summaries[
            RetrievalNegationCondition.LOW_OVERLAP_CONTROL
        ].induced_shadowing_scenarios
        expected_rate = (
            None
            if self.baseline_eligible_scenarios == 0
            else negated_induced / self.baseline_eligible_scenarios
        )
        if self.negated_induced_rate != expected_rate:
            raise ValueError("negated induced rate must match scenario counts")

        baseline_pass = self.baseline_eligible_scenarios >= BASELINE_ELIGIBILITY_MINIMUM
        replication_pass = (
            negated_induced >= NEGATION_INDUCED_MINIMUM
            and self.baseline_eligible_scenarios > 0
            and negated_induced * 3 >= self.baseline_eligible_scenarios * 2
        )
        specificity_pass = (
            self.negation_specific_scenarios >= NEGATION_SPECIFIC_MINIMUM
            and self.reverse_specific_scenarios <= REVERSE_SPECIFIC_MAXIMUM
        )
        contextual_pass = contextual_induced <= CONTEXTUAL_INDUCED_MAXIMUM
        low_pass = low_induced <= LOW_OVERLAP_INDUCED_MAXIMUM
        breadth_pass = len(self.domains_with_negated_induced) >= DOMAIN_BREADTH_MINIMUM
        expected_gates = (
            RetrievalNegationGateResult.BASELINE_ELIGIBILITY_GATE_PASS
            if baseline_pass
            else RetrievalNegationGateResult.BASELINE_ELIGIBILITY_GATE_FAIL,
            RetrievalNegationGateResult.NEGATION_REPLICATION_GATE_PASS
            if replication_pass
            else RetrievalNegationGateResult.NEGATION_REPLICATION_GATE_FAIL,
            RetrievalNegationGateResult.MATCHED_SPECIFICITY_GATE_PASS
            if specificity_pass
            else RetrievalNegationGateResult.MATCHED_SPECIFICITY_GATE_FAIL,
            RetrievalNegationGateResult.CONTEXTUAL_CONTROL_GATE_PASS
            if contextual_pass
            else RetrievalNegationGateResult.CONTEXTUAL_CONTROL_GATE_FAIL,
            RetrievalNegationGateResult.LOW_OVERLAP_CONTROL_GATE_PASS
            if low_pass
            else RetrievalNegationGateResult.LOW_OVERLAP_CONTROL_GATE_FAIL,
            RetrievalNegationGateResult.DOMAIN_BREADTH_GATE_PASS
            if breadth_pass
            else RetrievalNegationGateResult.DOMAIN_BREADTH_GATE_FAIL,
        )
        actual_gates = (
            self.baseline_eligibility_gate,
            self.negation_replication_gate,
            self.matched_specificity_gate,
            self.contextual_control_gate,
            self.low_overlap_control_gate,
            self.domain_breadth_gate,
        )
        if actual_gates != expected_gates:
            raise ValueError("confirmatory gates must match preregistered arithmetic")
        all_confirmatory_pass = all(
            gate.value.endswith("_PASS") for gate in actual_gates[1:]
        )
        expected_interpretation = (
            RetrievalNegationInterpretation.INCONCLUSIVE_BASELINE_CONSTRUCTION
            if not baseline_pass
            else RetrievalNegationInterpretation.SUPPORTS_H4_N
            if all_confirmatory_pass
            else RetrievalNegationInterpretation.DOES_NOT_SUPPORT_H4_N
        )
        if self.interpretation is not expected_interpretation:
            raise ValueError("interpretation must follow the frozen three-way rule")
        return self


class RetrievalNegationExecutionResult(_DeterministicModel):
    """Future deterministic 6I-B result without query or memory text."""

    schema_version: StrictStr
    probe_id: StrictStr
    hypothesis_id: StrictStr
    fixture_sha256: StrictStr
    scenarios: tuple[RetrievalNegationScenarioObservation, ...]
    summary: RetrievalNegationConfirmatorySummary

    @model_validator(mode="after")
    def result_recomputes_exactly(self) -> RetrievalNegationExecutionResult:
        if (
            self.schema_version != RETRIEVAL_NEGATION_CONFIRMATORY_SCHEMA_VERSION
            or self.probe_id != RETRIEVAL_NEGATION_CONFIRMATORY_ID
            or self.hypothesis_id != RETRIEVAL_NEGATION_CONFIRMATORY_HYPOTHESIS_ID
            or self.fixture_sha256 != RETRIEVAL_NEGATION_CONFIRMATORY_FIXTURE_SHA256
        ):
            raise ValueError("execution result identity must match the frozen probe")
        if tuple(item.scenario_id for item in self.scenarios) != EXPECTED_SCENARIO_IDS:
            raise ValueError("execution result must contain canonical scenario order")
        if self.summary != summarize_retrieval_negation_confirmatory(self.scenarios):
            raise ValueError("execution summary must be recomputed from validated scenarios")
        return self


def _condition_outcome_counts(
    scenarios: tuple[RetrievalNegationScenarioObservation, ...],
    condition: RetrievalNegationCondition,
) -> tuple[int, int, int]:
    assessments = tuple(
        next(item for item in scenario.conditions if item.condition is condition)
        .paired_assessment
        for scenario in scenarios
    )
    induced = sum(
        item.outcome is RetrievalChallengeOutcome.INDUCED_SHADOWING
        for item in assessments
    )
    resilient = sum(
        item.outcome is RetrievalChallengeOutcome.RESILIENT for item in assessments
    )
    insufficient = sum(
        item.outcome is RetrievalChallengeOutcome.BASELINE_INSUFFICIENT
        for item in assessments
    )
    return induced, resilient, insufficient


def summarize_retrieval_negation_confirmatory(
    scenarios: tuple[RetrievalNegationScenarioObservation, ...],
) -> RetrievalNegationConfirmatorySummary:
    """Apply the six frozen H4-N gates to 18 matched scenario observations."""

    if not isinstance(scenarios, tuple) or any(
        not isinstance(item, RetrievalNegationScenarioObservation) for item in scenarios
    ):
        raise RetrievalNegationConfirmatoryInputError(
            "summary requires a tuple of confirmatory scenario observations"
        )
    if tuple(item.scenario_id for item in scenarios) != EXPECTED_SCENARIO_IDS:
        raise RetrievalNegationConfirmatoryInputError(
            "summary requires all scenarios in canonical order"
        )

    first_assessments = tuple(scenario.conditions[0].paired_assessment for scenario in scenarios)
    baseline_eligible = sum(item.baseline_sufficient for item in first_assessments)
    baseline_insufficient = len(scenarios) - baseline_eligible
    condition_summaries: list[RetrievalNegationConditionSummary] = []
    outcomes_by_scenario: dict[
        str,
        dict[RetrievalNegationCondition, RetrievalChallengeOutcome],
    ] = {}
    for condition in CONDITION_ORDER:
        induced, resilient, insufficient = _condition_outcome_counts(scenarios, condition)
        condition_summaries.append(
            RetrievalNegationConditionSummary(
                condition=condition,
                total_scenarios=len(scenarios),
                baseline_eligible_scenarios=induced + resilient,
                baseline_insufficient_scenarios=insufficient,
                induced_shadowing_scenarios=induced,
                resilient_scenarios=resilient,
            )
        )
    for scenario in scenarios:
        outcomes_by_scenario[scenario.scenario_id] = {
            item.condition: item.paired_assessment.outcome for item in scenario.conditions
        }

    negation_specific = sum(
        outcomes[RetrievalNegationCondition.NEGATED_COMPETING_VALUE]
        is RetrievalChallengeOutcome.INDUCED_SHADOWING
        and outcomes[RetrievalNegationCondition.CONTEXTUAL_COMPETING_VALUE_CONTROL]
        is RetrievalChallengeOutcome.RESILIENT
        for outcomes in outcomes_by_scenario.values()
    )
    reverse_specific = sum(
        outcomes[RetrievalNegationCondition.NEGATED_COMPETING_VALUE]
        is RetrievalChallengeOutcome.RESILIENT
        and outcomes[RetrievalNegationCondition.CONTEXTUAL_COMPETING_VALUE_CONTROL]
        is RetrievalChallengeOutcome.INDUCED_SHADOWING
        for outcomes in outcomes_by_scenario.values()
    )
    affected_domains = tuple(
        domain
        for domain in DOMAIN_ORDER
        if any(
            scenario.domain is domain
            and outcomes_by_scenario[scenario.scenario_id][
                RetrievalNegationCondition.NEGATED_COMPETING_VALUE
            ]
            is RetrievalChallengeOutcome.INDUCED_SHADOWING
            for scenario in scenarios
        )
    )

    summary_by_condition = {item.condition: item for item in condition_summaries}
    negated_induced = summary_by_condition[
        RetrievalNegationCondition.NEGATED_COMPETING_VALUE
    ].induced_shadowing_scenarios
    contextual_induced = summary_by_condition[
        RetrievalNegationCondition.CONTEXTUAL_COMPETING_VALUE_CONTROL
    ].induced_shadowing_scenarios
    low_induced = summary_by_condition[
        RetrievalNegationCondition.LOW_OVERLAP_CONTROL
    ].induced_shadowing_scenarios
    baseline_pass = baseline_eligible >= BASELINE_ELIGIBILITY_MINIMUM
    replication_pass = (
        negated_induced >= NEGATION_INDUCED_MINIMUM
        and baseline_eligible > 0
        and negated_induced * 3 >= baseline_eligible * 2
    )
    specificity_pass = (
        negation_specific >= NEGATION_SPECIFIC_MINIMUM
        and reverse_specific <= REVERSE_SPECIFIC_MAXIMUM
    )
    contextual_pass = contextual_induced <= CONTEXTUAL_INDUCED_MAXIMUM
    low_pass = low_induced <= LOW_OVERLAP_INDUCED_MAXIMUM
    breadth_pass = len(affected_domains) >= DOMAIN_BREADTH_MINIMUM
    interpretation = (
        RetrievalNegationInterpretation.INCONCLUSIVE_BASELINE_CONSTRUCTION
        if not baseline_pass
        else RetrievalNegationInterpretation.SUPPORTS_H4_N
        if replication_pass
        and specificity_pass
        and contextual_pass
        and low_pass
        and breadth_pass
        else RetrievalNegationInterpretation.DOES_NOT_SUPPORT_H4_N
    )
    return RetrievalNegationConfirmatorySummary(
        scenario_total=len(scenarios),
        baseline_eligible_scenarios=baseline_eligible,
        baseline_insufficient_scenarios=baseline_insufficient,
        condition_summaries=tuple(condition_summaries),
        negated_induced_rate=(
            None if baseline_eligible == 0 else negated_induced / baseline_eligible
        ),
        negation_specific_scenarios=negation_specific,
        reverse_specific_scenarios=reverse_specific,
        domains_with_negated_induced=affected_domains,
        baseline_eligibility_gate=(
            RetrievalNegationGateResult.BASELINE_ELIGIBILITY_GATE_PASS
            if baseline_pass
            else RetrievalNegationGateResult.BASELINE_ELIGIBILITY_GATE_FAIL
        ),
        negation_replication_gate=(
            RetrievalNegationGateResult.NEGATION_REPLICATION_GATE_PASS
            if replication_pass
            else RetrievalNegationGateResult.NEGATION_REPLICATION_GATE_FAIL
        ),
        matched_specificity_gate=(
            RetrievalNegationGateResult.MATCHED_SPECIFICITY_GATE_PASS
            if specificity_pass
            else RetrievalNegationGateResult.MATCHED_SPECIFICITY_GATE_FAIL
        ),
        contextual_control_gate=(
            RetrievalNegationGateResult.CONTEXTUAL_CONTROL_GATE_PASS
            if contextual_pass
            else RetrievalNegationGateResult.CONTEXTUAL_CONTROL_GATE_FAIL
        ),
        low_overlap_control_gate=(
            RetrievalNegationGateResult.LOW_OVERLAP_CONTROL_GATE_PASS
            if low_pass
            else RetrievalNegationGateResult.LOW_OVERLAP_CONTROL_GATE_FAIL
        ),
        domain_breadth_gate=(
            RetrievalNegationGateResult.DOMAIN_BREADTH_GATE_PASS
            if breadth_pass
            else RetrievalNegationGateResult.DOMAIN_BREADTH_GATE_FAIL
        ),
        interpretation=interpretation,
    )


def sha256_file(path: Path) -> str:
    """Return the byte-level SHA-256 of one frozen input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_retrieval_negation_confirmatory(
    path: Path,
) -> RetrievalNegationConfirmatorySpec:
    """Load and validate a fixture without executing retrieval."""

    try:
        return RetrievalNegationConfirmatorySpec.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as error:
        raise RetrievalNegationConfirmatoryInputError(
            f"invalid negation confirmatory fixture: {path}"
        ) from error


def preflight_retrieval_negation_confirmatory(
    path: Path,
) -> RetrievalNegationConfirmatorySpec:
    """Verify frozen fixture bytes before parsing or retriever construction."""

    try:
        actual_sha256 = sha256_file(path)
    except OSError as error:
        raise RetrievalNegationConfirmatoryInputError(
            f"cannot read negation confirmatory fixture: {path}"
        ) from error
    if actual_sha256 != RETRIEVAL_NEGATION_CONFIRMATORY_FIXTURE_SHA256:
        raise RetrievalNegationConfirmatoryInputError(
            "negation confirmatory fixture SHA mismatch: "
            f"expected {RETRIEVAL_NEGATION_CONFIRMATORY_FIXTURE_SHA256}, "
            f"got {actual_sha256}"
        )
    return load_retrieval_negation_confirmatory(path)
