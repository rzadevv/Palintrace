"""Models and pure accounting for the preregistered strong retrieval probe."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
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

from memlint.retrieval import (
    PairedRetrievalChallengeAssessment,
    RetrievalChallengeOutcome,
    RetrievalObservation,
    RetrievalSufficiencyPolicy,
    assess_paired_retrieval_challenge,
)

RETRIEVAL_STRONG_PROBE_SCHEMA_VERSION = "0.1"
RETRIEVAL_STRONG_PROBE_ID = "retrieval-shadowing-strong-probe-v0.1"
RETRIEVAL_STRONG_PROBE_SPLIT = "development"
RETRIEVAL_STRONG_PROBE_FIXTURE_PATH = Path(
    "tests/fixtures/retrieval_shadowing_strong_probe_v0.1.json"
)
RETRIEVAL_STRONG_PROBE_FIXTURE_SHA256 = (
    "98c2a6e1f1f5a38f34691918dd99a1eeb9096888ca2c02d63469595820c87748"
)

FROZEN_RETRIEVER_KIND = "experimental_lexical"
FROZEN_RETRIEVER_VERSION = "0.1"
FROZEN_TOP_K = 3
FROZEN_POLICY = RetrievalSufficiencyPolicy.ALL_EXPECTED

BASELINE_ELIGIBILITY_MINIMUM = 20
STRONG_INDUCED_MINIMUM = 8
STRONG_INDUCED_RATE_MINIMUM = 0.40
FAMILY_BREADTH_MINIMUM = 2
CONTROL_BASELINE_ELIGIBILITY_MINIMUM = 5
CONTROL_INDUCED_MAXIMUM = 1

StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]
StrictUnitRatio = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]


class RetrievalStrongProbeInputError(ValueError):
    """The frozen probe cannot be loaded or summarized safely."""


class RetrievalStrongProbeCaseKind(StrEnum):
    """Preregistered experimental role for one case."""

    STRONG_CHALLENGE = "strong_challenge"
    RESILIENCE_CONTROL = "resilience_control"


class RetrievalStrongProbeChallengeFamily(StrEnum):
    """Exactly three strong mechanisms plus the low-overlap control family."""

    QUERY_TERM_CROWDING = "query_term_crowding"
    NEGATED_VALUE_DECOYS = "negated_value_decoys"
    CONTEXTUAL_MENTION_DECOYS = "contextual_mention_decoys"
    LOW_OVERLAP_CONTROL = "low_overlap_control"


class RetrievalStrongProbeDomain(StrEnum):
    """Frozen balanced synthetic domains."""

    SOFTWARE_TOOL = "software_tool"
    LOCATION = "location"
    SCHEDULE = "schedule"
    DEVICE = "device"
    PROJECT_OR_WORK = "project_or_work"
    PREFERENCE_OR_SUBSCRIPTION = "preference_or_subscription"


class RetrievalStrongProbeGateResult(StrEnum):
    """Named pass/fail outcomes for the four preregistered gates."""

    BASELINE_ELIGIBILITY_GATE_PASS = "BASELINE_ELIGIBILITY_GATE_PASS"
    BASELINE_ELIGIBILITY_GATE_FAIL = "BASELINE_ELIGIBILITY_GATE_FAIL"
    STRONG_SHADOWING_GATE_PASS = "STRONG_SHADOWING_GATE_PASS"
    STRONG_SHADOWING_GATE_FAIL = "STRONG_SHADOWING_GATE_FAIL"
    FAMILY_BREADTH_GATE_PASS = "FAMILY_BREADTH_GATE_PASS"
    FAMILY_BREADTH_GATE_FAIL = "FAMILY_BREADTH_GATE_FAIL"
    CONTROL_STABILITY_GATE_PASS = "CONTROL_STABILITY_GATE_PASS"
    CONTROL_STABILITY_GATE_FAIL = "CONTROL_STABILITY_GATE_FAIL"


class RetrievalStrongProbeInterpretation(StrEnum):
    """Allowed interpretations for the strong retrieval probe."""

    SUPPORTS_H4 = "SUPPORTS_H4"
    DOES_NOT_SUPPORT_H4 = "DOES_NOT_SUPPORT_H4"
    INCONCLUSIVE_BASELINE_CONSTRUCTION = "INCONCLUSIVE_BASELINE_CONSTRUCTION"


STRONG_FAMILIES = (
    RetrievalStrongProbeChallengeFamily.QUERY_TERM_CROWDING,
    RetrievalStrongProbeChallengeFamily.NEGATED_VALUE_DECOYS,
    RetrievalStrongProbeChallengeFamily.CONTEXTUAL_MENTION_DECOYS,
)
EXPECTED_CASE_IDS = tuple(
    [f"RS-Q{index:02d}" for index in range(1, 9)]
    + [f"RS-N{index:02d}" for index in range(1, 9)]
    + [f"RS-C{index:02d}" for index in range(1, 9)]
    + [f"RS-R{index:02d}" for index in range(1, 7)]
)


def _nonblank(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    if value != value.strip():
        raise ValueError(f"{field_name} must not have surrounding whitespace")
    return value


class _DeterministicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically without nonfinite JSON values."""

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


class RetrievalStrongProbeRetrieverSpec(_DeterministicModel):
    """Non-tunable identity of the execution retriever."""

    kind: StrictStr
    version: StrictStr
    top_k: StrictPositiveInt
    policy: RetrievalSufficiencyPolicy

    @model_validator(mode="after")
    def condition_is_exactly_frozen(self) -> RetrievalStrongProbeRetrieverSpec:
        if (
            self.kind != FROZEN_RETRIEVER_KIND
            or self.version != FROZEN_RETRIEVER_VERSION
            or self.top_k != FROZEN_TOP_K
            or self.policy is not FROZEN_POLICY
        ):
            raise ValueError("retrieval probe condition must match the frozen lexical setup")
        return self


class RetrievalStrongProbeMemory(_DeterministicModel):
    """One synthetic memory without scores, labels, or backend metadata."""

    id: StrictStr
    content: StrictStr

    @field_validator("id", "content")
    @classmethod
    def memory_strings_are_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="probe memory string")


def _expected_family(case_id: str) -> RetrievalStrongProbeChallengeFamily:
    prefix = case_id[3:4]
    mapping = {
        "Q": RetrievalStrongProbeChallengeFamily.QUERY_TERM_CROWDING,
        "N": RetrievalStrongProbeChallengeFamily.NEGATED_VALUE_DECOYS,
        "C": RetrievalStrongProbeChallengeFamily.CONTEXTUAL_MENTION_DECOYS,
        "R": RetrievalStrongProbeChallengeFamily.LOW_OVERLAP_CONTROL,
    }
    try:
        return mapping[prefix]
    except KeyError as error:  # pragma: no cover - exact ID inventory catches this
        raise ValueError(f"unsupported retrieval probe case ID: {case_id!r}") from error


class RetrievalStrongProbeCase(_DeterministicModel):
    """One frozen baseline store and its eight synthetic distractors."""

    case_id: StrictStr
    case_kind: RetrievalStrongProbeCaseKind
    challenge_family: RetrievalStrongProbeChallengeFamily
    domain: RetrievalStrongProbeDomain
    scope_user_id: StrictStr
    query: StrictStr
    target_memory: RetrievalStrongProbeMemory
    baseline_other_memories: tuple[RetrievalStrongProbeMemory, ...]
    distractor_memories: tuple[RetrievalStrongProbeMemory, ...]
    expected_memory_ids: tuple[StrictStr, ...]
    top_k: StrictPositiveInt
    policy: RetrievalSufficiencyPolicy
    target_value: StrictStr | None = None
    distractor_values: tuple[StrictStr, ...] = ()

    @field_validator("case_id", "scope_user_id", "query")
    @classmethod
    def case_strings_are_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="probe case string")

    @field_validator("baseline_other_memories")
    @classmethod
    def exactly_three_baseline_non_targets(
        cls,
        value: tuple[RetrievalStrongProbeMemory, ...],
    ) -> tuple[RetrievalStrongProbeMemory, ...]:
        if len(value) != 3:
            raise ValueError("each probe case requires exactly three baseline non-targets")
        return value

    @field_validator("distractor_memories")
    @classmethod
    def exactly_eight_distractors(
        cls,
        value: tuple[RetrievalStrongProbeMemory, ...],
    ) -> tuple[RetrievalStrongProbeMemory, ...]:
        if len(value) != 8:
            raise ValueError("each probe case requires exactly eight distractors")
        return value

    @field_validator("expected_memory_ids")
    @classmethod
    def exactly_one_expected_target(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 1 or not value[0].strip():
            raise ValueError("each probe case requires exactly one expected target")
        return value

    @field_validator("target_value")
    @classmethod
    def target_value_is_nonblank(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, field_name="target_value")

    @field_validator("distractor_values")
    @classmethod
    def distractor_values_are_nonblank_and_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        for item in value:
            _nonblank(item, field_name="distractor value")
        if len(set(item.casefold() for item in value)) != len(value):
            raise ValueError("distractor values must be unique")
        return value

    @model_validator(mode="after")
    def case_structure_is_frozen(self) -> RetrievalStrongProbeCase:
        if self.case_id not in EXPECTED_CASE_IDS:
            raise ValueError("case_id is outside the frozen 30-case inventory")
        expected_family = _expected_family(self.case_id)
        if self.challenge_family is not expected_family:
            raise ValueError("case family must match its deterministic case ID")
        expected_kind = (
            RetrievalStrongProbeCaseKind.RESILIENCE_CONTROL
            if expected_family is RetrievalStrongProbeChallengeFamily.LOW_OVERLAP_CONTROL
            else RetrievalStrongProbeCaseKind.STRONG_CHALLENGE
        )
        if self.case_kind is not expected_kind:
            raise ValueError("case kind must match its frozen challenge family")
        if self.top_k != FROZEN_TOP_K or self.policy is not FROZEN_POLICY:
            raise ValueError("every case must freeze top_k=3 and ALL_EXPECTED")

        stem = self.case_id.lower()
        if self.scope_user_id != f"probe-user-{stem}":
            raise ValueError("scope_user_id must be deterministic and case-local")
        if self.target_memory.id != f"{stem}-target":
            raise ValueError("target memory ID must be deterministic")
        if tuple(memory.id for memory in self.baseline_other_memories) != tuple(
            f"{stem}-base-{index:02d}" for index in range(1, 4)
        ):
            raise ValueError("baseline memory IDs or ordering are not canonical")
        if tuple(memory.id for memory in self.distractor_memories) != tuple(
            f"{stem}-dist-{index:02d}" for index in range(1, 9)
        ):
            raise ValueError("distractor memory IDs or ordering are not canonical")
        if self.expected_memory_ids != (self.target_memory.id,):
            raise ValueError("expected_memory_ids must contain only the target")

        memories = (
            self.target_memory,
            *self.baseline_other_memories,
            *self.distractor_memories,
        )
        memory_ids = [memory.id for memory in memories]
        contents = [memory.content for memory in memories]
        if len(set(memory_ids)) != 12:
            raise ValueError("all case memory IDs must be unique")
        if len(set(content.casefold() for content in contents)) != 12:
            raise ValueError("all case memory contents must be distinct")

        negated = (
            self.challenge_family
            is RetrievalStrongProbeChallengeFamily.NEGATED_VALUE_DECOYS
        )
        if negated:
            if self.target_value is None or len(self.distractor_values) != 8:
                raise ValueError("negated-value cases require one target and eight decoy values")
            target_value = self.target_value.casefold()
            if target_value not in self.target_memory.content.casefold():
                raise ValueError("target_value must occur in the target memory")
            if target_value in {value.casefold() for value in self.distractor_values}:
                raise ValueError("target_value must differ from every distractor value")
            for memory, value in zip(
                self.distractor_memories,
                self.distractor_values,
                strict=True,
            ):
                if value.casefold() not in memory.content.casefold():
                    raise ValueError("each decoy value must occur in its paired distractor")
                if target_value in memory.content.casefold():
                    raise ValueError("the positive target value must not appear in a decoy")
        elif self.target_value is not None or self.distractor_values:
            raise ValueError("value annotations are reserved for negated-value cases")
        return self


class RetrievalStrongProbeSpec(_DeterministicModel):
    """The complete immutable 30-case development preregistration."""

    schema_version: StrictStr
    probe_id: StrictStr
    split: StrictStr
    retriever: RetrievalStrongProbeRetrieverSpec
    cases: tuple[RetrievalStrongProbeCase, ...]

    @model_validator(mode="after")
    def exact_design_is_frozen(self) -> RetrievalStrongProbeSpec:
        if (
            self.schema_version != RETRIEVAL_STRONG_PROBE_SCHEMA_VERSION
            or self.probe_id != RETRIEVAL_STRONG_PROBE_ID
            or self.split != RETRIEVAL_STRONG_PROBE_SPLIT
        ):
            raise ValueError("probe identity must match the preregistered contract")
        if tuple(case.case_id for case in self.cases) != EXPECTED_CASE_IDS:
            raise ValueError("cases must use the exact canonical 30-case order")

        family_counts = Counter(case.challenge_family for case in self.cases)
        if family_counts != Counter(
            {
                RetrievalStrongProbeChallengeFamily.QUERY_TERM_CROWDING: 8,
                RetrievalStrongProbeChallengeFamily.NEGATED_VALUE_DECOYS: 8,
                RetrievalStrongProbeChallengeFamily.CONTEXTUAL_MENTION_DECOYS: 8,
                RetrievalStrongProbeChallengeFamily.LOW_OVERLAP_CONTROL: 6,
            }
        ):
            raise ValueError("probe family counts do not match the preregistration")
        kind_counts = Counter(case.case_kind for case in self.cases)
        if kind_counts != Counter(
            {
                RetrievalStrongProbeCaseKind.STRONG_CHALLENGE: 24,
                RetrievalStrongProbeCaseKind.RESILIENCE_CONTROL: 6,
            }
        ):
            raise ValueError("probe kind counts do not match the preregistration")

        strong_domain_counts = Counter(
            case.domain
            for case in self.cases
            if case.case_kind is RetrievalStrongProbeCaseKind.STRONG_CHALLENGE
        )
        control_domain_counts = Counter(
            case.domain
            for case in self.cases
            if case.case_kind is RetrievalStrongProbeCaseKind.RESILIENCE_CONTROL
        )
        expected_domains = set(RetrievalStrongProbeDomain)
        if set(strong_domain_counts) != expected_domains or any(
            count != 4 for count in strong_domain_counts.values()
        ):
            raise ValueError("each domain must contain exactly four strong cases")
        if set(control_domain_counts) != expected_domains or any(
            count != 1 for count in control_domain_counts.values()
        ):
            raise ValueError("each domain must contain exactly one resilience control")

        queries = [case.query.casefold() for case in self.cases]
        memories = [
            memory
            for case in self.cases
            for memory in (
                case.target_memory,
                *case.baseline_other_memories,
                *case.distractor_memories,
            )
        ]
        if len(set(queries)) != len(queries):
            raise ValueError("probe queries must be globally unique")
        if len({memory.id for memory in memories}) != len(memories):
            raise ValueError("probe memory IDs must be globally unique")
        if len({memory.content.casefold() for memory in memories}) != len(memories):
            raise ValueError("probe memory contents must be globally unique")
        return self


class RetrievalStrongProbeObservation(_DeterministicModel):
    """One paired runtime observation with frozen synthetic case metadata."""

    case_id: StrictStr
    case_kind: RetrievalStrongProbeCaseKind
    challenge_family: RetrievalStrongProbeChallengeFamily
    domain: RetrievalStrongProbeDomain
    baseline_observation: RetrievalObservation
    mutated_observation: RetrievalObservation
    paired_assessment: PairedRetrievalChallengeAssessment

    @model_validator(mode="after")
    def observations_match_case(self) -> RetrievalStrongProbeObservation:
        if self.case_id not in EXPECTED_CASE_IDS:
            raise ValueError("observation case_id is outside the frozen inventory")
        if self.challenge_family is not _expected_family(self.case_id):
            raise ValueError("observation family must match its case ID")
        expected_kind = (
            RetrievalStrongProbeCaseKind.RESILIENCE_CONTROL
            if self.challenge_family
            is RetrievalStrongProbeChallengeFamily.LOW_OVERLAP_CONTROL
            else RetrievalStrongProbeCaseKind.STRONG_CHALLENGE
        )
        if self.case_kind is not expected_kind:
            raise ValueError("observation kind must match its case ID")
        if self.case_id != self.paired_assessment.case_id:
            raise ValueError("observation and paired assessment case IDs must match")
        if (
            self.baseline_observation.request_id
            != self.paired_assessment.baseline_request_id
            or self.mutated_observation.request_id
            != self.paired_assessment.mutated_request_id
        ):
            raise ValueError("observation request IDs must match the paired assessment")
        if (
            self.baseline_observation.request_id != f"{self.case_id}:baseline"
            or self.mutated_observation.request_id != f"{self.case_id}:mutated"
        ):
            raise ValueError("observation request IDs must use the frozen condition suffixes")
        assessment = self.paired_assessment
        if (
            assessment.policy is not FROZEN_POLICY
            or assessment.top_k != FROZEN_TOP_K
            or assessment.retriever_id != FROZEN_RETRIEVER_KIND
            or assessment.retriever_version != FROZEN_RETRIEVER_VERSION
            or assessment.expected_memory_ids != (f"{self.case_id.lower()}-target",)
        ):
            raise ValueError("paired observation condition must match the frozen probe")
        if (
            self.baseline_observation.usage.retrieval_calls != 1
            or self.baseline_observation.usage.candidate_count != 4
            or self.mutated_observation.usage.retrieval_calls != 1
            or self.mutated_observation.usage.candidate_count != 12
        ):
            raise ValueError("paired observation usage must match four and twelve candidates")
        recomputed_assessment = assess_paired_retrieval_challenge(
            self.baseline_observation,
            self.mutated_observation,
            policy=FROZEN_POLICY,
            case_id=self.case_id,
        )
        if self.paired_assessment != recomputed_assessment:
            raise ValueError("paired assessment must be recomputed from raw observations")
        return self


class RetrievalStrongFamilySummary(_DeterministicModel):
    """Paired outcome counts for one of the three strong families."""

    family: RetrievalStrongProbeChallengeFamily
    total_cases: StrictPositiveInt
    baseline_eligible_cases: StrictNonNegativeInt
    baseline_insufficient_cases: StrictNonNegativeInt
    induced_shadowing_cases: StrictNonNegativeInt
    resilient_cases: StrictNonNegativeInt

    @model_validator(mode="after")
    def counts_are_consistent(self) -> RetrievalStrongFamilySummary:
        if self.family not in STRONG_FAMILIES or self.total_cases != 8:
            raise ValueError("family summaries cover exactly eight strong cases")
        if self.baseline_eligible_cases + self.baseline_insufficient_cases != 8:
            raise ValueError("family baseline counts must partition eight cases")
        if self.induced_shadowing_cases + self.resilient_cases != self.baseline_eligible_cases:
            raise ValueError("family eligible outcomes must partition eligible cases")
        return self


class RetrievalStrongProbeSummary(_DeterministicModel):
    """All preregistered counts, gates, and the mechanical interpretation."""

    strong_total_cases: StrictPositiveInt
    strong_baseline_eligible_cases: StrictNonNegativeInt
    strong_baseline_insufficient_cases: StrictNonNegativeInt
    strong_induced_shadowing_cases: StrictNonNegativeInt
    strong_resilient_cases: StrictNonNegativeInt
    strong_induced_shadowing_rate: StrictUnitRatio | None
    family_summaries: tuple[RetrievalStrongFamilySummary, ...]
    families_with_induced_shadowing: StrictNonNegativeInt
    control_total_cases: StrictPositiveInt
    control_baseline_eligible_cases: StrictNonNegativeInt
    control_baseline_insufficient_cases: StrictNonNegativeInt
    control_induced_shadowing_cases: StrictNonNegativeInt
    control_resilient_cases: StrictNonNegativeInt
    baseline_eligibility_gate: RetrievalStrongProbeGateResult
    strong_shadowing_gate: RetrievalStrongProbeGateResult
    family_breadth_gate: RetrievalStrongProbeGateResult
    control_stability_gate: RetrievalStrongProbeGateResult
    interpretation: RetrievalStrongProbeInterpretation

    @field_validator("strong_induced_shadowing_rate", mode="before")
    @classmethod
    def rate_must_be_a_python_float(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("strong induced-shadowing rate must be a Python float or null")
        return value

    @model_validator(mode="after")
    def gates_and_interpretation_are_mechanical(self) -> RetrievalStrongProbeSummary:
        if self.strong_total_cases != 24 or self.control_total_cases != 6:
            raise ValueError("summary totals must cover 24 strong cases and six controls")
        if (
            self.strong_baseline_eligible_cases
            + self.strong_baseline_insufficient_cases
            != 24
        ):
            raise ValueError("strong baseline counts must partition 24 cases")
        if (
            self.strong_induced_shadowing_cases + self.strong_resilient_cases
            != self.strong_baseline_eligible_cases
        ):
            raise ValueError("strong eligible outcomes must partition eligible cases")
        expected_rate = (
            None
            if self.strong_baseline_eligible_cases == 0
            else self.strong_induced_shadowing_cases
            / self.strong_baseline_eligible_cases
        )
        if self.strong_induced_shadowing_rate != expected_rate:
            raise ValueError("strong induced-shadowing rate must match raw counts")
        if (
            self.control_baseline_eligible_cases
            + self.control_baseline_insufficient_cases
            != 6
        ):
            raise ValueError("control baseline counts must partition six cases")
        if (
            self.control_induced_shadowing_cases + self.control_resilient_cases
            != self.control_baseline_eligible_cases
        ):
            raise ValueError("control eligible outcomes must partition eligible cases")
        if tuple(summary.family for summary in self.family_summaries) != STRONG_FAMILIES:
            raise ValueError("family summaries must use the frozen family order")
        if sum(summary.baseline_eligible_cases for summary in self.family_summaries) != (
            self.strong_baseline_eligible_cases
        ):
            raise ValueError("family baseline counts must sum to the strong total")
        if sum(summary.induced_shadowing_cases for summary in self.family_summaries) != (
            self.strong_induced_shadowing_cases
        ):
            raise ValueError("family induced counts must sum to the strong total")
        expected_breadth = sum(
            summary.induced_shadowing_cases > 0 for summary in self.family_summaries
        )
        if self.families_with_induced_shadowing != expected_breadth:
            raise ValueError("family breadth must match family-level induced counts")

        baseline_pass = self.strong_baseline_eligible_cases >= BASELINE_ELIGIBILITY_MINIMUM
        strong_pass = (
            self.strong_induced_shadowing_cases >= STRONG_INDUCED_MINIMUM
            and self.strong_baseline_eligible_cases > 0
            and self.strong_induced_shadowing_cases * 5
            >= self.strong_baseline_eligible_cases * 2
        )
        breadth_pass = self.families_with_induced_shadowing >= FAMILY_BREADTH_MINIMUM
        control_pass = (
            self.control_baseline_eligible_cases
            >= CONTROL_BASELINE_ELIGIBILITY_MINIMUM
            and self.control_induced_shadowing_cases <= CONTROL_INDUCED_MAXIMUM
        )
        expected_gates = (
            RetrievalStrongProbeGateResult.BASELINE_ELIGIBILITY_GATE_PASS
            if baseline_pass
            else RetrievalStrongProbeGateResult.BASELINE_ELIGIBILITY_GATE_FAIL,
            RetrievalStrongProbeGateResult.STRONG_SHADOWING_GATE_PASS
            if strong_pass
            else RetrievalStrongProbeGateResult.STRONG_SHADOWING_GATE_FAIL,
            RetrievalStrongProbeGateResult.FAMILY_BREADTH_GATE_PASS
            if breadth_pass
            else RetrievalStrongProbeGateResult.FAMILY_BREADTH_GATE_FAIL,
            RetrievalStrongProbeGateResult.CONTROL_STABILITY_GATE_PASS
            if control_pass
            else RetrievalStrongProbeGateResult.CONTROL_STABILITY_GATE_FAIL,
        )
        actual_gates = (
            self.baseline_eligibility_gate,
            self.strong_shadowing_gate,
            self.family_breadth_gate,
            self.control_stability_gate,
        )
        if actual_gates != expected_gates:
            raise ValueError("gate results must match preregistered arithmetic")
        expected_interpretation = (
            RetrievalStrongProbeInterpretation.INCONCLUSIVE_BASELINE_CONSTRUCTION
            if not baseline_pass
            else RetrievalStrongProbeInterpretation.SUPPORTS_H4
            if strong_pass and breadth_pass and control_pass
            else RetrievalStrongProbeInterpretation.DOES_NOT_SUPPORT_H4
        )
        if self.interpretation is not expected_interpretation:
            raise ValueError("interpretation must follow the frozen three-way rule")
        return self


class RetrievalStrongProbeExecutionResult(_DeterministicModel):
    """Deterministic execution result without query or memory text."""

    schema_version: StrictStr
    probe_id: StrictStr
    fixture_sha256: StrictStr
    observations: tuple[RetrievalStrongProbeObservation, ...]
    summary: RetrievalStrongProbeSummary

    @model_validator(mode="after")
    def result_is_complete_and_recomputable(self) -> RetrievalStrongProbeExecutionResult:
        if (
            self.schema_version != RETRIEVAL_STRONG_PROBE_SCHEMA_VERSION
            or self.probe_id != RETRIEVAL_STRONG_PROBE_ID
            or self.fixture_sha256 != RETRIEVAL_STRONG_PROBE_FIXTURE_SHA256
        ):
            raise ValueError("execution result identity does not match the frozen probe")
        if tuple(observation.case_id for observation in self.observations) != EXPECTED_CASE_IDS:
            raise ValueError("execution result must contain the complete canonical case order")
        if self.summary != summarize_retrieval_strong_probe(self.observations):
            raise ValueError("execution summary must be recomputed from paired observations")
        return self


def _outcome_counts(
    observations: tuple[RetrievalStrongProbeObservation, ...],
) -> tuple[int, int, int]:
    induced = sum(
        observation.paired_assessment.outcome
        is RetrievalChallengeOutcome.INDUCED_SHADOWING
        for observation in observations
    )
    resilient = sum(
        observation.paired_assessment.outcome is RetrievalChallengeOutcome.RESILIENT
        for observation in observations
    )
    baseline_insufficient = sum(
        observation.paired_assessment.outcome
        is RetrievalChallengeOutcome.BASELINE_INSUFFICIENT
        for observation in observations
    )
    return induced, resilient, baseline_insufficient


def summarize_retrieval_strong_probe(
    observations: tuple[RetrievalStrongProbeObservation, ...],
) -> RetrievalStrongProbeSummary:
    """Apply the preregistered gates to complete paired observations."""

    if not isinstance(observations, tuple) or any(
        not isinstance(observation, RetrievalStrongProbeObservation)
        for observation in observations
    ):
        raise RetrievalStrongProbeInputError(
            "summary requires a tuple of retrieval-strong observations"
        )
    if tuple(observation.case_id for observation in observations) != EXPECTED_CASE_IDS:
        raise RetrievalStrongProbeInputError(
            "summary requires every frozen case in canonical order"
        )

    strong = tuple(
        observation
        for observation in observations
        if observation.case_kind is RetrievalStrongProbeCaseKind.STRONG_CHALLENGE
    )
    controls = tuple(
        observation
        for observation in observations
        if observation.case_kind is RetrievalStrongProbeCaseKind.RESILIENCE_CONTROL
    )
    strong_induced, strong_resilient, strong_insufficient = _outcome_counts(strong)
    control_induced, control_resilient, control_insufficient = _outcome_counts(controls)
    strong_eligible = strong_induced + strong_resilient
    control_eligible = control_induced + control_resilient

    family_summaries: list[RetrievalStrongFamilySummary] = []
    for family in STRONG_FAMILIES:
        family_observations = tuple(
            observation
            for observation in strong
            if observation.challenge_family is family
        )
        induced, resilient, insufficient = _outcome_counts(family_observations)
        family_summaries.append(
            RetrievalStrongFamilySummary(
                family=family,
                total_cases=len(family_observations),
                baseline_eligible_cases=induced + resilient,
                baseline_insufficient_cases=insufficient,
                induced_shadowing_cases=induced,
                resilient_cases=resilient,
            )
        )

    breadth = sum(summary.induced_shadowing_cases > 0 for summary in family_summaries)
    baseline_pass = strong_eligible >= BASELINE_ELIGIBILITY_MINIMUM
    strong_pass = (
        strong_induced >= STRONG_INDUCED_MINIMUM
        and strong_eligible > 0
        and strong_induced * 5 >= strong_eligible * 2
    )
    breadth_pass = breadth >= FAMILY_BREADTH_MINIMUM
    control_pass = (
        control_eligible >= CONTROL_BASELINE_ELIGIBILITY_MINIMUM
        and control_induced <= CONTROL_INDUCED_MAXIMUM
    )
    interpretation = (
        RetrievalStrongProbeInterpretation.INCONCLUSIVE_BASELINE_CONSTRUCTION
        if not baseline_pass
        else RetrievalStrongProbeInterpretation.SUPPORTS_H4
        if strong_pass and breadth_pass and control_pass
        else RetrievalStrongProbeInterpretation.DOES_NOT_SUPPORT_H4
    )
    return RetrievalStrongProbeSummary(
        strong_total_cases=len(strong),
        strong_baseline_eligible_cases=strong_eligible,
        strong_baseline_insufficient_cases=strong_insufficient,
        strong_induced_shadowing_cases=strong_induced,
        strong_resilient_cases=strong_resilient,
        strong_induced_shadowing_rate=(
            None if strong_eligible == 0 else strong_induced / strong_eligible
        ),
        family_summaries=tuple(family_summaries),
        families_with_induced_shadowing=breadth,
        control_total_cases=len(controls),
        control_baseline_eligible_cases=control_eligible,
        control_baseline_insufficient_cases=control_insufficient,
        control_induced_shadowing_cases=control_induced,
        control_resilient_cases=control_resilient,
        baseline_eligibility_gate=(
            RetrievalStrongProbeGateResult.BASELINE_ELIGIBILITY_GATE_PASS
            if baseline_pass
            else RetrievalStrongProbeGateResult.BASELINE_ELIGIBILITY_GATE_FAIL
        ),
        strong_shadowing_gate=(
            RetrievalStrongProbeGateResult.STRONG_SHADOWING_GATE_PASS
            if strong_pass
            else RetrievalStrongProbeGateResult.STRONG_SHADOWING_GATE_FAIL
        ),
        family_breadth_gate=(
            RetrievalStrongProbeGateResult.FAMILY_BREADTH_GATE_PASS
            if breadth_pass
            else RetrievalStrongProbeGateResult.FAMILY_BREADTH_GATE_FAIL
        ),
        control_stability_gate=(
            RetrievalStrongProbeGateResult.CONTROL_STABILITY_GATE_PASS
            if control_pass
            else RetrievalStrongProbeGateResult.CONTROL_STABILITY_GATE_FAIL
        ),
        interpretation=interpretation,
    )


def sha256_file(path: Path) -> str:
    """Return the byte-level SHA-256 for one frozen input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_retrieval_strong_probe(path: Path) -> RetrievalStrongProbeSpec:
    """Load and validate a probe without executing retrieval."""

    try:
        return RetrievalStrongProbeSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as error:
        raise RetrievalStrongProbeInputError(f"invalid retrieval probe fixture: {path}") from error


def preflight_retrieval_strong_probe(path: Path) -> RetrievalStrongProbeSpec:
    """Verify frozen bytes before parsing and before retriever construction."""

    try:
        actual_sha256 = sha256_file(path)
    except OSError as error:
        raise RetrievalStrongProbeInputError(
            f"cannot read retrieval probe fixture: {path}"
        ) from error
    if actual_sha256 != RETRIEVAL_STRONG_PROBE_FIXTURE_SHA256:
        raise RetrievalStrongProbeInputError(
            "retrieval strong probe fixture SHA mismatch: "
            f"expected {RETRIEVAL_STRONG_PROBE_FIXTURE_SHA256}, got {actual_sha256}"
        )
    return load_retrieval_strong_probe(path)
