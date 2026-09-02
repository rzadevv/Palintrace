"""Frozen development probe for speaker-identity premise grounding."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from memlint.semantics import (
    SemanticJudge,
    SemanticJudgment,
    SemanticRelation,
    semantic_judge_identity,
)

IDENTITY_PROBE_SCHEMA_VERSION = "0.1"
IDENTITY_PROBE_ID = "unsupported-identity-counterfactual-v0.1"
IDENTITY_PROBE_SPLIT = "development"
IDENTITY_PROBE_FIXTURE_PATH = Path(
    "tests/fixtures/unsupported_identity_probe_v0.1.json"
)
IDENTITY_PROBE_FIXTURE_SHA256 = (
    "4cbc1dc77b1d6a315992c2b564e438f77d6ff6df3f5dfa621d0542cdf7ea7beb"
)
IDENTITY_PROBE_MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
IDENTITY_PROBE_MODEL_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
IDENTITY_PROBE_DEVICE = "cpu"
PLAIN_PREMISE_TEMPLATE = "{source_text}"
SPEAKER_GROUNDED_PREMISE_TEMPLATE = "The speaker is {person_name}.\n{source_text}"

_FROZEN_PEOPLE = frozenset({"Mireya", "Tomasz", "Yuna"})
_FORBIDDEN_HELD_OUT_PEOPLE = frozenset({"Ava", "Kenji", "Lina"})
_REQUIRED_DOMAINS = frozenset(
    {
        "tool_or_software",
        "location",
        "schedule",
        "device_or_hardware",
        "project_or_employment",
        "preference_or_subscription",
    }
)
_EXPECTED_CASE_IDS = frozenset(
    {
        *(f"IG-C{index:02d}" for index in range(1, 7)),
        *(f"IG-S{index:02d}" for index in range(1, 19)),
    }
)
_FIRST_PERSON_PATTERN = re.compile(r"\b(?:i|me|my)\b", re.IGNORECASE)

StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]
StrictUnitScore = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]


class IdentityProbeInputError(ValueError):
    """The frozen probe input or result cannot be validated."""


class IdentityProbeCaseKind(StrEnum):
    """Whether resolving the source claim requires the supplied speaker identity."""

    IDENTITY_SENSITIVE = "identity_sensitive"
    IDENTITY_FREE_CONTROL = "identity_free_control"


class IdentityProbeTransformationKind(StrEnum):
    """Exact representation transformation applied by one frozen case."""

    FIRST_PERSON_SUBJECT = "first_person_subject"
    FIRST_PERSON_POSSESSIVE = "first_person_possessive"
    ALREADY_NAMED_SUBJECT = "already_named_subject"


class IdentityProbeConditionName(StrEnum):
    """The two frozen premise representations."""

    PLAIN = "plain"
    SPEAKER_GROUNDED = "speaker_grounded_v0.1"


class IdentityProbeHypothesisKind(StrEnum):
    """The two hypotheses evaluated for every case and condition."""

    CLEAN = "clean"
    UNSUPPORTED = "unsupported"


class IdentityProbeInterpretation(StrEnum):
    """The only preregistered overall interpretations."""

    SUPPORTS_H1 = "SUPPORTS_H1"
    DOES_NOT_SUPPORT_H1 = "DOES_NOT_SUPPORT_H1"
    INCONCLUSIVE_FAILURE_NOT_REPRODUCED = "INCONCLUSIVE_FAILURE_NOT_REPRODUCED"


class FailurePatternState(StrEnum):
    """Whether PLAIN reproduced the preregistered clean-selectivity problem."""

    REPRODUCED = "FAILURE_PATTERN_REPRODUCED"
    WEAK_OR_NOT_REPRODUCED = "FAILURE_PATTERN_WEAK_OR_NOT_REPRODUCED"


class CleanRescueGateState(StrEnum):
    """Frozen clean-rescue gate labels."""

    PASS = "CLEAN_RESCUE_GATE_PASS"
    FAIL = "CLEAN_RESCUE_GATE_FAIL"


class UnsupportedSafetyGateState(StrEnum):
    """Frozen unsupported-safety gate labels."""

    PASS = "UNSUPPORTED_SAFETY_GATE_PASS"
    FAIL = "UNSUPPORTED_SAFETY_GATE_FAIL"


class PrefixStabilityGateState(StrEnum):
    """Frozen identity-free prefix-stability gate labels."""

    PASS = "PREFIX_STABILITY_GATE_PASS"
    FAIL = "PREFIX_STABILITY_GATE_FAIL"


def _nonblank(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


class _DeterministicIdentityProbeModel(BaseModel):
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


class IdentityProbeCondition(_DeterministicIdentityProbeModel):
    """One exact premise rendering condition."""

    condition: IdentityProbeConditionName
    premise_template: str

    @field_validator("premise_template")
    @classmethod
    def premise_template_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="premise_template")

    @model_validator(mode="after")
    def condition_has_frozen_template(self) -> IdentityProbeCondition:
        expected = {
            IdentityProbeConditionName.PLAIN: PLAIN_PREMISE_TEMPLATE,
            IdentityProbeConditionName.SPEAKER_GROUNDED: (
                SPEAKER_GROUNDED_PREMISE_TEMPLATE
            ),
        }[self.condition]
        if self.premise_template != expected:
            raise ValueError("identity-probe premise template does not match its condition")
        return self


class IdentityProbeCase(_DeterministicIdentityProbeModel):
    """One frozen source with clean and one-value unsupported hypotheses."""

    case_id: str
    case_kind: IdentityProbeCaseKind
    person_name: str
    domain: str
    transformation_kind: IdentityProbeTransformationKind
    source_text: str
    clean_hypothesis: str
    unsupported_hypothesis: str
    changed_field: str
    source_value: str
    replacement_value: str

    @field_validator(
        "case_id",
        "person_name",
        "domain",
        "source_text",
        "clean_hypothesis",
        "unsupported_hypothesis",
        "changed_field",
        "source_value",
        "replacement_value",
    )
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="identity-probe case field")

    @model_validator(mode="after")
    def representation_and_substitution_are_structurally_valid(self) -> IdentityProbeCase:
        if self.person_name in _FORBIDDEN_HELD_OUT_PEOPLE:
            raise ValueError("held-out v0.1 person names are forbidden")
        text_fields = (
            self.source_text,
            self.clean_hypothesis,
            self.unsupported_hypothesis,
        )
        if any(name in text for name in _FORBIDDEN_HELD_OUT_PEOPLE for text in text_fields):
            raise ValueError("held-out v0.1 person names are forbidden in probe text")
        if self.source_value == self.replacement_value:
            raise ValueError("source and replacement values must differ")
        if self.source_text.count(self.source_value) != 1:
            raise ValueError("source_value must occur exactly once in source_text")
        if self.clean_hypothesis.count(self.source_value) != 1:
            raise ValueError("source_value must occur exactly once in clean_hypothesis")
        if self.replacement_value in self.source_text:
            raise ValueError("replacement_value must be absent from source_text")
        if self.replacement_value in self.clean_hypothesis:
            raise ValueError("replacement_value must be absent from clean_hypothesis")
        expected_unsupported = self.clean_hypothesis.replace(
            self.source_value,
            self.replacement_value,
            1,
        )
        if self.unsupported_hypothesis != expected_unsupported:
            raise ValueError(
                "unsupported_hypothesis must change exactly the declared factual value"
            )
        if self.clean_hypothesis.count(self.person_name) != 1:
            raise ValueError("clean_hypothesis must contain person_name exactly once")
        if self.unsupported_hypothesis.count(self.person_name) != 1:
            raise ValueError("unsupported_hypothesis must contain person_name exactly once")

        if self.case_kind is IdentityProbeCaseKind.IDENTITY_FREE_CONTROL:
            if (
                self.transformation_kind
                is not IdentityProbeTransformationKind.ALREADY_NAMED_SUBJECT
            ):
                raise ValueError("identity-free controls require already_named_subject")
            if not self.case_id.startswith("IG-C"):
                raise ValueError("identity-free control IDs must start with IG-C")
            if self.source_text != self.clean_hypothesis:
                raise ValueError("identity-free clean hypothesis must equal its named source")
            if _FIRST_PERSON_PATTERN.search(self.source_text):
                raise ValueError("identity-free source_text must contain no first-person pronoun")
            named_subject = self.source_text.startswith(f"{self.person_name} ")
            named_possessive = self.source_text.startswith(f"{self.person_name}'s ")
            if not named_subject and not named_possessive:
                raise ValueError("identity-free source_text must explicitly name the person")
        else:
            if not self.case_id.startswith("IG-S"):
                raise ValueError("identity-sensitive IDs must start with IG-S")
            if self.person_name in self.source_text:
                raise ValueError("identity-sensitive source_text must not already name the person")
            if self.clean_hypothesis == self.source_text:
                raise ValueError("identity-sensitive clean hypothesis must normalize first person")
            if self.transformation_kind is IdentityProbeTransformationKind.FIRST_PERSON_SUBJECT:
                if not self.source_text.startswith("I "):
                    raise ValueError("first_person_subject source_text must begin with 'I '")
                if not self.clean_hypothesis.startswith(f"{self.person_name} "):
                    raise ValueError("first_person_subject clean hypothesis must name the subject")
            elif (
                self.transformation_kind
                is IdentityProbeTransformationKind.FIRST_PERSON_POSSESSIVE
            ):
                if not self.source_text.startswith("My "):
                    raise ValueError("first_person_possessive source_text must begin with 'My '")
                if not self.clean_hypothesis.startswith(f"{self.person_name}'s "):
                    raise ValueError(
                        "first_person_possessive clean hypothesis must use named possession"
                    )
            else:
                raise ValueError("identity-sensitive cases require a first-person transformation")
        return self


class IdentityProbeSpec(_DeterministicIdentityProbeModel):
    """Complete frozen development-only question set, without semantic answers."""

    schema_version: str
    probe_id: str
    split: str
    conditions: tuple[IdentityProbeCondition, ...]
    cases: tuple[IdentityProbeCase, ...]

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_frozen(cls, value: str) -> str:
        if value != IDENTITY_PROBE_SCHEMA_VERSION:
            raise ValueError("identity probe schema_version must be '0.1'")
        return value

    @field_validator("probe_id")
    @classmethod
    def probe_id_is_frozen(cls, value: str) -> str:
        if value != IDENTITY_PROBE_ID:
            raise ValueError("identity probe ID does not match the v0.1 freeze")
        return value

    @field_validator("split")
    @classmethod
    def split_is_development_only(cls, value: str) -> str:
        if value != IDENTITY_PROBE_SPLIT:
            raise ValueError("identity probe split must be development")
        return value

    @field_validator("conditions")
    @classmethod
    def conditions_are_canonical(
        cls, value: tuple[IdentityProbeCondition, ...]
    ) -> tuple[IdentityProbeCondition, ...]:
        return tuple(sorted(value, key=lambda item: item.condition.value))

    @field_validator("cases")
    @classmethod
    def cases_are_canonical(
        cls, value: tuple[IdentityProbeCase, ...]
    ) -> tuple[IdentityProbeCase, ...]:
        return tuple(sorted(value, key=lambda item: item.case_id))

    @model_validator(mode="after")
    def frozen_matrix_is_complete_and_balanced(self) -> IdentityProbeSpec:
        if tuple(item.condition for item in self.conditions) != tuple(IdentityProbeConditionName):
            raise ValueError("identity probe requires exactly the two frozen conditions")
        if len(self.cases) != 24:
            raise ValueError("identity probe requires exactly 24 scenarios")
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids) or set(case_ids) != _EXPECTED_CASE_IDS:
            raise ValueError("identity probe case IDs must match the frozen 24-case matrix")
        kind_counts = Counter(case.case_kind for case in self.cases)
        if kind_counts != Counter(
            {
                IdentityProbeCaseKind.IDENTITY_SENSITIVE: 18,
                IdentityProbeCaseKind.IDENTITY_FREE_CONTROL: 6,
            }
        ):
            raise ValueError("identity probe requires 18 sensitive cases and 6 controls")
        transformation_counts = Counter(case.transformation_kind for case in self.cases)
        if transformation_counts != Counter(
            {
                IdentityProbeTransformationKind.FIRST_PERSON_SUBJECT: 12,
                IdentityProbeTransformationKind.FIRST_PERSON_POSSESSIVE: 6,
                IdentityProbeTransformationKind.ALREADY_NAMED_SUBJECT: 6,
            }
        ):
            raise ValueError("identity probe transformation counts do not match the freeze")
        people = {case.person_name for case in self.cases}
        if people != _FROZEN_PEOPLE:
            raise ValueError("identity probe requires exactly the three frozen new people")
        sensitive = tuple(
            case
            for case in self.cases
            if case.case_kind is IdentityProbeCaseKind.IDENTITY_SENSITIVE
        )
        controls = tuple(
            case
            for case in self.cases
            if case.case_kind is IdentityProbeCaseKind.IDENTITY_FREE_CONTROL
        )
        if any(sum(case.person_name == person for case in sensitive) != 6 for person in people):
            raise ValueError("each person requires exactly six identity-sensitive cases")
        if any(sum(case.person_name == person for case in controls) != 2 for person in people):
            raise ValueError("each person requires exactly two identity-free controls")
        domain_counts = Counter(case.domain for case in sensitive)
        if set(domain_counts) != _REQUIRED_DOMAINS or any(
            domain_counts[domain] != 3 for domain in _REQUIRED_DOMAINS
        ):
            raise ValueError("each required domain needs three identity-sensitive cases")
        return self


class IdentityProbeJudgment(_DeterministicIdentityProbeModel):
    """One pinned-judge answer without rationale or chain of thought."""

    case_id: str
    case_kind: IdentityProbeCaseKind
    condition: IdentityProbeConditionName
    hypothesis_kind: IdentityProbeHypothesisKind
    relation: SemanticRelation
    score: StrictUnitScore
    judge_id: str
    judge_version: str
    model_calls: StrictNonNegativeInt
    input_tokens: StrictNonNegativeInt
    output_tokens: StrictNonNegativeInt

    @field_validator("case_id", "judge_id", "judge_version")
    @classmethod
    def identity_strings_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="identity-probe judgment identity")

    @field_validator("score", mode="before")
    @classmethod
    def score_must_be_a_python_float(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("identity-probe score must be a Python float")
        return value


class IdentityProbeGroupSummary(_DeterministicIdentityProbeModel):
    """Paired outcome counts for one sensitive/control case group."""

    scenario_count: StrictPositiveInt
    plain_clean_entailments: StrictNonNegativeInt
    grounded_clean_entailments: StrictNonNegativeInt
    plain_clean_non_entailments: StrictNonNegativeInt
    grounded_clean_non_entailments: StrictNonNegativeInt
    clean_non_entailment_to_entailment: StrictNonNegativeInt
    clean_entailment_to_non_entailment: StrictNonNegativeInt
    clean_unchanged_entailment: StrictNonNegativeInt
    clean_unchanged_non_entailment: StrictNonNegativeInt
    clean_exact_relation_changes: StrictNonNegativeInt
    plain_unsupported_detected: StrictNonNegativeInt
    grounded_unsupported_detected: StrictNonNegativeInt
    plain_unsupported_missed: StrictNonNegativeInt
    grounded_unsupported_missed: StrictNonNegativeInt
    unsupported_detected_to_missed: StrictNonNegativeInt
    unsupported_missed_to_detected: StrictNonNegativeInt
    unsupported_unchanged_detected: StrictNonNegativeInt
    unsupported_unchanged_missed: StrictNonNegativeInt

    @model_validator(mode="after")
    def partitions_are_consistent(self) -> IdentityProbeGroupSummary:
        count = self.scenario_count
        if self.plain_clean_entailments + self.plain_clean_non_entailments != count:
            raise ValueError("plain clean counts must partition the group")
        if self.grounded_clean_entailments + self.grounded_clean_non_entailments != count:
            raise ValueError("grounded clean counts must partition the group")
        if self.plain_unsupported_detected + self.plain_unsupported_missed != count:
            raise ValueError("plain unsupported counts must partition the group")
        if self.grounded_unsupported_detected + self.grounded_unsupported_missed != count:
            raise ValueError("grounded unsupported counts must partition the group")
        clean_transitions = (
            self.clean_non_entailment_to_entailment
            + self.clean_entailment_to_non_entailment
            + self.clean_unchanged_entailment
            + self.clean_unchanged_non_entailment
        )
        if clean_transitions != count:
            raise ValueError("clean transitions must partition the group")
        unsupported_transitions = (
            self.unsupported_detected_to_missed
            + self.unsupported_missed_to_detected
            + self.unsupported_unchanged_detected
            + self.unsupported_unchanged_missed
        )
        if unsupported_transitions != count:
            raise ValueError("unsupported transitions must partition the group")
        if self.plain_clean_entailments != (
            self.clean_entailment_to_non_entailment + self.clean_unchanged_entailment
        ):
            raise ValueError("plain clean entailments do not match transitions")
        if self.grounded_clean_entailments != (
            self.clean_non_entailment_to_entailment + self.clean_unchanged_entailment
        ):
            raise ValueError("grounded clean entailments do not match transitions")
        if self.plain_unsupported_detected != (
            self.unsupported_detected_to_missed + self.unsupported_unchanged_detected
        ):
            raise ValueError("plain unsupported detections do not match transitions")
        if self.grounded_unsupported_detected != (
            self.unsupported_missed_to_detected + self.unsupported_unchanged_detected
        ):
            raise ValueError("grounded unsupported detections do not match transitions")
        if self.clean_exact_relation_changes > count:
            raise ValueError("exact clean relation changes cannot exceed scenario_count")
        return self


class IdentityProbeSummary(_DeterministicIdentityProbeModel):
    """Frozen primary outcomes reported separately by case kind."""

    identity_sensitive: IdentityProbeGroupSummary
    identity_free_control: IdentityProbeGroupSummary


class IdentityProbeGateEvaluation(_DeterministicIdentityProbeModel):
    """All four preregistered gate results."""

    failure_pattern: FailurePatternState
    clean_rescue: CleanRescueGateState
    unsupported_safety: UnsupportedSafetyGateState
    prefix_stability: PrefixStabilityGateState


class IdentityProbeExecutionResult(_DeterministicIdentityProbeModel):
    """Complete execution artifact for exactly 96 frozen judgments."""

    schema_version: str = IDENTITY_PROBE_SCHEMA_VERSION
    probe_id: str
    fixture_sha256: str
    judge_id: str
    judge_version: str
    judgments: tuple[IdentityProbeJudgment, ...]
    summary: IdentityProbeSummary
    gates: IdentityProbeGateEvaluation
    interpretation: IdentityProbeInterpretation

    @field_validator("schema_version")
    @classmethod
    def result_schema_is_frozen(cls, value: str) -> str:
        if value != IDENTITY_PROBE_SCHEMA_VERSION:
            raise ValueError("identity-probe result schema_version must be '0.1'")
        return value

    @field_validator("probe_id")
    @classmethod
    def result_probe_id_is_frozen(cls, value: str) -> str:
        if value != IDENTITY_PROBE_ID:
            raise ValueError("identity-probe result probe_id does not match the freeze")
        return value

    @field_validator("fixture_sha256")
    @classmethod
    def result_fixture_hash_is_frozen(cls, value: str) -> str:
        if value != IDENTITY_PROBE_FIXTURE_SHA256:
            raise ValueError("identity-probe result fixture SHA does not match the freeze")
        return value

    @field_validator("judge_id", "judge_version")
    @classmethod
    def result_judge_identity_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="identity-probe result judge identity")

    @field_validator("judgments")
    @classmethod
    def judgments_are_canonical(
        cls, value: tuple[IdentityProbeJudgment, ...]
    ) -> tuple[IdentityProbeJudgment, ...]:
        return tuple(
            sorted(
                value,
                key=lambda item: (
                    item.case_id,
                    item.condition.value,
                    item.hypothesis_kind.value,
                ),
            )
        )

    @model_validator(mode="after")
    def complete_result_matches_frozen_matrix_and_arithmetic(
        self,
    ) -> IdentityProbeExecutionResult:
        expected_judge_id = f"hf-nli:{IDENTITY_PROBE_MODEL_ID}"
        if self.judge_id != expected_judge_id:
            raise ValueError("identity-probe result judge_id does not match pinned MiniLM")
        if self.judge_version != IDENTITY_PROBE_MODEL_REVISION:
            raise ValueError("identity-probe result judge_version does not match pinned revision")
        if len(self.judgments) != 96:
            raise ValueError("complete identity-probe result requires exactly 96 judgments")
        keys = [
            (item.case_id, item.condition, item.hypothesis_kind)
            for item in self.judgments
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("identity-probe judgment keys must be unique")
        if {item.case_id for item in self.judgments} != _EXPECTED_CASE_IDS:
            raise ValueError("identity-probe result case IDs do not match the frozen fixture")
        for case_id in _EXPECTED_CASE_IDS:
            case_rows = [item for item in self.judgments if item.case_id == case_id]
            expected_kind = (
                IdentityProbeCaseKind.IDENTITY_FREE_CONTROL
                if case_id.startswith("IG-C")
                else IdentityProbeCaseKind.IDENTITY_SENSITIVE
            )
            wrong_kind = any(item.case_kind is not expected_kind for item in case_rows)
            if len(case_rows) != 4 or wrong_kind:
                raise ValueError("each frozen case requires its exact four judgments")
            if {
                (item.condition, item.hypothesis_kind) for item in case_rows
            } != {
                (condition, hypothesis)
                for condition in IdentityProbeConditionName
                for hypothesis in IdentityProbeHypothesisKind
            }:
                raise ValueError("each case requires both hypotheses under both conditions")
        if any(
            (item.judge_id, item.judge_version)
            != (self.judge_id, self.judge_version)
            for item in self.judgments
        ):
            raise ValueError("every judgment must match the result judge identity")
        recomputed_summary = summarize_identity_probe_judgments(self.judgments)
        if self.summary != recomputed_summary:
            raise ValueError("identity-probe summary does not match individual judgments")
        recomputed_gates = evaluate_identity_probe_gates(self.summary)
        if self.gates != recomputed_gates:
            raise ValueError("identity-probe gates do not match the frozen arithmetic")
        recomputed_interpretation = interpret_identity_probe_gates(self.gates)
        if self.interpretation is not recomputed_interpretation:
            raise ValueError("identity-probe interpretation does not match frozen logic")
        return self


def validate_identity_probe_model_identity() -> None:
    """Fail closed if execution constants no longer match the freeze."""

    if IDENTITY_PROBE_MODEL_ID != "cross-encoder/nli-MiniLM2-L6-H768":
        raise IdentityProbeInputError("identity-probe model ID does not match the freeze")
    if (
        IDENTITY_PROBE_MODEL_REVISION
        != "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
    ):
        raise IdentityProbeInputError("identity-probe model revision does not match the freeze")
    if IDENTITY_PROBE_DEVICE != "cpu":
        raise IdentityProbeInputError("identity-probe device must be cpu")


def preflight_identity_probe(
    path: Path = IDENTITY_PROBE_FIXTURE_PATH,
) -> IdentityProbeSpec:
    """Verify frozen fixture bytes and schema without constructing a semantic model."""

    if not isinstance(path, Path):
        raise IdentityProbeInputError("identity-probe path must be a pathlib.Path")
    try:
        fixture_bytes = path.read_bytes()
    except OSError as error:
        raise IdentityProbeInputError("could not read frozen identity-probe fixture") from error
    actual_sha = hashlib.sha256(fixture_bytes).hexdigest()
    if actual_sha != IDENTITY_PROBE_FIXTURE_SHA256:
        raise IdentityProbeInputError("frozen identity-probe fixture SHA mismatch")
    try:
        spec = IdentityProbeSpec.model_validate_json(fixture_bytes)
    except ValidationError as error:
        raise IdentityProbeInputError("invalid frozen identity-probe fixture") from error
    validate_identity_probe_model_identity()
    return spec


def build_identity_probe_premise(
    case: IdentityProbeCase,
    condition: IdentityProbeConditionName,
) -> str:
    """Construct exactly one of the two preregistered premises."""

    if not isinstance(case, IdentityProbeCase):
        raise IdentityProbeInputError("case must be an IdentityProbeCase")
    if not isinstance(condition, IdentityProbeConditionName):
        raise IdentityProbeInputError("condition must be an IdentityProbeConditionName")
    if condition is IdentityProbeConditionName.PLAIN:
        return case.source_text
    return f"The speaker is {case.person_name}.\n{case.source_text}"


def identity_probe_hypothesis(
    case: IdentityProbeCase,
    hypothesis_kind: IdentityProbeHypothesisKind,
) -> str:
    """Return the exact clean or unsupported frozen hypothesis."""

    if not isinstance(case, IdentityProbeCase):
        raise IdentityProbeInputError("case must be an IdentityProbeCase")
    if not isinstance(hypothesis_kind, IdentityProbeHypothesisKind):
        raise IdentityProbeInputError(
            "hypothesis_kind must be an IdentityProbeHypothesisKind"
        )
    if hypothesis_kind is IdentityProbeHypothesisKind.CLEAN:
        return case.clean_hypothesis
    return case.unsupported_hypothesis


def _is_entailment(relation: SemanticRelation) -> bool:
    return relation is SemanticRelation.ENTAILMENT


def _summarize_group(
    judgments: tuple[IdentityProbeJudgment, ...],
    case_kind: IdentityProbeCaseKind,
) -> IdentityProbeGroupSummary:
    rows = tuple(item for item in judgments if item.case_kind is case_kind)
    case_ids = sorted({item.case_id for item in rows})
    by_key = {
        (item.case_id, item.condition, item.hypothesis_kind): item for item in rows
    }

    plain_clean_entailments = 0
    grounded_clean_entailments = 0
    clean_non_to_entailment = 0
    clean_entailment_to_non = 0
    clean_unchanged_entailment = 0
    clean_unchanged_non = 0
    clean_exact_relation_changes = 0
    plain_unsupported_detected = 0
    grounded_unsupported_detected = 0
    unsupported_detected_to_missed = 0
    unsupported_missed_to_detected = 0
    unsupported_unchanged_detected = 0
    unsupported_unchanged_missed = 0

    for case_id in case_ids:
        plain_clean = by_key[
            (
                case_id,
                IdentityProbeConditionName.PLAIN,
                IdentityProbeHypothesisKind.CLEAN,
            )
        ].relation
        grounded_clean = by_key[
            (
                case_id,
                IdentityProbeConditionName.SPEAKER_GROUNDED,
                IdentityProbeHypothesisKind.CLEAN,
            )
        ].relation
        plain_clean_is_entailment = _is_entailment(plain_clean)
        grounded_clean_is_entailment = _is_entailment(grounded_clean)
        plain_clean_entailments += plain_clean_is_entailment
        grounded_clean_entailments += grounded_clean_is_entailment
        clean_exact_relation_changes += plain_clean is not grounded_clean
        if not plain_clean_is_entailment and grounded_clean_is_entailment:
            clean_non_to_entailment += 1
        elif plain_clean_is_entailment and not grounded_clean_is_entailment:
            clean_entailment_to_non += 1
        elif plain_clean_is_entailment:
            clean_unchanged_entailment += 1
        else:
            clean_unchanged_non += 1

        plain_unsupported = by_key[
            (
                case_id,
                IdentityProbeConditionName.PLAIN,
                IdentityProbeHypothesisKind.UNSUPPORTED,
            )
        ].relation
        grounded_unsupported = by_key[
            (
                case_id,
                IdentityProbeConditionName.SPEAKER_GROUNDED,
                IdentityProbeHypothesisKind.UNSUPPORTED,
            )
        ].relation
        plain_detected = not _is_entailment(plain_unsupported)
        grounded_detected = not _is_entailment(grounded_unsupported)
        plain_unsupported_detected += plain_detected
        grounded_unsupported_detected += grounded_detected
        if plain_detected and not grounded_detected:
            unsupported_detected_to_missed += 1
        elif not plain_detected and grounded_detected:
            unsupported_missed_to_detected += 1
        elif plain_detected:
            unsupported_unchanged_detected += 1
        else:
            unsupported_unchanged_missed += 1

    scenario_count = len(case_ids)
    return IdentityProbeGroupSummary(
        scenario_count=scenario_count,
        plain_clean_entailments=plain_clean_entailments,
        grounded_clean_entailments=grounded_clean_entailments,
        plain_clean_non_entailments=scenario_count - plain_clean_entailments,
        grounded_clean_non_entailments=scenario_count - grounded_clean_entailments,
        clean_non_entailment_to_entailment=clean_non_to_entailment,
        clean_entailment_to_non_entailment=clean_entailment_to_non,
        clean_unchanged_entailment=clean_unchanged_entailment,
        clean_unchanged_non_entailment=clean_unchanged_non,
        clean_exact_relation_changes=clean_exact_relation_changes,
        plain_unsupported_detected=plain_unsupported_detected,
        grounded_unsupported_detected=grounded_unsupported_detected,
        plain_unsupported_missed=scenario_count - plain_unsupported_detected,
        grounded_unsupported_missed=scenario_count - grounded_unsupported_detected,
        unsupported_detected_to_missed=unsupported_detected_to_missed,
        unsupported_missed_to_detected=unsupported_missed_to_detected,
        unsupported_unchanged_detected=unsupported_unchanged_detected,
        unsupported_unchanged_missed=unsupported_unchanged_missed,
    )


def summarize_identity_probe_judgments(
    judgments: tuple[IdentityProbeJudgment, ...],
) -> IdentityProbeSummary:
    """Recompute all frozen paired primary outcomes from individual judgments."""

    if not isinstance(judgments, tuple) or len(judgments) != 96:
        raise IdentityProbeInputError("identity-probe summary requires exactly 96 judgments")
    if any(not isinstance(item, IdentityProbeJudgment) for item in judgments):
        raise IdentityProbeInputError(
            "identity-probe summary accepts only IdentityProbeJudgment values"
        )
    keys = [
        (item.case_id, item.condition, item.hypothesis_kind) for item in judgments
    ]
    if len(set(keys)) != len(keys):
        raise IdentityProbeInputError("identity-probe judgment keys must be unique")
    return IdentityProbeSummary(
        identity_sensitive=_summarize_group(
            judgments,
            IdentityProbeCaseKind.IDENTITY_SENSITIVE,
        ),
        identity_free_control=_summarize_group(
            judgments,
            IdentityProbeCaseKind.IDENTITY_FREE_CONTROL,
        ),
    )


def evaluate_identity_probe_gates(
    summary: IdentityProbeSummary,
) -> IdentityProbeGateEvaluation:
    """Apply only the four preregistered development-probe gates."""

    if not isinstance(summary, IdentityProbeSummary):
        raise IdentityProbeInputError("summary must be an IdentityProbeSummary")
    sensitive = summary.identity_sensitive
    controls = summary.identity_free_control
    if sensitive.scenario_count != 18 or controls.scenario_count != 6:
        raise IdentityProbeInputError("identity-probe gate groups must contain 18 and 6 cases")

    failure_pattern = (
        FailurePatternState.REPRODUCED
        if sensitive.plain_clean_entailments <= 12
        else FailurePatternState.WEAK_OR_NOT_REPRODUCED
    )
    clean_rescue_passes = (
        sensitive.grounded_clean_entailments >= 16
        and sensitive.grounded_clean_entailments
        - sensitive.plain_clean_entailments
        >= 4
    )
    unsupported_safety_passes = (
        sensitive.grounded_unsupported_detected >= 17
        and sensitive.grounded_unsupported_detected
        >= sensitive.plain_unsupported_detected - 1
    )
    prefix_stability_passes = (
        controls.clean_exact_relation_changes <= 1
        and controls.unsupported_detected_to_missed <= 1
    )
    return IdentityProbeGateEvaluation(
        failure_pattern=failure_pattern,
        clean_rescue=(
            CleanRescueGateState.PASS
            if clean_rescue_passes
            else CleanRescueGateState.FAIL
        ),
        unsupported_safety=(
            UnsupportedSafetyGateState.PASS
            if unsupported_safety_passes
            else UnsupportedSafetyGateState.FAIL
        ),
        prefix_stability=(
            PrefixStabilityGateState.PASS
            if prefix_stability_passes
            else PrefixStabilityGateState.FAIL
        ),
    )


def interpret_identity_probe_gates(
    gates: IdentityProbeGateEvaluation,
) -> IdentityProbeInterpretation:
    """Return exactly one of the three frozen interpretation states."""

    if not isinstance(gates, IdentityProbeGateEvaluation):
        raise IdentityProbeInputError("gates must be an IdentityProbeGateEvaluation")
    if gates.failure_pattern is FailurePatternState.WEAK_OR_NOT_REPRODUCED:
        return IdentityProbeInterpretation.INCONCLUSIVE_FAILURE_NOT_REPRODUCED
    if (
        gates.clean_rescue is CleanRescueGateState.PASS
        and gates.unsupported_safety is UnsupportedSafetyGateState.PASS
        and gates.prefix_stability is PrefixStabilityGateState.PASS
    ):
        return IdentityProbeInterpretation.SUPPORTS_H1
    return IdentityProbeInterpretation.DOES_NOT_SUPPORT_H1


def execute_identity_probe(
    *,
    spec: IdentityProbeSpec,
    semantic_judge: SemanticJudge,
) -> IdentityProbeExecutionResult:
    """Execute the already-preflighted questions with an injected pinned-identity judge."""

    if not isinstance(spec, IdentityProbeSpec):
        raise IdentityProbeInputError("spec must be an IdentityProbeSpec")
    validate_identity_probe_model_identity()
    judge_id, judge_version = semantic_judge_identity(semantic_judge)
    if judge_id != f"hf-nli:{IDENTITY_PROBE_MODEL_ID}":
        raise IdentityProbeInputError("semantic judge ID does not match pinned MiniLM")
    if judge_version != IDENTITY_PROBE_MODEL_REVISION:
        raise IdentityProbeInputError("semantic judge version does not match pinned revision")

    judgments: list[IdentityProbeJudgment] = []
    for case in spec.cases:
        for condition in IdentityProbeConditionName:
            premise = build_identity_probe_premise(case, condition)
            for hypothesis_kind in IdentityProbeHypothesisKind:
                response = semantic_judge.judge(
                    premise=premise,
                    hypothesis=identity_probe_hypothesis(case, hypothesis_kind),
                )
                if not isinstance(response, SemanticJudgment):
                    raise IdentityProbeInputError("semantic judge returned an invalid judgment")
                judgments.append(
                    IdentityProbeJudgment(
                        case_id=case.case_id,
                        case_kind=case.case_kind,
                        condition=condition,
                        hypothesis_kind=hypothesis_kind,
                        relation=response.relation,
                        score=response.score,
                        judge_id=judge_id,
                        judge_version=judge_version,
                        model_calls=response.usage.model_calls,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    )
                )
    canonical_judgments = tuple(judgments)
    summary = summarize_identity_probe_judgments(canonical_judgments)
    gates = evaluate_identity_probe_gates(summary)
    interpretation = interpret_identity_probe_gates(gates)
    return IdentityProbeExecutionResult(
        probe_id=IDENTITY_PROBE_ID,
        fixture_sha256=IDENTITY_PROBE_FIXTURE_SHA256,
        judge_id=judge_id,
        judge_version=judge_version,
        judgments=canonical_judgments,
        summary=summary,
        gates=gates,
        interpretation=interpretation,
    )
