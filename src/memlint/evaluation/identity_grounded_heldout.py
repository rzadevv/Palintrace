"""Preregistered held-out evaluation for the frozen identity-grounded candidate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from memlint.checkers import UnsupportedClaimChecker
from memlint.checkers.models import CheckerResult
from memlint.checkers.unsupported_claim_identity_grounded import (
    IdentityGroundedUnsupportedClaimChecker,
)
from memlint.models import (
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    Transcript,
    TranscriptSet,
    TranscriptTurn,
)
from memlint.semantics import SemanticJudge, SemanticJudgment, SemanticRelation
from memlint.semantics.identity import (
    SpeakerIdentityBinding,
    SpeakerIdentityBindings,
    SpeakerIdentityResolutionStatus,
    resolve_speaker_identity,
)

HELDOUT_SCHEMA_VERSION = "0.1"
HELDOUT_EVALUATION_ID = "unsupported-identity-grounded-heldout-v0.1"
HELDOUT_SPLIT = "held_out"
HELDOUT_FIXTURE_PATH = Path(
    "tests/fixtures/unsupported_identity_grounded_heldout_v0.1.json"
)
HELDOUT_FREEZE_MANIFEST_PATH = Path(
    "tests/fixtures/unsupported_identity_grounded_heldout_v0.1.sha256.json"
)
HELDOUT_FIXTURE_SHA256 = (
    "a0384e2d4e5d7764c45c87e1c729762cbd2714ced2faa3cb7e36a2b50283169b"
)
HELDOUT_MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
HELDOUT_MODEL_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
HELDOUT_DEVICE = "cpu"

FROZEN_UNSUPPORTED_CLAIM_SHA256 = (
    "04fd713308d9ed55e79501a31e99904939a2caf8ef90f2187e3fe1f594d09a8a"
)
FROZEN_IDENTITY_CONTRACT_SHA256 = (
    "c6b54d0229cb6b87b5e23997685e9855b8b789ea2d68f6e6f07ee45a749f82f9"
)
FROZEN_LOCAL_NLI_SHA256 = (
    "aafe1e1a9d662879640285784704cdbfecefec4c25e402fae07101dd7ea087b1"
)
FROZEN_COMPOSITION_SHA256 = (
    "cd617221c65bb6a58de7164f7438143d661903f62f57076f4869d3e28d6a7629"
)
FROZEN_CANDIDATE_SHA256 = (
    "6b742eeff6d4280661adba61ed201b67a6bc25a7d9a2b0c967ebbccd0c3210c5"
)
FROZEN_BENCHMARK_SHA256 = (
    "fd11b0d547197495d51684f005ac17c861392891e464d818815e04eb6f37dad0"
)
FROZEN_BENCHMARK_MANIFEST_SHA256 = (
    "de4bb8c2076a2c89b7e2df95518ef5588934644b711119fccc8727e0e9ac73fb"
)
FROZEN_IDENTITY_PROBE_FIXTURE_SHA256 = (
    "4cbc1dc77b1d6a315992c2b564e438f77d6ff6df3f5dfa621d0542cdf7ea7beb"
)
FROZEN_IDENTITY_PROBE_RESULT_SHA256 = (
    "a205fd355291d42aab0dc267241d5b5ea03613f1d324e64ca6cf4ec8a9320219"
)

_REQUIRED_DOMAINS = frozenset(
    {
        "tool_or_software",
        "device_or_hardware",
        "workplace_or_project",
        "location",
        "schedule",
        "subscription_or_preference",
        "education_or_course",
        "travel",
        "ordinary_possession",
        "biography",
    }
)
_FIRST_PERSON_PATTERN = re.compile(r"\b(?:i|me|my)\b", re.IGNORECASE)
_EXPECTED_SEMANTIC_CASE_IDS = frozenset(
    {
        *(f"H6GC-C{index:02d}" for index in range(1, 11)),
        *(f"H6GC-S{index:02d}" for index in range(1, 31)),
    }
)

StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]
StrictUnitScore = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]
StrictUnitRate = StrictUnitScore


class HeldoutInputError(ValueError):
    """The preregistered held-out input or result failed closed validation."""


class SemanticCaseKind(StrEnum):
    """Whether the clean entailment depends on resolving first-person identity."""

    IDENTITY_SENSITIVE = "identity_sensitive"
    IDENTITY_FREE_CONTROL = "identity_free_control"


class TransformationKind(StrEnum):
    """Frozen source-to-claim transformation category."""

    FIRST_PERSON_SUBJECT = "first_person_subject"
    FIRST_PERSON_POSSESSIVE = "first_person_possessive"
    ALREADY_NAMED_SUBJECT = "already_named_subject"


class ConditionName(StrEnum):
    """Frozen checker conditions in the paired evaluation."""

    BASELINE_PLAIN = "baseline_plain"
    IDENTITY_GROUNDED = "identity_grounded"


class HypothesisKind(StrEnum):
    """Clean and exactly one-value unsupported hypotheses."""

    CLEAN = "clean"
    UNSUPPORTED = "unsupported"


class CoverageKind(StrEnum):
    """Frozen capability-contract cases."""

    SINGLE_TURN_RESOLVED = "single_turn_resolved"
    MULTI_TURN_SAME_SPEAKER_RESOLVED = "multi_turn_same_speaker_resolved"
    MISSING_BINDING_UNAVAILABLE = "missing_binding_unavailable"
    TRANSCRIPT_LEVEL_UNAVAILABLE = "transcript_level_unavailable"
    INCOMPLETE_MULTI_TURN_UNAVAILABLE = "incomplete_multi_turn_unavailable"
    MIXED_SPEAKERS_CONFLICT = "mixed_speakers_conflict"


_EXPECTED_COVERAGE_MATRIX = {
    "H6GC-R01": (
        CoverageKind.SINGLE_TURN_RESOLVED,
        SpeakerIdentityResolutionStatus.RESOLVED,
    ),
    "H6GC-R02": (
        CoverageKind.MULTI_TURN_SAME_SPEAKER_RESOLVED,
        SpeakerIdentityResolutionStatus.RESOLVED,
    ),
    "H6GC-U01": (
        CoverageKind.MISSING_BINDING_UNAVAILABLE,
        SpeakerIdentityResolutionStatus.UNAVAILABLE,
    ),
    "H6GC-U02": (
        CoverageKind.TRANSCRIPT_LEVEL_UNAVAILABLE,
        SpeakerIdentityResolutionStatus.UNAVAILABLE,
    ),
    "H6GC-U03": (
        CoverageKind.INCOMPLETE_MULTI_TURN_UNAVAILABLE,
        SpeakerIdentityResolutionStatus.UNAVAILABLE,
    ),
    "H6GC-X01": (
        CoverageKind.MIXED_SPEAKERS_CONFLICT,
        SpeakerIdentityResolutionStatus.CONFLICT,
    ),
}


class FinalInterpretation(StrEnum):
    """Only preregistered final interpretations."""

    SUPPORTS_CANDIDATE = "SUPPORTS_CANDIDATE"
    DOES_NOT_SUPPORT_CANDIDATE = "DOES_NOT_SUPPORT_CANDIDATE"
    INCONCLUSIVE_BASELINE_FAILURE_NOT_REPRODUCED = (
        "INCONCLUSIVE_BASELINE_FAILURE_NOT_REPRODUCED"
    )


class GateState(StrEnum):
    """Common frozen gate state."""

    PASS = "PASS"
    FAIL = "FAIL"


def _nonblank(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of exact file bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    """Return a UTF-8 text SHA-256."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _DeterministicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically and reject nonfinite JSON."""

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


class HeldoutModelSpec(_DeterministicModel):
    model_id: str
    revision: str
    device: str

    @model_validator(mode="after")
    def is_pinned_model(self) -> HeldoutModelSpec:
        if (self.model_id, self.revision, self.device) != (
            HELDOUT_MODEL_ID,
            HELDOUT_MODEL_REVISION,
            HELDOUT_DEVICE,
        ):
            raise ValueError("held-out model identity does not match the preregistration")
        return self


class HeldoutCondition(_DeterministicModel):
    condition: ConditionName
    checker_id: str
    checker_version: str
    composition_style: str
    identity_grounding: str

    @model_validator(mode="after")
    def matches_frozen_checker(self) -> HeldoutCondition:
        expected = {
            ConditionName.BASELINE_PLAIN: (
                "unsupported_claim",
                "1.0",
                "plain",
                "none",
            ),
            ConditionName.IDENTITY_GROUNDED: (
                "unsupported_claim_identity_grounded",
                "0.1",
                "plain",
                "explicit_turn_binding_v0.1",
            ),
        }[self.condition]
        if (
            self.checker_id,
            self.checker_version,
            self.composition_style,
            self.identity_grounding,
        ) != expected:
            raise ValueError("condition does not match its frozen checker method")
        return self


class HeldoutGateThresholds(_DeterministicModel):
    baseline_failure_min_false_alerts: StrictPositiveInt
    clean_min_candidate_entailments: StrictPositiveInt
    clean_min_rescues: StrictPositiveInt
    clean_max_regressions: StrictNonNegativeInt
    unsupported_min_candidate_detections: StrictPositiveInt
    unsupported_max_detected_to_missed: StrictNonNegativeInt
    unsupported_max_detection_drop: StrictNonNegativeInt
    control_max_relation_changes: StrictNonNegativeInt
    control_max_clean_regressions: StrictNonNegativeInt
    control_max_unsupported_regressions: StrictNonNegativeInt

    @model_validator(mode="after")
    def matches_preregistered_thresholds(self) -> HeldoutGateThresholds:
        if tuple(self.model_dump().values()) != (8, 24, 8, 2, 27, 2, 2, 2, 1, 1):
            raise ValueError("held-out gate thresholds do not match the preregistration")
        return self


class HeldoutSemanticCase(_DeterministicModel):
    case_id: str
    case_kind: SemanticCaseKind
    person_name: str
    domain: str
    transformation_kind: TransformationKind
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
    def strings_are_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="held-out semantic case field")

    @model_validator(mode="after")
    def pair_is_exact_and_nonambiguous(self) -> HeldoutSemanticCase:
        if self.source_value == self.replacement_value:
            raise ValueError("source and replacement values must differ")
        if self.source_text.count(self.source_value) != 1:
            raise ValueError("source_value must occur exactly once in source_text")
        if self.clean_hypothesis.count(self.source_value) != 1:
            raise ValueError("source_value must occur exactly once in clean_hypothesis")
        if self.replacement_value in self.source_text:
            raise ValueError("replacement_value must be absent from source_text")
        expected = self.clean_hypothesis.replace(
            self.source_value, self.replacement_value, 1
        )
        if self.unsupported_hypothesis != expected:
            raise ValueError("unsupported hypothesis must replace exactly one factual value")
        if self.clean_hypothesis.count(self.person_name) != 1:
            raise ValueError("clean hypothesis must name the person exactly once")
        if self.unsupported_hypothesis.count(self.person_name) != 1:
            raise ValueError("unsupported hypothesis must name the person exactly once")
        if self.case_kind is SemanticCaseKind.IDENTITY_FREE_CONTROL:
            if self.transformation_kind is not TransformationKind.ALREADY_NAMED_SUBJECT:
                raise ValueError("identity-free controls require already_named_subject")
            if self.source_text != self.clean_hypothesis:
                raise ValueError("identity-free clean hypothesis must equal source text")
            if _FIRST_PERSON_PATTERN.search(self.source_text):
                raise ValueError("identity-free source must contain no first-person pronoun")
        else:
            if self.transformation_kind is TransformationKind.ALREADY_NAMED_SUBJECT:
                raise ValueError("identity-sensitive cases require a first-person transformation")
            if self.person_name in self.source_text:
                raise ValueError("identity-sensitive source must not name the bound speaker")
            if not _FIRST_PERSON_PATTERN.search(self.source_text):
                raise ValueError("identity-sensitive source must contain first-person language")
        return self


class CoverageTurn(_DeterministicModel):
    turn_idx: StrictNonNegativeInt
    role: str
    content: str

    @field_validator("role", "content")
    @classmethod
    def strings_are_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="coverage turn field")


class CoverageTranscript(_DeterministicModel):
    transcript_id: str
    turns: tuple[CoverageTurn, ...]

    @field_validator("transcript_id")
    @classmethod
    def transcript_id_is_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="coverage transcript_id")

    @field_validator("turns")
    @classmethod
    def turns_are_canonical(cls, value: tuple[CoverageTurn, ...]) -> tuple[CoverageTurn, ...]:
        if not value:
            raise ValueError("coverage transcript requires turns")
        indices = [turn.turn_idx for turn in value]
        if len(indices) != len(set(indices)):
            raise ValueError("coverage turn indices must be unique")
        return tuple(sorted(value, key=lambda item: item.turn_idx))


class CoverageSourceRef(_DeterministicModel):
    transcript_id: str
    turn_idx: StrictNonNegativeInt | None = None

    @field_validator("transcript_id")
    @classmethod
    def transcript_id_is_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="coverage source transcript_id")


class CoverageBinding(_DeterministicModel):
    transcript_id: str
    turn_idx: StrictNonNegativeInt
    speaker_label: str

    @field_validator("transcript_id", "speaker_label")
    @classmethod
    def strings_are_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="coverage binding field")


class HeldoutCoverageCase(_DeterministicModel):
    case_id: str
    coverage_kind: CoverageKind
    memory_content: str
    transcripts: tuple[CoverageTranscript, ...]
    source_refs: tuple[CoverageSourceRef, ...]
    bindings: tuple[CoverageBinding, ...]
    expected_identity_status: SpeakerIdentityResolutionStatus

    @field_validator("case_id", "memory_content")
    @classmethod
    def strings_are_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="coverage case field")

    @field_validator("transcripts")
    @classmethod
    def transcripts_are_canonical(
        cls, value: tuple[CoverageTranscript, ...]
    ) -> tuple[CoverageTranscript, ...]:
        ids = [item.transcript_id for item in value]
        if not value or len(ids) != len(set(ids)):
            raise ValueError("coverage transcripts require unique IDs")
        return tuple(sorted(value, key=lambda item: item.transcript_id))

    @field_validator("source_refs")
    @classmethod
    def refs_are_nonempty(
        cls, value: tuple[CoverageSourceRef, ...]
    ) -> tuple[CoverageSourceRef, ...]:
        if not value:
            raise ValueError("coverage case requires source refs")
        return value

    @field_validator("bindings")
    @classmethod
    def bindings_are_canonical(
        cls, value: tuple[CoverageBinding, ...]
    ) -> tuple[CoverageBinding, ...]:
        keys = [(item.transcript_id, item.turn_idx) for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("coverage binding keys must be unique")
        return tuple(sorted(value, key=lambda item: (item.transcript_id, item.turn_idx)))


class HeldoutSpec(_DeterministicModel):
    """Complete preregistered fixture without semantic outcomes."""

    schema_version: str
    evaluation_id: str
    split: str
    model: HeldoutModelSpec
    conditions: tuple[HeldoutCondition, ...]
    gates: HeldoutGateThresholds
    semantic_cases: tuple[HeldoutSemanticCase, ...]
    coverage_cases: tuple[HeldoutCoverageCase, ...]

    @field_validator("schema_version")
    @classmethod
    def schema_is_frozen(cls, value: str) -> str:
        if value != HELDOUT_SCHEMA_VERSION:
            raise ValueError("held-out schema_version must be '0.1'")
        return value

    @field_validator("evaluation_id")
    @classmethod
    def evaluation_id_is_frozen(cls, value: str) -> str:
        if value != HELDOUT_EVALUATION_ID:
            raise ValueError("held-out evaluation_id does not match the freeze")
        return value

    @field_validator("split")
    @classmethod
    def split_is_held_out(cls, value: str) -> str:
        if value != HELDOUT_SPLIT:
            raise ValueError("held-out split must be held_out")
        return value

    @field_validator("conditions")
    @classmethod
    def conditions_are_canonical(
        cls, value: tuple[HeldoutCondition, ...]
    ) -> tuple[HeldoutCondition, ...]:
        return tuple(sorted(value, key=lambda item: item.condition.value))

    @field_validator("semantic_cases")
    @classmethod
    def cases_are_canonical(
        cls, value: tuple[HeldoutSemanticCase, ...]
    ) -> tuple[HeldoutSemanticCase, ...]:
        return tuple(sorted(value, key=lambda item: item.case_id))

    @field_validator("coverage_cases")
    @classmethod
    def coverage_is_canonical(
        cls, value: tuple[HeldoutCoverageCase, ...]
    ) -> tuple[HeldoutCoverageCase, ...]:
        return tuple(sorted(value, key=lambda item: item.case_id))

    @model_validator(mode="after")
    def matrix_is_complete_and_balanced(self) -> HeldoutSpec:
        if tuple(item.condition for item in self.conditions) != tuple(ConditionName):
            raise ValueError("held-out evaluation requires exactly both frozen conditions")
        if len(self.semantic_cases) != 40:
            raise ValueError("held-out evaluation requires exactly 40 semantic cases")
        ids = [item.case_id for item in self.semantic_cases]
        if len(ids) != len(set(ids)) or set(ids) != _EXPECTED_SEMANTIC_CASE_IDS:
            raise ValueError("semantic case IDs must match the frozen 40-case matrix")
        if Counter(item.case_kind for item in self.semantic_cases) != Counter(
            {
                SemanticCaseKind.IDENTITY_SENSITIVE: 30,
                SemanticCaseKind.IDENTITY_FREE_CONTROL: 10,
            }
        ):
            raise ValueError("held-out evaluation requires 30 sensitive cases and 10 controls")
        if Counter(item.transformation_kind for item in self.semantic_cases) != Counter(
            {
                TransformationKind.FIRST_PERSON_SUBJECT: 20,
                TransformationKind.FIRST_PERSON_POSSESSIVE: 10,
                TransformationKind.ALREADY_NAMED_SUBJECT: 10,
            }
        ):
            raise ValueError("held-out transformation counts do not match the freeze")
        if Counter(item.domain for item in self.semantic_cases) != Counter(
            {domain: 4 for domain in _REQUIRED_DOMAINS}
        ):
            raise ValueError("each preregistered domain requires exactly four cases")
        names = [item.person_name for item in self.semantic_cases]
        if len(names) != len(set(names)):
            raise ValueError("held-out semantic identities must be unique")
        coverage_ids = [item.case_id for item in self.coverage_cases]
        if set(coverage_ids) != set(_EXPECTED_COVERAGE_MATRIX):
            raise ValueError("coverage case IDs do not match the frozen matrix")
        if any(
            (item.coverage_kind, item.expected_identity_status)
            != _EXPECTED_COVERAGE_MATRIX[item.case_id]
            for item in self.coverage_cases
        ):
            raise ValueError("coverage kinds/statuses do not match the frozen matrix")
        return self


class SourceCoordinate(_DeterministicModel):
    transcript_id: str
    turn_idx: StrictNonNegativeInt | None


class SemanticTrial(_DeterministicModel):
    """One safe checker outcome from the paired semantic matrix."""

    case_id: str
    case_kind: SemanticCaseKind
    domain: str
    condition: ConditionName
    hypothesis_kind: HypothesisKind
    identity_status: SpeakerIdentityResolutionStatus | None
    assessed: StrictBool
    alert: StrictBool
    relation: SemanticRelation | None
    score: StrictUnitScore | None
    model_calls: StrictNonNegativeInt
    input_tokens: StrictNonNegativeInt
    output_tokens: StrictNonNegativeInt
    premise_sha256: str | None
    hypothesis_sha256: str | None
    source_coordinates: tuple[SourceCoordinate, ...]
    finding_ids: tuple[str, ...]

    @field_validator("score", mode="before")
    @classmethod
    def score_is_exact_float_or_none(cls, value: object) -> object:
        if value is not None and (isinstance(value, bool) or not isinstance(value, float)):
            raise ValueError("semantic trial score must be a Python float")
        return value

    @field_validator("finding_ids")
    @classmethod
    def finding_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("finding IDs must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> SemanticTrial:
        if self.condition is ConditionName.BASELINE_PLAIN:
            if self.identity_status is not None:
                raise ValueError("baseline trial cannot carry identity status")
        elif self.identity_status is not SpeakerIdentityResolutionStatus.RESOLVED:
            raise ValueError("semantic candidate trials require resolved identity")
        if self.assessed:
            if None in (self.relation, self.score, self.premise_sha256, self.hypothesis_sha256):
                raise ValueError("assessed trial requires relation, score, and hashes")
            if self.model_calls != 1:
                raise ValueError("each assessed semantic trial requires one model call")
            expected_alert = self.relation is not SemanticRelation.ENTAILMENT
            if self.alert is not expected_alert:
                raise ValueError("semantic alert must match frozen relation policy")
        elif any(
            value is not None
            for value in (self.relation, self.score, self.premise_sha256, self.hypothesis_sha256)
        ) or self.model_calls != 0 or self.alert:
            raise ValueError("unassessed semantic trial cannot contain a semantic outcome")
        if self.alert is not bool(self.finding_ids):
            raise ValueError("alert must match finding presence")
        return self


class CoverageTrial(_DeterministicModel):
    """One safe candidate capability outcome."""

    case_id: str
    coverage_kind: CoverageKind
    evidence_resolvable: StrictBool
    identity_status: SpeakerIdentityResolutionStatus
    assessed: StrictBool
    alert: StrictBool
    relation: SemanticRelation | None
    score: StrictUnitScore | None
    model_calls: StrictNonNegativeInt
    input_tokens: StrictNonNegativeInt
    output_tokens: StrictNonNegativeInt
    premise_sha256: str | None
    hypothesis_sha256: str | None
    source_coordinates: tuple[SourceCoordinate, ...]
    finding_ids: tuple[str, ...]

    @field_validator("score", mode="before")
    @classmethod
    def score_is_exact_float_or_none(cls, value: object) -> object:
        if value is not None and (isinstance(value, bool) or not isinstance(value, float)):
            raise ValueError("coverage trial score must be a Python float")
        return value

    @model_validator(mode="after")
    def capability_outcome_is_consistent(self) -> CoverageTrial:
        if not self.evidence_resolvable:
            raise ValueError("all preregistered coverage evidence must be structurally resolvable")
        if self.identity_status is SpeakerIdentityResolutionStatus.RESOLVED:
            if not self.assessed or self.model_calls != 1 or self.relation is None:
                raise ValueError("resolved coverage case must be assessed exactly once")
            if None in (self.score, self.premise_sha256, self.hypothesis_sha256):
                raise ValueError("assessed coverage case requires score and hashes")
            if self.alert is not (self.relation is not SemanticRelation.ENTAILMENT):
                raise ValueError("coverage alert must match frozen relation policy")
        else:
            if self.assessed or self.model_calls != 0 or self.alert or self.finding_ids:
                raise ValueError("unavailable/conflict coverage case must abstain without finding")
            if any(
                value is not None
                for value in (
                    self.relation,
                    self.score,
                    self.premise_sha256,
                    self.hypothesis_sha256,
                )
            ):
                raise ValueError("abstaining coverage case cannot contain semantic output")
        if self.alert is not bool(self.finding_ids):
            raise ValueError("coverage alert must match finding presence")
        return self


class GroupMetrics(_DeterministicModel):
    """Raw paired counts and explicit conditional/effective rates."""

    cases: StrictPositiveInt
    baseline_clean_assessed: StrictNonNegativeInt
    candidate_clean_assessed: StrictNonNegativeInt
    baseline_clean_entailments: StrictNonNegativeInt
    candidate_clean_entailments: StrictNonNegativeInt
    baseline_false_alerts: StrictNonNegativeInt
    candidate_false_alerts: StrictNonNegativeInt
    clean_rescues: StrictNonNegativeInt
    clean_regressions: StrictNonNegativeInt
    clean_exact_relation_changes: StrictNonNegativeInt
    baseline_unsupported_assessed: StrictNonNegativeInt
    candidate_unsupported_assessed: StrictNonNegativeInt
    baseline_unsupported_detections: StrictNonNegativeInt
    candidate_unsupported_detections: StrictNonNegativeInt
    baseline_unsupported_misses: StrictNonNegativeInt
    candidate_unsupported_misses: StrictNonNegativeInt
    unsupported_detected_to_missed: StrictNonNegativeInt
    unsupported_missed_to_detected: StrictNonNegativeInt
    unsupported_exact_relation_changes: StrictNonNegativeInt
    candidate_clean_conditional_entailment_rate: StrictUnitRate
    candidate_clean_effective_entailment_rate: StrictUnitRate
    candidate_unsupported_conditional_detection_rate: StrictUnitRate
    candidate_unsupported_effective_detection_rate: StrictUnitRate

    @field_validator(
        "candidate_clean_conditional_entailment_rate",
        "candidate_clean_effective_entailment_rate",
        "candidate_unsupported_conditional_detection_rate",
        "candidate_unsupported_effective_detection_rate",
        mode="before",
    )
    @classmethod
    def rates_are_python_floats(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("held-out rates must be Python floats")
        return value


class CoverageSummary(_DeterministicModel):
    total_candidate_memories: StrictPositiveInt
    declared_memories: StrictPositiveInt
    evidence_resolvable_memories: StrictPositiveInt
    identity_resolved: StrictNonNegativeInt
    identity_unavailable: StrictNonNegativeInt
    identity_conflict: StrictNonNegativeInt
    assessed_memories: StrictNonNegativeInt
    abstained_memories: StrictNonNegativeInt
    semantic_model_calls: StrictNonNegativeInt
    assessment_coverage_rate: StrictUnitRate
    unavailable_rate: StrictUnitRate
    conflict_rate: StrictUnitRate

    @field_validator("assessment_coverage_rate", "unavailable_rate", "conflict_rate", mode="before")
    @classmethod
    def rates_are_python_floats(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("coverage rates must be Python floats")
        return value

    @model_validator(mode="after")
    def partitions_are_exact(self) -> CoverageSummary:
        total = self.total_candidate_memories
        if self.declared_memories != total or self.evidence_resolvable_memories != total:
            raise ValueError("all preregistered candidate memories must be declared/resolvable")
        if self.identity_resolved + self.identity_unavailable + self.identity_conflict != total:
            raise ValueError("identity statuses must partition candidate memories")
        if self.assessed_memories + self.abstained_memories != total:
            raise ValueError("assessed and abstained must partition candidate memories")
        if self.semantic_model_calls != self.assessed_memories:
            raise ValueError("model calls must equal assessed candidate memories")
        if self.assessment_coverage_rate != self.assessed_memories / total:
            raise ValueError("assessment coverage rate does not match counts")
        if self.unavailable_rate != self.identity_unavailable / total:
            raise ValueError("unavailable rate does not match counts")
        if self.conflict_rate != self.identity_conflict / total:
            raise ValueError("conflict rate does not match counts")
        return self


class IntegrityAssessment(_DeterministicModel):
    protected_hashes_valid: StrictBool
    candidate_nonpublic: StrictBool
    candidate_noncli: StrictBool
    privacy_literal_matches: StrictNonNegativeInt


class HeldoutGates(_DeterministicModel):
    baseline_failure_reproduced: StrictBool
    clean_selectivity: GateState
    unsupported_safety: GateState
    identity_free_stability: GateState
    abstention_contract: GateState
    regression_privacy: GateState


class HeldoutExecutionResult(_DeterministicModel):
    """Canonical result whose metrics, gates, and interpretation are recomputed."""

    schema_version: str
    evaluation_id: str
    split: str
    fixture_sha256: str
    judge_id: str
    judge_version: str
    semantic_trials: tuple[SemanticTrial, ...]
    coverage_trials: tuple[CoverageTrial, ...]
    identity_sensitive: GroupMetrics
    identity_free_controls: GroupMetrics
    coverage: CoverageSummary
    integrity: IntegrityAssessment
    gates: HeldoutGates
    interpretation: FinalInterpretation

    @field_validator("semantic_trials")
    @classmethod
    def semantic_trials_are_canonical(
        cls, value: tuple[SemanticTrial, ...]
    ) -> tuple[SemanticTrial, ...]:
        return tuple(
            sorted(
                value,
                key=lambda item: (
                    item.case_id,
                    item.hypothesis_kind.value,
                    item.condition.value,
                ),
            )
        )

    @field_validator("coverage_trials")
    @classmethod
    def coverage_trials_are_canonical(
        cls, value: tuple[CoverageTrial, ...]
    ) -> tuple[CoverageTrial, ...]:
        return tuple(sorted(value, key=lambda item: item.case_id))

    @model_validator(mode="after")
    def result_matches_frozen_matrix_and_arithmetic(self) -> HeldoutExecutionResult:
        if (
            self.schema_version,
            self.evaluation_id,
            self.split,
            self.fixture_sha256,
        ) != (
            HELDOUT_SCHEMA_VERSION,
            HELDOUT_EVALUATION_ID,
            HELDOUT_SPLIT,
            HELDOUT_FIXTURE_SHA256,
        ):
            raise ValueError("held-out result identity does not match the freeze")
        if (self.judge_id, self.judge_version) != (
            f"hf-nli:{HELDOUT_MODEL_ID}",
            HELDOUT_MODEL_REVISION,
        ):
            raise ValueError("held-out result judge identity does not match pinned MiniLM")
        if len(self.semantic_trials) != 160 or len(self.coverage_trials) != 6:
            raise ValueError("held-out result requires 160 semantic and six coverage trials")
        keys = [
            (item.case_id, item.condition, item.hypothesis_kind)
            for item in self.semantic_trials
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("semantic trial keys must be unique")
        if {item.case_id for item in self.semantic_trials} != _EXPECTED_SEMANTIC_CASE_IDS:
            raise ValueError("semantic result case IDs do not match the frozen matrix")
        for case_id in _EXPECTED_SEMANTIC_CASE_IDS:
            rows = [item for item in self.semantic_trials if item.case_id == case_id]
            expected_kind = (
                SemanticCaseKind.IDENTITY_FREE_CONTROL
                if case_id.startswith("H6GC-C")
                else SemanticCaseKind.IDENTITY_SENSITIVE
            )
            if (
                len(rows) != 4
                or any(item.case_kind is not expected_kind for item in rows)
                or len({item.domain for item in rows}) != 1
            ):
                raise ValueError("each semantic case requires four consistent result rows")
        if {
            item.domain
            for item in self.semantic_trials
        } != _REQUIRED_DOMAINS:
            raise ValueError("semantic result domains do not match the frozen matrix")
        if {item.case_id for item in self.coverage_trials} != set(
            _EXPECTED_COVERAGE_MATRIX
        ):
            raise ValueError("coverage result case IDs do not match the frozen matrix")
        if any(
            (item.coverage_kind, item.identity_status)
            != _EXPECTED_COVERAGE_MATRIX[item.case_id]
            for item in self.coverage_trials
        ):
            raise ValueError("coverage result kinds/statuses do not match the frozen matrix")
        sensitive = tuple(
            item
            for item in self.semantic_trials
            if item.case_kind is SemanticCaseKind.IDENTITY_SENSITIVE
        )
        controls = tuple(
            item
            for item in self.semantic_trials
            if item.case_kind is SemanticCaseKind.IDENTITY_FREE_CONTROL
        )
        if self.identity_sensitive != summarize_group(sensitive, expected_cases=30):
            raise ValueError("identity-sensitive metrics do not match trials")
        if self.identity_free_controls != summarize_group(controls, expected_cases=10):
            raise ValueError("identity-free metrics do not match trials")
        if self.coverage != summarize_coverage(self.semantic_trials, self.coverage_trials):
            raise ValueError("coverage summary does not match trials")
        expected_gates = evaluate_gates(
            self.identity_sensitive,
            self.identity_free_controls,
            self.coverage_trials,
            self.integrity,
            HeldoutGateThresholds(
                baseline_failure_min_false_alerts=8,
                clean_min_candidate_entailments=24,
                clean_min_rescues=8,
                clean_max_regressions=2,
                unsupported_min_candidate_detections=27,
                unsupported_max_detected_to_missed=2,
                unsupported_max_detection_drop=2,
                control_max_relation_changes=2,
                control_max_clean_regressions=1,
                control_max_unsupported_regressions=1,
            ),
        )
        if self.gates != expected_gates:
            raise ValueError("held-out gates do not match preregistered arithmetic")
        if self.interpretation is not interpret_gates(self.gates):
            raise ValueError("held-out interpretation does not match preregistered logic")
        return self


class HeldoutExecutionProvenance(_DeterministicModel):
    """Safe separate execution environment and byte-identity record."""

    schema_version: str
    evaluation_id: str
    fixture_sha256: str
    evaluation_module_sha256: str
    runner_sha256: str
    candidate_sha256: str
    baseline_sha256: str
    identity_contract_sha256: str
    local_nli_sha256: str
    composition_sha256: str
    benchmark_sha256: str
    benchmark_manifest_sha256: str
    identity_probe_fixture_sha256: str
    identity_probe_result_sha256: str
    result_sha256: str
    model_id: str
    model_revision: str
    device: str
    python_version: str
    platform: str
    torch_version: str
    transformers_version: str

    @field_validator(
        "evaluation_module_sha256",
        "runner_sha256",
        "result_sha256",
        "python_version",
        "platform",
        "torch_version",
        "transformers_version",
    )
    @classmethod
    def variable_strings_are_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="execution provenance field")

    @model_validator(mode="after")
    def frozen_fields_match(self) -> HeldoutExecutionProvenance:
        expected = (
            HELDOUT_SCHEMA_VERSION,
            HELDOUT_EVALUATION_ID,
            HELDOUT_FIXTURE_SHA256,
            FROZEN_CANDIDATE_SHA256,
            FROZEN_UNSUPPORTED_CLAIM_SHA256,
            FROZEN_IDENTITY_CONTRACT_SHA256,
            FROZEN_LOCAL_NLI_SHA256,
            FROZEN_COMPOSITION_SHA256,
            FROZEN_BENCHMARK_SHA256,
            FROZEN_BENCHMARK_MANIFEST_SHA256,
            FROZEN_IDENTITY_PROBE_FIXTURE_SHA256,
            FROZEN_IDENTITY_PROBE_RESULT_SHA256,
            HELDOUT_MODEL_ID,
            HELDOUT_MODEL_REVISION,
            HELDOUT_DEVICE,
        )
        actual = (
            self.schema_version,
            self.evaluation_id,
            self.fixture_sha256,
            self.candidate_sha256,
            self.baseline_sha256,
            self.identity_contract_sha256,
            self.local_nli_sha256,
            self.composition_sha256,
            self.benchmark_sha256,
            self.benchmark_manifest_sha256,
            self.identity_probe_fixture_sha256,
            self.identity_probe_result_sha256,
            self.model_id,
            self.model_revision,
            self.device,
        )
        if actual != expected:
            raise ValueError("execution provenance does not match frozen inputs")
        return self


class HeldoutFreezeFile(_DeterministicModel):
    """One exact repository-relative file digest."""

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def path_is_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="freeze file path")

    @field_validator("sha256")
    @classmethod
    def sha_is_valid(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("freeze file SHA-256 is invalid")
        return value


class HeldoutFreezeManifest(_DeterministicModel):
    """Phase-A exact-byte manifest anchored by the preregistration commit."""

    schema_version: str
    evaluation_id: str
    files: tuple[HeldoutFreezeFile, ...]

    @field_validator("files")
    @classmethod
    def files_are_canonical(
        cls, value: tuple[HeldoutFreezeFile, ...]
    ) -> tuple[HeldoutFreezeFile, ...]:
        paths = [item.path for item in value]
        if len(paths) != len(set(paths)):
            raise ValueError("freeze manifest paths must be unique")
        return tuple(sorted(value, key=lambda item: item.path))

    @model_validator(mode="after")
    def identities_and_hashes_are_valid(self) -> HeldoutFreezeManifest:
        if (self.schema_version, self.evaluation_id) != (
            HELDOUT_SCHEMA_VERSION,
            HELDOUT_EVALUATION_ID,
        ):
            raise ValueError("held-out freeze manifest identity does not match")
        required = {
            str(HELDOUT_FIXTURE_PATH),
            "src/memlint/evaluation/identity_grounded_heldout.py",
            "tools/run_identity_grounded_heldout_v0_1.py",
            "src/memlint/checkers/unsupported_claim.py",
            "src/memlint/checkers/unsupported_claim_identity_grounded.py",
            "src/memlint/semantics/identity.py",
            "src/memlint/semantics/local_nli.py",
            "src/memlint/semantics/composition.py",
            "tests/fixtures/benchmark_v0.1.sha256.json",
            "tests/fixtures/unsupported_identity_probe_v0.1.json",
        }
        if {item.path for item in self.files} != required:
            raise ValueError("held-out freeze manifest file set does not match")
        return self


def load_heldout_spec(path: Path) -> HeldoutSpec:
    """Load and validate the complete preregistered fixture."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return HeldoutSpec.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise HeldoutInputError(f"invalid held-out fixture: {error}") from error


def preflight_heldout_fixture(path: Path) -> HeldoutSpec:
    """Verify exact fixture bytes before parsing any experimental inputs."""

    actual = sha256_file(path)
    if actual != HELDOUT_FIXTURE_SHA256:
        raise HeldoutInputError(
            f"held-out fixture SHA mismatch: expected {HELDOUT_FIXTURE_SHA256}, got {actual}"
        )
    return load_heldout_spec(path)


def validate_phase_a_manifest(repository_root: Path) -> HeldoutFreezeManifest:
    """Verify every exact byte frozen by the committed Phase-A manifest."""

    path = repository_root / HELDOUT_FREEZE_MANIFEST_PATH
    try:
        manifest = HeldoutFreezeManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise HeldoutInputError(f"invalid Phase-A freeze manifest: {error}") from error
    mismatches = []
    for item in manifest.files:
        actual = sha256_file(repository_root / item.path)
        if actual != item.sha256:
            mismatches.append(f"{item.path}: expected {item.sha256}, got {actual}")
    if mismatches:
        raise HeldoutInputError("Phase-A freeze mismatch: " + "; ".join(mismatches))
    return manifest


def validate_freshness(spec: HeldoutSpec, development_fixture_path: Path) -> None:
    """Reject reuse or close surface copies of the frozen development probe."""

    try:
        payload = json.loads(development_fixture_path.read_text(encoding="utf-8"))
        development_cases = payload["cases"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise HeldoutInputError(f"cannot validate held-out freshness: {error}") from error
    old_ids = {item["case_id"] for item in development_cases}
    old_people = {item["person_name"] for item in development_cases}
    old_texts = {
        item[field]
        for item in development_cases
        for field in ("source_text", "clean_hypothesis", "unsupported_hypothesis")
    }
    for case in spec.semantic_cases:
        if case.case_id in old_ids:
            raise HeldoutInputError("held-out case ID overlaps the development probe")
        if case.person_name in old_people:
            raise HeldoutInputError("held-out identity overlaps the development probe")
        if {
            case.source_text,
            case.clean_hypothesis,
            case.unsupported_hypothesis,
        } & old_texts:
            raise HeldoutInputError("held-out text exactly reuses the development probe")
        nearest = max(
            SequenceMatcher(
                None,
                case.source_text.casefold(),
                item["source_text"].casefold(),
            ).ratio()
            for item in development_cases
        )
        if nearest >= 0.75:
            raise HeldoutInputError("held-out source is too close to a development source")


def validate_coverage_contract(spec: HeldoutSpec) -> None:
    """Resolve every capability case without calling a semantic judge."""

    for case in spec.coverage_cases:
        store, _transcripts, bindings = _coverage_inputs(case)
        actual = resolve_speaker_identity(store.memories[0], bindings).status
        if actual is not case.expected_identity_status:
            raise HeldoutInputError(
                f"coverage case {case.case_id} expected {case.expected_identity_status.value}, "
                f"got {actual.value}"
            )


def validate_frozen_repository(
    repository_root: Path,
    *,
    external_identity_result: Path | None = None,
) -> tuple[bool, bool, bool]:
    """Verify protected bytes and that the candidate remains nonpublic/non-CLI."""

    expected_files = {
        "src/memlint/checkers/unsupported_claim.py": FROZEN_UNSUPPORTED_CLAIM_SHA256,
        "src/memlint/semantics/identity.py": FROZEN_IDENTITY_CONTRACT_SHA256,
        "src/memlint/semantics/local_nli.py": FROZEN_LOCAL_NLI_SHA256,
        "src/memlint/semantics/composition.py": FROZEN_COMPOSITION_SHA256,
        "src/memlint/checkers/unsupported_claim_identity_grounded.py": (
            FROZEN_CANDIDATE_SHA256
        ),
        "tests/fixtures/benchmark_v0.1.sha256.json": (
            FROZEN_BENCHMARK_MANIFEST_SHA256
        ),
        "tests/fixtures/unsupported_identity_probe_v0.1.json": (
            FROZEN_IDENTITY_PROBE_FIXTURE_SHA256
        ),
    }
    mismatches = []
    for relative, expected in expected_files.items():
        actual = sha256_file(repository_root / relative)
        if actual != expected:
            mismatches.append(f"{relative}: expected {expected}, got {actual}")
    from memlint.evaluation.benchmark import load_benchmark_spec

    benchmark_sha = hashlib.sha256(
        load_benchmark_spec(
            repository_root / "tests/fixtures/benchmark_v0.1/benchmark.json"
        )
        .to_json(indent=None)
        .encode("utf-8")
    ).hexdigest()
    if benchmark_sha != FROZEN_BENCHMARK_SHA256:
        mismatches.append(
            f"benchmark canonical SHA: expected {FROZEN_BENCHMARK_SHA256}, got {benchmark_sha}"
        )
    if external_identity_result is not None:
        actual = sha256_file(external_identity_result)
        if actual != FROZEN_IDENTITY_PROBE_RESULT_SHA256:
            mismatches.append(
                "6F-B result SHA: "
                f"expected {FROZEN_IDENTITY_PROBE_RESULT_SHA256}, got {actual}"
            )
    if mismatches:
        raise HeldoutInputError("frozen predecessor mismatch: " + "; ".join(mismatches))

    public_api = (repository_root / "src/memlint/checkers/__init__.py").read_text(
        encoding="utf-8"
    )
    cli_source = (repository_root / "src/memlint/cli.py").read_text(encoding="utf-8")
    candidate_name = "IdentityGroundedUnsupportedClaimChecker"
    candidate_nonpublic = candidate_name not in public_api
    candidate_noncli = (
        candidate_name not in cli_source
        and "unsupported_claim_identity_grounded" not in cli_source
    )
    if not candidate_nonpublic or not candidate_noncli:
        raise HeldoutInputError("candidate unexpectedly became public or CLI-integrated")
    return True, candidate_nonpublic, candidate_noncli


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def summarize_group(
    trials: tuple[SemanticTrial, ...], *, expected_cases: int
) -> GroupMetrics:
    """Compute preregistered paired semantic counts for one stratum."""

    by_key = {
        (item.case_id, item.condition, item.hypothesis_kind): item for item in trials
    }
    case_ids = sorted({item.case_id for item in trials})
    if len(case_ids) != expected_cases or len(trials) != expected_cases * 4:
        raise ValueError("group trial matrix is incomplete")
    expected_keys = {
        (case_id, condition, hypothesis)
        for case_id in case_ids
        for condition in ConditionName
        for hypothesis in HypothesisKind
    }
    if set(by_key) != expected_keys:
        raise ValueError("group trial matrix does not contain each paired condition")

    def row(case_id: str, condition: ConditionName, hypothesis: HypothesisKind) -> SemanticTrial:
        return by_key[(case_id, condition, hypothesis)]

    baseline_clean = [
        row(case_id, ConditionName.BASELINE_PLAIN, HypothesisKind.CLEAN)
        for case_id in case_ids
    ]
    candidate_clean = [
        row(case_id, ConditionName.IDENTITY_GROUNDED, HypothesisKind.CLEAN)
        for case_id in case_ids
    ]
    baseline_bad = [
        row(case_id, ConditionName.BASELINE_PLAIN, HypothesisKind.UNSUPPORTED)
        for case_id in case_ids
    ]
    candidate_bad = [
        row(case_id, ConditionName.IDENTITY_GROUNDED, HypothesisKind.UNSUPPORTED)
        for case_id in case_ids
    ]
    baseline_clean_entail = sum(
        item.relation is SemanticRelation.ENTAILMENT for item in baseline_clean
    )
    candidate_clean_entail = sum(
        item.relation is SemanticRelation.ENTAILMENT for item in candidate_clean
    )
    baseline_bad_detect = sum(item.alert for item in baseline_bad)
    candidate_bad_detect = sum(item.alert for item in candidate_bad)
    clean_rescues = sum(
        before.assessed
        and after.assessed
        and before.relation is not SemanticRelation.ENTAILMENT
        and after.relation is SemanticRelation.ENTAILMENT
        for before, after in zip(baseline_clean, candidate_clean, strict=True)
    )
    clean_regressions = sum(
        before.assessed
        and after.assessed
        and before.relation is SemanticRelation.ENTAILMENT
        and after.relation is not SemanticRelation.ENTAILMENT
        for before, after in zip(baseline_clean, candidate_clean, strict=True)
    )
    bad_regressions = sum(
        before.assessed and after.assessed and before.alert and not after.alert
        for before, after in zip(baseline_bad, candidate_bad, strict=True)
    )
    bad_rescues = sum(
        before.assessed and after.assessed and not before.alert and after.alert
        for before, after in zip(baseline_bad, candidate_bad, strict=True)
    )
    baseline_clean_assessed = sum(item.assessed for item in baseline_clean)
    candidate_clean_assessed = sum(item.assessed for item in candidate_clean)
    baseline_bad_assessed = sum(item.assessed for item in baseline_bad)
    candidate_bad_assessed = sum(item.assessed for item in candidate_bad)
    return GroupMetrics(
        cases=expected_cases,
        baseline_clean_assessed=baseline_clean_assessed,
        candidate_clean_assessed=candidate_clean_assessed,
        baseline_clean_entailments=baseline_clean_entail,
        candidate_clean_entailments=candidate_clean_entail,
        baseline_false_alerts=sum(item.alert for item in baseline_clean),
        candidate_false_alerts=sum(item.alert for item in candidate_clean),
        clean_rescues=clean_rescues,
        clean_regressions=clean_regressions,
        clean_exact_relation_changes=sum(
            before.assessed and after.assessed and before.relation is not after.relation
            for before, after in zip(baseline_clean, candidate_clean, strict=True)
        ),
        baseline_unsupported_assessed=baseline_bad_assessed,
        candidate_unsupported_assessed=candidate_bad_assessed,
        baseline_unsupported_detections=baseline_bad_detect,
        candidate_unsupported_detections=candidate_bad_detect,
        baseline_unsupported_misses=baseline_bad_assessed - baseline_bad_detect,
        candidate_unsupported_misses=candidate_bad_assessed - candidate_bad_detect,
        unsupported_detected_to_missed=bad_regressions,
        unsupported_missed_to_detected=bad_rescues,
        unsupported_exact_relation_changes=sum(
            before.assessed and after.assessed and before.relation is not after.relation
            for before, after in zip(baseline_bad, candidate_bad, strict=True)
        ),
        candidate_clean_conditional_entailment_rate=_rate(
            candidate_clean_entail, candidate_clean_assessed
        ),
        candidate_clean_effective_entailment_rate=candidate_clean_entail / expected_cases,
        candidate_unsupported_conditional_detection_rate=_rate(
            candidate_bad_detect, candidate_bad_assessed
        ),
        candidate_unsupported_effective_detection_rate=candidate_bad_detect / expected_cases,
    )


def summarize_coverage(
    semantic_trials: tuple[SemanticTrial, ...],
    coverage_trials: tuple[CoverageTrial, ...],
) -> CoverageSummary:
    """Count all candidate memories, including explicit capability abstentions."""

    candidate = [
        item for item in semantic_trials if item.condition is ConditionName.IDENTITY_GROUNDED
    ]
    total = len(candidate) + len(coverage_trials)
    resolved = len(candidate) + sum(
        item.identity_status is SpeakerIdentityResolutionStatus.RESOLVED
        for item in coverage_trials
    )
    unavailable = sum(
        item.identity_status is SpeakerIdentityResolutionStatus.UNAVAILABLE
        for item in coverage_trials
    )
    conflict = sum(
        item.identity_status is SpeakerIdentityResolutionStatus.CONFLICT
        for item in coverage_trials
    )
    assessed = sum(item.assessed for item in candidate) + sum(
        item.assessed for item in coverage_trials
    )
    calls = sum(item.model_calls for item in candidate) + sum(
        item.model_calls for item in coverage_trials
    )
    return CoverageSummary(
        total_candidate_memories=total,
        declared_memories=total,
        evidence_resolvable_memories=total,
        identity_resolved=resolved,
        identity_unavailable=unavailable,
        identity_conflict=conflict,
        assessed_memories=assessed,
        abstained_memories=total - assessed,
        semantic_model_calls=calls,
        assessment_coverage_rate=assessed / total,
        unavailable_rate=unavailable / total,
        conflict_rate=conflict / total,
    )


def evaluate_gates(
    sensitive: GroupMetrics,
    controls: GroupMetrics,
    coverage_trials: tuple[CoverageTrial, ...],
    integrity: IntegrityAssessment,
    thresholds: HeldoutGateThresholds,
) -> HeldoutGates:
    """Apply only the numerical and contract gates fixed before execution."""

    baseline_reproduced = (
        sensitive.baseline_false_alerts
        >= thresholds.baseline_failure_min_false_alerts
    )
    clean_pass = all(
        (
            sensitive.candidate_clean_entailments
            >= thresholds.clean_min_candidate_entailments,
            sensitive.clean_rescues >= thresholds.clean_min_rescues,
            sensitive.clean_regressions <= thresholds.clean_max_regressions,
            sensitive.baseline_false_alerts - sensitive.candidate_false_alerts
            >= thresholds.clean_min_rescues,
        )
    )
    safety_pass = all(
        (
            sensitive.candidate_unsupported_detections
            >= thresholds.unsupported_min_candidate_detections,
            sensitive.unsupported_detected_to_missed
            <= thresholds.unsupported_max_detected_to_missed,
            sensitive.baseline_unsupported_detections
            - sensitive.candidate_unsupported_detections
            <= thresholds.unsupported_max_detection_drop,
        )
    )
    stability_pass = all(
        (
            controls.clean_exact_relation_changes
            + controls.unsupported_exact_relation_changes
            <= thresholds.control_max_relation_changes,
            controls.clean_regressions <= thresholds.control_max_clean_regressions,
            controls.unsupported_detected_to_missed
            <= thresholds.control_max_unsupported_regressions,
        )
    )
    abstention_pass = all(
        item.assessed is (item.identity_status is SpeakerIdentityResolutionStatus.RESOLVED)
        and item.model_calls == (
            1 if item.identity_status is SpeakerIdentityResolutionStatus.RESOLVED else 0
        )
        and (
            item.identity_status is SpeakerIdentityResolutionStatus.RESOLVED
            or (not item.alert and not item.finding_ids)
        )
        for item in coverage_trials
    )
    integrity_pass = all(
        (
            integrity.protected_hashes_valid,
            integrity.candidate_nonpublic,
            integrity.candidate_noncli,
            integrity.privacy_literal_matches == 0,
        )
    )
    return HeldoutGates(
        baseline_failure_reproduced=baseline_reproduced,
        clean_selectivity=GateState.PASS if clean_pass else GateState.FAIL,
        unsupported_safety=GateState.PASS if safety_pass else GateState.FAIL,
        identity_free_stability=GateState.PASS if stability_pass else GateState.FAIL,
        abstention_contract=GateState.PASS if abstention_pass else GateState.FAIL,
        regression_privacy=GateState.PASS if integrity_pass else GateState.FAIL,
    )


def interpret_gates(gates: HeldoutGates) -> FinalInterpretation:
    """Map frozen gate results to the only preregistered interpretation labels."""

    if not gates.baseline_failure_reproduced:
        return FinalInterpretation.INCONCLUSIVE_BASELINE_FAILURE_NOT_REPRODUCED
    if all(
        state is GateState.PASS
        for state in (
            gates.clean_selectivity,
            gates.unsupported_safety,
            gates.identity_free_stability,
            gates.abstention_contract,
            gates.regression_privacy,
        )
    ):
        return FinalInterpretation.SUPPORTS_CANDIDATE
    return FinalInterpretation.DOES_NOT_SUPPORT_CANDIDATE


class _RecordingJudge:
    """Record only safe hashes and the last result while delegating one judgment."""

    def __init__(self, judge: SemanticJudge) -> None:
        self.judge_id = judge.judge_id
        self.judge_version = judge.judge_version
        self._judge = judge
        self.last: tuple[str, str, SemanticJudgment] | None = None

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        judgment = self._judge.judge(premise=premise, hypothesis=hypothesis)
        self.last = (sha256_text(premise), sha256_text(hypothesis), judgment)
        return judgment


def _semantic_inputs(
    case: HeldoutSemanticCase, hypothesis: str
) -> tuple[NormalizedStore, TranscriptSet, SpeakerIdentityBindings]:
    transcript_id = f"heldout-{case.case_id.lower()}"
    memory = NormalizedMemory(
        id=f"memory-{case.case_id.lower()}",
        content=hypothesis,
        source_refs=(SourceRef(transcript_id=transcript_id, turn_idx=0),),
        provenance_status=ProvenanceStatus.DECLARED,
    )
    store = NormalizedStore(adapter="heldout-v0.1", memories=(memory,))
    transcripts = TranscriptSet(
        transcripts=(
            Transcript(
                id=transcript_id,
                turns=(TranscriptTurn(index=0, role="user", content=case.source_text),),
            ),
        )
    )
    bindings = SpeakerIdentityBindings(
        bindings=(
            SpeakerIdentityBinding(
                transcript_id=transcript_id,
                turn_idx=0,
                speaker_label=case.person_name,
            ),
        )
    )
    return store, transcripts, bindings


def _coverage_inputs(
    case: HeldoutCoverageCase,
) -> tuple[NormalizedStore, TranscriptSet, SpeakerIdentityBindings]:
    memory = NormalizedMemory(
        id=f"memory-{case.case_id.lower()}",
        content=case.memory_content,
        source_refs=tuple(
            SourceRef(transcript_id=item.transcript_id, turn_idx=item.turn_idx)
            for item in case.source_refs
        ),
        provenance_status=ProvenanceStatus.DECLARED,
    )
    store = NormalizedStore(adapter="heldout-v0.1", memories=(memory,))
    transcripts = TranscriptSet(
        transcripts=tuple(
            Transcript(
                id=item.transcript_id,
                turns=tuple(
                    TranscriptTurn(
                        index=turn.turn_idx,
                        role=turn.role,
                        content=turn.content,
                    )
                    for turn in item.turns
                ),
            )
            for item in case.transcripts
        )
    )
    bindings = SpeakerIdentityBindings(
        bindings=tuple(
            SpeakerIdentityBinding(
                transcript_id=item.transcript_id,
                turn_idx=item.turn_idx,
                speaker_label=item.speaker_label,
            )
            for item in case.bindings
        )
    )
    return store, transcripts, bindings


def _source_coordinates(memory: NormalizedMemory) -> tuple[SourceCoordinate, ...]:
    return tuple(
        SourceCoordinate(transcript_id=ref.transcript_id, turn_idx=ref.turn_idx)
        for ref in memory.source_refs
    )


def _trial_values(
    result: CheckerResult,
    recorder: _RecordingJudge,
) -> tuple[
    bool,
    bool,
    SemanticRelation | None,
    float | None,
    int,
    int,
    int,
    str | None,
    str | None,
    tuple[str, ...],
]:
    assessed = result.stats.details["assessed_memories"] == 1
    if not assessed:
        return False, False, None, None, 0, 0, 0, None, None, ()
    if recorder.last is None:
        raise HeldoutInputError("assessed checker result lacks a recorded judgment")
    premise_hash, hypothesis_hash, judgment = recorder.last
    return (
        True,
        bool(result.findings),
        judgment.relation,
        float(judgment.score),
        judgment.usage.model_calls,
        judgment.usage.input_tokens,
        judgment.usage.output_tokens,
        premise_hash,
        hypothesis_hash,
        tuple(finding.finding_id for finding in result.findings),
    )


def _execute_semantic_trial(
    *,
    case: HeldoutSemanticCase,
    condition: ConditionName,
    hypothesis_kind: HypothesisKind,
    semantic_judge: SemanticJudge,
) -> SemanticTrial:
    hypothesis = (
        case.clean_hypothesis
        if hypothesis_kind is HypothesisKind.CLEAN
        else case.unsupported_hypothesis
    )
    store, transcripts, bindings = _semantic_inputs(case, hypothesis)
    recorder = _RecordingJudge(semantic_judge)
    if condition is ConditionName.BASELINE_PLAIN:
        result = UnsupportedClaimChecker(judge=recorder).check(
            store, transcripts=transcripts
        )
        identity_status = None
    else:
        result = IdentityGroundedUnsupportedClaimChecker(
            judge=recorder,
            speaker_bindings=bindings,
        ).check(store, transcripts=transcripts)
        identity_status = SpeakerIdentityResolutionStatus.RESOLVED
    values = _trial_values(result, recorder)
    return SemanticTrial(
        case_id=case.case_id,
        case_kind=case.case_kind,
        domain=case.domain,
        condition=condition,
        hypothesis_kind=hypothesis_kind,
        identity_status=identity_status,
        assessed=values[0],
        alert=values[1],
        relation=values[2],
        score=values[3],
        model_calls=values[4],
        input_tokens=values[5],
        output_tokens=values[6],
        premise_sha256=values[7],
        hypothesis_sha256=values[8],
        source_coordinates=_source_coordinates(store.memories[0]),
        finding_ids=values[9],
    )


def _execute_coverage_trial(
    *, case: HeldoutCoverageCase, semantic_judge: SemanticJudge
) -> CoverageTrial:
    store, transcripts, bindings = _coverage_inputs(case)
    memory = store.memories[0]
    resolution = resolve_speaker_identity(memory, bindings)
    if resolution.status is not case.expected_identity_status:
        raise HeldoutInputError(
            f"coverage case {case.case_id} resolved {resolution.status.value}, "
            f"expected {case.expected_identity_status.value}"
        )
    recorder = _RecordingJudge(semantic_judge)
    result = IdentityGroundedUnsupportedClaimChecker(
        judge=recorder,
        speaker_bindings=bindings,
    ).check(store, transcripts=transcripts)
    values = _trial_values(result, recorder)
    return CoverageTrial(
        case_id=case.case_id,
        coverage_kind=case.coverage_kind,
        evidence_resolvable=result.stats.details["skipped_resolution_issues"] == 0,
        identity_status=resolution.status,
        assessed=values[0],
        alert=values[1],
        relation=values[2],
        score=values[3],
        model_calls=values[4],
        input_tokens=values[5],
        output_tokens=values[6],
        premise_sha256=values[7],
        hypothesis_sha256=values[8],
        source_coordinates=_source_coordinates(memory),
        finding_ids=values[9],
    )


def privacy_literals(spec: HeldoutSpec) -> frozenset[str]:
    """Return every sensitive fixture literal forbidden from result JSON."""

    values: set[str] = set()
    for case in spec.semantic_cases:
        values.update(
            {
                case.person_name,
                case.source_text,
                case.clean_hypothesis,
                case.unsupported_hypothesis,
            }
        )
    for coverage_case in spec.coverage_cases:
        values.add(coverage_case.memory_content)
        values.update(
            turn.content
            for transcript in coverage_case.transcripts
            for turn in transcript.turns
        )
        values.update(binding.speaker_label for binding in coverage_case.bindings)
    return frozenset(value for value in values if value)


def execute_heldout(
    *,
    spec: HeldoutSpec,
    semantic_judge: SemanticJudge,
    protected_hashes_valid: bool,
    candidate_nonpublic: bool,
    candidate_noncli: bool,
) -> HeldoutExecutionResult:
    """Execute the fixed order and compute only preregistered results."""

    semantic_trials = tuple(
        _execute_semantic_trial(
            case=case,
            hypothesis_kind=hypothesis_kind,
            condition=condition,
            semantic_judge=semantic_judge,
        )
        for case in spec.semantic_cases
        for hypothesis_kind in HypothesisKind
        for condition in ConditionName
    )
    coverage_trials = tuple(
        _execute_coverage_trial(case=case, semantic_judge=semantic_judge)
        for case in spec.coverage_cases
    )
    sensitive = summarize_group(
        tuple(
            item
            for item in semantic_trials
            if item.case_kind is SemanticCaseKind.IDENTITY_SENSITIVE
        ),
        expected_cases=30,
    )
    controls = summarize_group(
        tuple(
            item
            for item in semantic_trials
            if item.case_kind is SemanticCaseKind.IDENTITY_FREE_CONTROL
        ),
        expected_cases=10,
    )
    coverage = summarize_coverage(semantic_trials, coverage_trials)
    preliminary_integrity = IntegrityAssessment(
        protected_hashes_valid=protected_hashes_valid,
        candidate_nonpublic=candidate_nonpublic,
        candidate_noncli=candidate_noncli,
        privacy_literal_matches=0,
    )
    gates = evaluate_gates(
        sensitive, controls, coverage_trials, preliminary_integrity, spec.gates
    )
    result = HeldoutExecutionResult(
        schema_version=HELDOUT_SCHEMA_VERSION,
        evaluation_id=HELDOUT_EVALUATION_ID,
        split=HELDOUT_SPLIT,
        fixture_sha256=HELDOUT_FIXTURE_SHA256,
        judge_id=semantic_judge.judge_id,
        judge_version=semantic_judge.judge_version,
        semantic_trials=semantic_trials,
        coverage_trials=coverage_trials,
        identity_sensitive=sensitive,
        identity_free_controls=controls,
        coverage=coverage,
        integrity=preliminary_integrity,
        gates=gates,
        interpretation=interpret_gates(gates),
    )
    matches = sum(literal in result.to_json() for literal in privacy_literals(spec))
    if matches:
        raise HeldoutInputError(
            f"held-out result serialization leaked {matches} forbidden fixture literals"
        )
    return result
