"""Paired retrieval challenge assessment over recorded baseline and mutated runs."""

from __future__ import annotations

import json
import string
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from palintrace.retrieval.models import RetrievalObservation, StrictPositiveInt
from palintrace.retrieval.policy import (
    RetrievalSufficiencyAssessment,
    RetrievalSufficiencyPolicy,
    assess_retrieval_sufficiency,
)


class RetrievalChallengeOutcome(StrEnum):
    """Exact controlled outcomes for one compatible baseline/mutated pair."""

    INDUCED_SHADOWING = "induced_shadowing"
    RESILIENT = "resilient"
    BASELINE_INSUFFICIENT = "baseline_insufficient"


class RetrievalChallengeInputError(ValueError):
    """The recorded runs cannot support the requested controlled comparison."""


def _canonical_ids(
    value: tuple[str, ...],
    *,
    field_name: str,
    require_nonempty: bool,
) -> tuple[str, ...]:
    if require_nonempty and not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(not memory_id.strip() for memory_id in value):
        raise ValueError(f"{field_name} must contain only nonblank IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} must be unique")
    return tuple(sorted(value))


class PairedRetrievalChallengeAssessment(BaseModel):
    """Self-validating result for one compatible recorded retrieval challenge pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    policy: RetrievalSufficiencyPolicy
    outcome: RetrievalChallengeOutcome
    baseline_request_id: str
    mutated_request_id: str
    query_sha256: str
    expected_memory_ids: tuple[str, ...]
    top_k: StrictPositiveInt
    retriever_id: str
    retriever_version: str
    baseline_sufficient: StrictBool
    mutated_sufficient: StrictBool
    baseline_retrieved_expected_memory_ids: tuple[str, ...]
    baseline_missing_expected_memory_ids: tuple[str, ...]
    mutated_retrieved_expected_memory_ids: tuple[str, ...]
    mutated_missing_expected_memory_ids: tuple[str, ...]

    @field_validator(
        "case_id",
        "baseline_request_id",
        "mutated_request_id",
        "retriever_id",
        "retriever_version",
    )
    @classmethod
    def identity_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("paired challenge identity strings must not be blank")
        return value

    @field_validator("query_sha256")
    @classmethod
    def query_sha256_must_be_canonical(cls, value: str) -> str:
        if len(value) != 64 or any(character not in string.hexdigits for character in value):
            raise ValueError("query_sha256 must contain exactly 64 hexadecimal characters")
        if value != value.lower():
            raise ValueError("query_sha256 must use lowercase hexadecimal")
        return value

    @field_validator("expected_memory_ids")
    @classmethod
    def expected_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_ids(
            value,
            field_name="expected_memory_ids",
            require_nonempty=True,
        )

    @field_validator(
        "baseline_retrieved_expected_memory_ids",
        "baseline_missing_expected_memory_ids",
        "mutated_retrieved_expected_memory_ids",
        "mutated_missing_expected_memory_ids",
    )
    @classmethod
    def partition_ids_are_canonical(
        cls,
        value: tuple[str, ...],
        info: ValidationInfo,
    ) -> tuple[str, ...]:
        return _canonical_ids(
            value,
            field_name=info.field_name or "challenge partition",
            require_nonempty=False,
        )

    @model_validator(mode="after")
    def assessments_and_outcome_are_consistent(self) -> PairedRetrievalChallengeAssessment:
        for run_name, request_id, sufficient, retrieved, missing in (
            (
                "baseline",
                self.baseline_request_id,
                self.baseline_sufficient,
                self.baseline_retrieved_expected_memory_ids,
                self.baseline_missing_expected_memory_ids,
            ),
            (
                "mutated",
                self.mutated_request_id,
                self.mutated_sufficient,
                self.mutated_retrieved_expected_memory_ids,
                self.mutated_missing_expected_memory_ids,
            ),
        ):
            try:
                RetrievalSufficiencyAssessment(
                    request_id=request_id,
                    policy=self.policy,
                    sufficient=sufficient,
                    expected_memory_ids=self.expected_memory_ids,
                    retrieved_expected_memory_ids=retrieved,
                    missing_expected_memory_ids=missing,
                    top_k=self.top_k,
                )
            except ValidationError as error:
                raise ValueError(f"{run_name} sufficiency partition is inconsistent") from error

        if not self.baseline_sufficient:
            expected_outcome = RetrievalChallengeOutcome.BASELINE_INSUFFICIENT
        elif self.mutated_sufficient:
            expected_outcome = RetrievalChallengeOutcome.RESILIENT
        else:
            expected_outcome = RetrievalChallengeOutcome.INDUCED_SHADOWING
        if self.outcome is not expected_outcome:
            raise ValueError("outcome must match baseline and mutated sufficiency")
        return self

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the paired assessment deterministically."""

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


def assess_paired_retrieval_challenge(
    baseline: RetrievalObservation,
    mutated: RetrievalObservation,
    *,
    policy: RetrievalSufficiencyPolicy,
    case_id: str,
) -> PairedRetrievalChallengeAssessment:
    """Assess mutation effect from one compatible baseline/mutated observation pair."""

    if not isinstance(baseline, RetrievalObservation):
        raise RetrievalChallengeInputError("baseline must be a RetrievalObservation")
    if not isinstance(mutated, RetrievalObservation):
        raise RetrievalChallengeInputError("mutated must be a RetrievalObservation")
    if not isinstance(policy, RetrievalSufficiencyPolicy):
        raise RetrievalChallengeInputError("policy must be a RetrievalSufficiencyPolicy")
    if not isinstance(case_id, str) or not case_id.strip():
        raise RetrievalChallengeInputError("case_id must be a nonblank string")

    compatibility_fields = (
        "query_sha256",
        "expected_memory_ids",
        "top_k",
        "retriever_id",
        "retriever_version",
    )
    mismatches = tuple(
        field_name
        for field_name in compatibility_fields
        if getattr(baseline, field_name) != getattr(mutated, field_name)
    )
    if mismatches:
        raise RetrievalChallengeInputError(
            "paired retrieval observations must match exactly on: " + ", ".join(mismatches)
        )

    baseline_assessment = assess_retrieval_sufficiency(baseline, policy=policy)
    mutated_assessment = assess_retrieval_sufficiency(mutated, policy=policy)

    if not baseline_assessment.sufficient:
        outcome = RetrievalChallengeOutcome.BASELINE_INSUFFICIENT
    elif mutated_assessment.sufficient:
        outcome = RetrievalChallengeOutcome.RESILIENT
    else:
        outcome = RetrievalChallengeOutcome.INDUCED_SHADOWING

    return PairedRetrievalChallengeAssessment(
        case_id=case_id,
        policy=policy,
        outcome=outcome,
        baseline_request_id=baseline.request_id,
        mutated_request_id=mutated.request_id,
        query_sha256=baseline.query_sha256,
        expected_memory_ids=baseline.expected_memory_ids,
        top_k=baseline.top_k,
        retriever_id=baseline.retriever_id,
        retriever_version=baseline.retriever_version,
        baseline_sufficient=baseline_assessment.sufficient,
        mutated_sufficient=mutated_assessment.sufficient,
        baseline_retrieved_expected_memory_ids=(
            baseline_assessment.retrieved_expected_memory_ids
        ),
        baseline_missing_expected_memory_ids=baseline_assessment.missing_expected_memory_ids,
        mutated_retrieved_expected_memory_ids=(
            mutated_assessment.retrieved_expected_memory_ids
        ),
        mutated_missing_expected_memory_ids=mutated_assessment.missing_expected_memory_ids,
    )
