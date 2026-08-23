"""Explicit retrieval sufficiency policies over recorded runtime observations."""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    ValidationInfo,
    field_validator,
    model_validator,
)

from memlint.retrieval.models import RetrievalObservation, StrictPositiveInt


class RetrievalSufficiencyPolicy(StrEnum):
    """Caller-selected meaning of success for one or more expected targets."""

    ALL_EXPECTED = "all_expected"
    ANY_EXPECTED = "any_expected"


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


class RetrievalSufficiencyAssessment(BaseModel):
    """Deterministic target-set sufficiency under one explicitly selected policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    policy: RetrievalSufficiencyPolicy
    sufficient: StrictBool
    expected_memory_ids: tuple[str, ...]
    retrieved_expected_memory_ids: tuple[str, ...]
    missing_expected_memory_ids: tuple[str, ...]
    top_k: StrictPositiveInt

    @field_validator("request_id")
    @classmethod
    def request_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request_id must not be blank")
        return value

    @field_validator("expected_memory_ids")
    @classmethod
    def expected_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_ids(
            value,
            field_name="expected_memory_ids",
            require_nonempty=True,
        )

    @field_validator("retrieved_expected_memory_ids", "missing_expected_memory_ids")
    @classmethod
    def subset_ids_are_canonical(
        cls,
        value: tuple[str, ...],
        info: ValidationInfo,
    ) -> tuple[str, ...]:
        field_name = info.field_name or "assessment subset"
        return _canonical_ids(
            value,
            field_name=field_name,
            require_nonempty=False,
        )

    @model_validator(mode="after")
    def subsets_partition_expected_ids(self) -> RetrievalSufficiencyAssessment:
        expected = set(self.expected_memory_ids)
        retrieved = set(self.retrieved_expected_memory_ids)
        missing = set(self.missing_expected_memory_ids)
        if retrieved & missing:
            raise ValueError("retrieved and missing expected memory IDs must be disjoint")
        if retrieved | missing != expected:
            raise ValueError("retrieved and missing IDs must partition expected_memory_ids")

        if self.policy is RetrievalSufficiencyPolicy.ALL_EXPECTED:
            policy_result = not missing
        elif self.policy is RetrievalSufficiencyPolicy.ANY_EXPECTED:
            policy_result = bool(retrieved)
        else:  # pragma: no cover - enum validation and the frozen two-value contract exclude this
            raise AssertionError("unsupported retrieval sufficiency policy")
        if self.sufficient is not policy_result:
            raise ValueError("sufficient must match the selected retrieval policy")
        return self

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically without query, score, usage, or defect fields."""

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


def assess_retrieval_sufficiency(
    observation: RetrievalObservation,
    *,
    policy: RetrievalSufficiencyPolicy,
) -> RetrievalSufficiencyAssessment:
    """Assess recorded hit membership under an explicit caller-selected policy."""

    if not isinstance(observation, RetrievalObservation):
        raise TypeError("observation must be a RetrievalObservation")
    if not isinstance(policy, RetrievalSufficiencyPolicy):
        raise TypeError("policy must be a RetrievalSufficiencyPolicy")

    returned_ids = {hit.memory_id for hit in observation.hits}
    retrieved_expected = tuple(
        memory_id
        for memory_id in observation.expected_memory_ids
        if memory_id in returned_ids
    )
    missing_expected = tuple(
        memory_id
        for memory_id in observation.expected_memory_ids
        if memory_id not in returned_ids
    )

    if policy is RetrievalSufficiencyPolicy.ALL_EXPECTED:
        sufficient = len(missing_expected) == 0
    elif policy is RetrievalSufficiencyPolicy.ANY_EXPECTED:
        sufficient = len(retrieved_expected) > 0
    else:  # pragma: no cover - the runtime type check and frozen enum exclude this
        raise AssertionError("unsupported retrieval sufficiency policy")

    return RetrievalSufficiencyAssessment(
        request_id=observation.request_id,
        policy=policy,
        sufficient=sufficient,
        expected_memory_ids=observation.expected_memory_ids,
        retrieved_expected_memory_ids=retrieved_expected,
        missing_expected_memory_ids=missing_expected,
        top_k=observation.top_k,
    )
