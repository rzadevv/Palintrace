"""Immutable gold-safe evaluation accounting models."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from memlint.mutations import BaseStoreStatus, GoldLabelUnit
from memlint.retrieval import RetrievalSufficiencyPolicy
from memlint.taxonomy import DefectClass

StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]
StrictUnitRatio = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]


class EvaluationError(ValueError):
    """Base error for controlled evaluation accounting."""


class EvaluationInputError(EvaluationError):
    """Inputs cannot be reconciled into a valid controlled evaluation."""


class MutationScientificLabel(StrEnum):
    """Exact scientific label concepts available to controlled static trials."""

    INJECTED_POSITIVE = "injected_positive"
    VERIFIED_CLEAN = "verified_clean"
    UNKNOWN_NATURAL = "unknown_natural"


def _canonical_references(value: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if any(not reference.strip() for reference in value):
        raise ValueError(f"{field_name} must contain only nonblank finding IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} must contain unique finding IDs")
    return tuple(sorted(value))


class _DeterministicEvaluationModel(BaseModel):
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


class MutationTrialEvaluation(_DeterministicEvaluationModel):
    """Gold-safe accounting for one eligible controlled static mutation trial."""

    mutation_id: str
    defect_class: DefectClass
    subtype: str
    gold_unit: GoldLabelUnit
    base_store_status: BaseStoreStatus
    checker_id: str
    checker_version: str
    injected_positive_detected: StrictBool
    gold_matching_finding_ids: tuple[str, ...]
    duplicate_positive_findings: StrictNonNegativeInt
    verified_clean_alert_finding_ids: tuple[str, ...]
    unknown_natural_alert_finding_ids: tuple[str, ...]
    mutation_context_alert_finding_ids: tuple[str, ...]
    total_findings: StrictNonNegativeInt

    @field_validator("mutation_id", "subtype", "checker_id", "checker_version")
    @classmethod
    def identity_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evaluation identity strings must not be blank")
        return value

    @field_validator(
        "gold_matching_finding_ids",
        "verified_clean_alert_finding_ids",
        "unknown_natural_alert_finding_ids",
        "mutation_context_alert_finding_ids",
    )
    @classmethod
    def finding_references_are_canonical(
        cls,
        value: tuple[str, ...],
        info: object,
    ) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", None) or "finding references"
        return _canonical_references(value, field_name=field_name)

    @model_validator(mode="after")
    def accounting_is_complete_and_disjoint(self) -> MutationTrialEvaluation:
        buckets = (
            self.gold_matching_finding_ids,
            self.verified_clean_alert_finding_ids,
            self.unknown_natural_alert_finding_ids,
            self.mutation_context_alert_finding_ids,
        )
        all_references = tuple(reference for bucket in buckets for reference in bucket)
        if len(set(all_references)) != len(all_references):
            raise ValueError("finding IDs must not appear in multiple accounting buckets")
        if self.total_findings != len(all_references):
            raise ValueError("total_findings must equal all accounted findings")

        detected = bool(self.gold_matching_finding_ids)
        if self.injected_positive_detected is not detected:
            raise ValueError(
                "injected_positive_detected must match gold finding presence"
            )
        expected_duplicates = max(0, len(self.gold_matching_finding_ids) - 1)
        if self.duplicate_positive_findings != expected_duplicates:
            raise ValueError(
                "duplicate_positive_findings must count excess gold matches"
            )
        return self


class MutationEvaluationSummary(_DeterministicEvaluationModel):
    """Safe aggregate accounting over controlled injected-positive trials."""

    trials: StrictPositiveInt
    injected_positives: StrictPositiveInt
    injected_positives_detected: StrictNonNegativeInt
    injected_positives_missed: StrictNonNegativeInt
    injected_positive_recall: StrictUnitRatio
    gold_matching_findings: StrictNonNegativeInt
    duplicate_positive_findings: StrictNonNegativeInt
    verified_clean_alerts: StrictNonNegativeInt
    unknown_natural_alerts: StrictNonNegativeInt
    mutation_context_alerts: StrictNonNegativeInt
    total_findings: StrictNonNegativeInt

    @field_validator("injected_positive_recall", mode="before")
    @classmethod
    def recall_must_be_a_python_float(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("injected_positive_recall must be a Python float")
        return value

    @model_validator(mode="after")
    def aggregate_counts_are_consistent(self) -> MutationEvaluationSummary:
        if self.injected_positives != self.trials:
            raise ValueError("every eligible static trial contributes one injected positive")
        if (
            self.injected_positives_detected + self.injected_positives_missed
            != self.injected_positives
        ):
            raise ValueError("detected and missed positives must partition injected positives")
        expected_recall = self.injected_positives_detected / self.injected_positives
        if self.injected_positive_recall != expected_recall:
            raise ValueError("injected_positive_recall must match detected positive trials")
        if (
            self.duplicate_positive_findings
            != self.gold_matching_findings - self.injected_positives_detected
        ):
            raise ValueError("duplicate positive count must match excess gold findings")
        accounted_findings = (
            self.gold_matching_findings
            + self.verified_clean_alerts
            + self.unknown_natural_alerts
            + self.mutation_context_alerts
        )
        if self.total_findings != accounted_findings:
            raise ValueError("total_findings must equal all aggregate accounting buckets")
        return self


class RetrievalChallengeSummary(_DeterministicEvaluationModel):
    """Outcome counts for one controlled, baseline-eligible retrieval condition."""

    policy: RetrievalSufficiencyPolicy
    retriever_id: str
    retriever_version: str
    top_k: StrictPositiveInt
    total_cases: StrictPositiveInt
    eligible_cases: StrictNonNegativeInt
    baseline_insufficient_cases: StrictNonNegativeInt
    induced_shadowing_cases: StrictNonNegativeInt
    resilient_cases: StrictNonNegativeInt
    induced_shadowing_rate: StrictUnitRatio | None

    @field_validator("retriever_id", "retriever_version")
    @classmethod
    def retriever_identity_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("retriever identity strings must not be blank")
        return value

    @field_validator("induced_shadowing_rate", mode="before")
    @classmethod
    def rate_must_be_a_python_float_or_none(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("induced_shadowing_rate must be a Python float or null")
        return value

    @model_validator(mode="after")
    def outcome_counts_are_consistent(self) -> RetrievalChallengeSummary:
        if self.eligible_cases != self.induced_shadowing_cases + self.resilient_cases:
            raise ValueError("eligible cases must be induced plus resilient cases")
        if self.total_cases != self.eligible_cases + self.baseline_insufficient_cases:
            raise ValueError("all retrieval cases must be accounted exactly once")
        if self.eligible_cases == 0:
            if self.induced_shadowing_rate is not None:
                raise ValueError("zero eligible cases require a null induced-shadowing rate")
        else:
            expected_rate = self.induced_shadowing_cases / self.eligible_cases
            if self.induced_shadowing_rate != expected_rate:
                raise ValueError("induced_shadowing_rate must use eligible cases only")
        return self
