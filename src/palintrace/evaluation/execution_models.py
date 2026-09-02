"""Immutable benchmark execution, clean-control, and provenance models."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from palintrace.checkers import CheckerResult
from palintrace.evaluation.benchmark import (
    BENCHMARK_ID,
    BENCHMARK_SCHEMA_VERSION,
    BENCHMARK_SPEC_SHA256,
    STATIC_BENCHMARK_DEFECTS,
)
from palintrace.evaluation.models import MutationTrialEvaluation, RetrievalChallengeSummary
from palintrace.retrieval import PairedRetrievalChallengeAssessment, RetrievalObservation
from palintrace.taxonomy import DefectClass

StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]
StrictUnitRatio = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]


def _nonblank(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _canonical_finding_ids(value: tuple[str, ...]) -> tuple[str, ...]:
    if any(not finding_id.strip() for finding_id in value):
        raise ValueError("finding IDs must not be blank")
    if len(set(value)) != len(value):
        raise ValueError("finding IDs must be unique")
    return tuple(sorted(value))


class _DeterministicExecutionModel(BaseModel):
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


class CleanControlEvaluation(_DeterministicExecutionModel):
    """Case-level accounting for one explicitly curated-clean benchmark store."""

    case_id: str
    defect_class: DefectClass
    base_fixture_id: str
    checker_id: str
    checker_version: str
    alert_present: StrictBool
    finding_ids: tuple[str, ...]
    findings_emitted: StrictNonNegativeInt

    @field_validator("case_id", "base_fixture_id", "checker_id", "checker_version")
    @classmethod
    def identity_strings_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="clean-control identity")

    @field_validator("finding_ids")
    @classmethod
    def finding_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_finding_ids(value)

    @model_validator(mode="after")
    def alert_accounting_is_consistent(self) -> CleanControlEvaluation:
        if self.findings_emitted != len(self.finding_ids):
            raise ValueError("findings_emitted must equal the finding reference count")
        if self.alert_present is not bool(self.finding_ids):
            raise ValueError("alert_present must match finding presence")
        return self


class StaticDefectBenchmarkSummary(_DeterministicExecutionModel):
    """Descriptive positive and curated-clean accounting for one defect class."""

    defect_class: DefectClass
    positive_trials: StrictPositiveInt
    positive_trials_detected: StrictNonNegativeInt
    positive_trials_missed: StrictNonNegativeInt
    injected_positive_recall: StrictUnitRatio
    clean_controls: StrictPositiveInt
    clean_controls_with_alert: StrictNonNegativeInt
    clean_control_alert_rate: StrictUnitRatio
    verified_clean_alerts: StrictNonNegativeInt
    unknown_natural_alerts: StrictNonNegativeInt
    mutation_context_alerts: StrictNonNegativeInt
    duplicate_positive_findings: StrictNonNegativeInt

    @field_validator("injected_positive_recall", "clean_control_alert_rate", mode="before")
    @classmethod
    def ratios_must_be_python_floats(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("benchmark rates must be Python floats")
        return value

    @model_validator(mode="after")
    def counts_and_rates_are_consistent(self) -> StaticDefectBenchmarkSummary:
        if self.defect_class not in STATIC_BENCHMARK_DEFECTS:
            raise ValueError("static summaries require an implemented benchmark defect")
        if self.positive_trials_detected + self.positive_trials_missed != self.positive_trials:
            raise ValueError("detected and missed trials must partition positive trials")
        if self.positive_trials_detected / self.positive_trials != self.injected_positive_recall:
            raise ValueError("injected_positive_recall must match positive trial counts")
        if self.clean_controls_with_alert > self.clean_controls:
            raise ValueError("alerting clean controls cannot exceed clean controls")
        if self.clean_controls_with_alert / self.clean_controls != self.clean_control_alert_rate:
            raise ValueError("clean_control_alert_rate must be case-level")
        return self


class StaticCaseExecution(_DeterministicExecutionModel):
    """Public result for one static controlled mutation case."""

    case_id: str
    mutation_id: str
    checker_result: CheckerResult
    trial_evaluation: MutationTrialEvaluation

    @field_validator("case_id", "mutation_id")
    @classmethod
    def identity_strings_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="static execution identity")

    @model_validator(mode="after")
    def result_and_evaluation_must_match(self) -> StaticCaseExecution:
        trial = self.trial_evaluation
        result = self.checker_result
        if self.mutation_id != trial.mutation_id:
            raise ValueError("mutation_id must match the trial evaluation")
        if result.defect_class is not trial.defect_class:
            raise ValueError("checker result and trial defect classes must match")
        if (result.checker_id, result.checker_version) != (
            trial.checker_id,
            trial.checker_version,
        ):
            raise ValueError("checker result and trial identities must match")
        return self


class CleanControlCaseExecution(_DeterministicExecutionModel):
    """Public result for one unmutated curated-clean control."""

    case_id: str
    checker_result: CheckerResult
    clean_control_evaluation: CleanControlEvaluation

    @field_validator("case_id")
    @classmethod
    def case_id_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="clean-control case_id")

    @model_validator(mode="after")
    def result_and_evaluation_must_match(self) -> CleanControlCaseExecution:
        evaluation = self.clean_control_evaluation
        result = self.checker_result
        if self.case_id != evaluation.case_id:
            raise ValueError("case_id must match the clean-control evaluation")
        if result.defect_class is not evaluation.defect_class:
            raise ValueError("checker result and clean-control defect classes must match")
        if (result.checker_id, result.checker_version) != (
            evaluation.checker_id,
            evaluation.checker_version,
        ):
            raise ValueError("checker result and clean-control identities must match")
        return self


class RetrievalCaseExecution(_DeterministicExecutionModel):
    """Public paired runtime evidence for one retrieval challenge."""

    case_id: str
    baseline_observation: RetrievalObservation
    mutated_observation: RetrievalObservation
    paired_assessment: PairedRetrievalChallengeAssessment

    @field_validator("case_id")
    @classmethod
    def case_id_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="retrieval case_id")

    @model_validator(mode="after")
    def observations_and_assessment_must_match(self) -> RetrievalCaseExecution:
        assessment = self.paired_assessment
        if self.case_id != assessment.case_id:
            raise ValueError("case_id must match the paired assessment")
        if self.baseline_observation.request_id != assessment.baseline_request_id:
            raise ValueError("baseline request ID must match the paired assessment")
        if self.mutated_observation.request_id != assessment.mutated_request_id:
            raise ValueError("mutated request ID must match the paired assessment")
        return self


class BenchmarkExecutionResult(_DeterministicExecutionModel):
    """Canonical result artifact for one complete frozen benchmark execution."""

    schema_version: str = BENCHMARK_SCHEMA_VERSION
    benchmark_id: str
    benchmark_spec_sha256: str
    static_cases: tuple[StaticCaseExecution, ...]
    clean_controls: tuple[CleanControlCaseExecution, ...]
    static_defect_summaries: tuple[StaticDefectBenchmarkSummary, ...]
    retrieval_cases: tuple[RetrievalCaseExecution, ...]
    retrieval_summary: RetrievalChallengeSummary

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_frozen(cls, value: str) -> str:
        if value != BENCHMARK_SCHEMA_VERSION:
            raise ValueError("benchmark execution schema_version must be '0.1'")
        return value

    @field_validator("benchmark_id")
    @classmethod
    def benchmark_id_is_frozen(cls, value: str) -> str:
        if value != BENCHMARK_ID:
            raise ValueError("benchmark execution benchmark_id does not match v0.1")
        return value

    @field_validator("benchmark_spec_sha256")
    @classmethod
    def benchmark_sha_is_frozen(cls, value: str) -> str:
        if value != BENCHMARK_SPEC_SHA256:
            raise ValueError("benchmark execution SHA does not match frozen v0.1")
        return value

    @field_validator("static_cases")
    @classmethod
    def static_cases_are_canonical(
        cls, value: tuple[StaticCaseExecution, ...]
    ) -> tuple[StaticCaseExecution, ...]:
        return tuple(sorted(value, key=lambda item: item.case_id))

    @field_validator("clean_controls")
    @classmethod
    def clean_controls_are_canonical(
        cls, value: tuple[CleanControlCaseExecution, ...]
    ) -> tuple[CleanControlCaseExecution, ...]:
        return tuple(sorted(value, key=lambda item: item.case_id))

    @field_validator("static_defect_summaries")
    @classmethod
    def summaries_are_canonical(
        cls, value: tuple[StaticDefectBenchmarkSummary, ...]
    ) -> tuple[StaticDefectBenchmarkSummary, ...]:
        return tuple(sorted(value, key=lambda item: item.defect_class.value))

    @field_validator("retrieval_cases")
    @classmethod
    def retrieval_cases_are_canonical(
        cls, value: tuple[RetrievalCaseExecution, ...]
    ) -> tuple[RetrievalCaseExecution, ...]:
        return tuple(sorted(value, key=lambda item: item.case_id))

    @model_validator(mode="after")
    def all_case_and_summary_identities_are_unique(self) -> BenchmarkExecutionResult:
        case_ids = [
            *(item.case_id for item in self.static_cases),
            *(item.case_id for item in self.clean_controls),
            *(item.case_id for item in self.retrieval_cases),
        ]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("benchmark execution case IDs must be globally unique")
        summary_defects = tuple(item.defect_class for item in self.static_defect_summaries)
        if set(summary_defects) != set(STATIC_BENCHMARK_DEFECTS):
            raise ValueError("benchmark execution requires all five static summaries")
        if len(summary_defects) != len(set(summary_defects)):
            raise ValueError("benchmark execution static summary defects must be unique")
        return self


class BenchmarkExecutionProvenance(_DeterministicExecutionModel):
    """Safe environment identity stored separately from benchmark scoring."""

    schema_version: str = BENCHMARK_SCHEMA_VERSION
    benchmark_id: str
    benchmark_spec_sha256: str
    python_version: str
    platform: str
    torch_version: str
    transformers_version: str
    tokenizers_version: str
    safetensors_version: str
    unsupported_model_id: str
    unsupported_model_revision: str
    device: str

    @field_validator(
        "benchmark_id",
        "benchmark_spec_sha256",
        "python_version",
        "platform",
        "torch_version",
        "transformers_version",
        "tokenizers_version",
        "safetensors_version",
        "unsupported_model_id",
        "unsupported_model_revision",
        "device",
    )
    @classmethod
    def provenance_strings_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="execution provenance")

    @model_validator(mode="after")
    def frozen_method_identity_must_match(self) -> BenchmarkExecutionProvenance:
        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ValueError("provenance schema_version must be '0.1'")
        if self.benchmark_id != BENCHMARK_ID:
            raise ValueError("provenance benchmark_id does not match v0.1")
        if self.benchmark_spec_sha256 != BENCHMARK_SPEC_SHA256:
            raise ValueError("provenance benchmark SHA does not match v0.1")
        if self.unsupported_model_id != "cross-encoder/nli-MiniLM2-L6-H768":
            raise ValueError("provenance unsupported model ID does not match the freeze")
        if (
            self.unsupported_model_revision
            != "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
        ):
            raise ValueError("provenance unsupported model revision does not match the freeze")
        if self.device != "cpu":
            raise ValueError("benchmark v0.1 execution device must be cpu")
        return self
