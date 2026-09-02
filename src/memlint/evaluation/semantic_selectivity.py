"""Frozen H3 semantic selectivity and abstention probe contract."""

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
    StrictBool,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from memlint.semantics import (
    SemanticJudge,
    SemanticJudgment,
    SemanticRelation,
    SemanticUsage,
    semantic_judge_identity,
)

SEMANTIC_SELECTIVITY_SCHEMA_VERSION = "0.1"
SEMANTIC_SELECTIVITY_PROBE_ID = "semantic-selectivity-probe-v0.1"
SEMANTIC_SELECTIVITY_HYPOTHESIS_ID = "H3"
SEMANTIC_SELECTIVITY_EXPERIMENT_TYPE = "fresh_calibration_and_confirmatory_synthetic"
SEMANTIC_SELECTIVITY_FIXTURE_PATH = Path(
    "tests/fixtures/semantic_selectivity_probe_v0.1.json"
)
SEMANTIC_SELECTIVITY_FIXTURE_SHA256 = (
    "55dd1f32fe55511f17752586cb550211903a9ef92a92a7c05bc7785f33bba9a4"
)

SEMANTIC_SELECTIVITY_MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
SEMANTIC_SELECTIVITY_MODEL_REVISION = (
    "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
)
SEMANTIC_SELECTIVITY_DEVICE = "cpu"
SEMANTIC_SELECTIVITY_THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

CALIBRATION_UNSUPPORTED_ALERT_MINIMUM = 22
BASELINE_CLEAN_ALERT_MINIMUM = 4
CONFIRMATORY_UNSUPPORTED_ALERT_MINIMUM = 22
CONFIRMATORY_UNSUPPORTED_MAXIMUM_DROP = 1
CLEAN_ALERT_MINIMUM_REDUCTION = 3
CONFIRMATORY_ABSTAIN_MAXIMUM = 8

StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictUnitScore = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]

_FIRST_PERSON_PATTERN = re.compile(r"\b(?:i|my|we|our)\b", re.IGNORECASE)


class SemanticSelectivityInputError(ValueError):
    """The frozen H3 fixture, execution input, or result is invalid."""


class SemanticSelectivitySplit(StrEnum):
    """The calibration split precedes the untouched confirmatory split."""

    CALIBRATION = "calibration"
    CONFIRMATORY = "confirmatory"


class SemanticSelectivityDomain(StrEnum):
    """The six balanced factual domains."""

    SOFTWARE_TOOL = "software_tool"
    LOCATION = "location"
    SCHEDULE = "schedule"
    DEVICE = "device"
    PROJECT_OR_WORK = "project_or_work"
    PREFERENCE_OR_SUBSCRIPTION = "preference_or_subscription"


class SemanticSelectivityHypothesisKind(StrEnum):
    """The clearly supported and one-value unsupported hypotheses."""

    CLEAN = "clean"
    UNSUPPORTED = "unsupported"


class SemanticSelectivityDecision(StrEnum):
    """The only evaluation-layer decisions in the probe."""

    ALERT = "ALERT"
    NO_ALERT = "NO_ALERT"
    ABSTAIN = "ABSTAIN"


class CalibrationSelectionStatus(StrEnum):
    """Whether the frozen calibration rule found an eligible threshold."""

    SELECTED = "SELECTED"
    FAILED = "FAILED"


class SemanticSelectivityGate(StrEnum):
    """Exact names for the four confirmatory gates."""

    BASELINE_SELECTIVITY_CHALLENGE_PASS = "BASELINE_SELECTIVITY_CHALLENGE_PASS"
    BASELINE_SELECTIVITY_CHALLENGE_FAIL = "BASELINE_SELECTIVITY_CHALLENGE_FAIL"
    UNSUPPORTED_SAFETY_GATE_PASS = "UNSUPPORTED_SAFETY_GATE_PASS"
    UNSUPPORTED_SAFETY_GATE_FAIL = "UNSUPPORTED_SAFETY_GATE_FAIL"
    CLEAN_ALERT_REDUCTION_GATE_PASS = "CLEAN_ALERT_REDUCTION_GATE_PASS"
    CLEAN_ALERT_REDUCTION_GATE_FAIL = "CLEAN_ALERT_REDUCTION_GATE_FAIL"
    COVERAGE_GATE_PASS = "COVERAGE_GATE_PASS"
    COVERAGE_GATE_FAIL = "COVERAGE_GATE_FAIL"


class SemanticSelectivityInterpretation(StrEnum):
    """The only preregistered H3 interpretations."""

    SUPPORTS_H3 = "SUPPORTS_H3"
    DOES_NOT_SUPPORT_H3 = "DOES_NOT_SUPPORT_H3"
    INCONCLUSIVE_BASELINE_TOO_EASY = "INCONCLUSIVE_BASELINE_TOO_EASY"


DOMAIN_ORDER = tuple(SemanticSelectivityDomain)
EXPECTED_IDS_BY_SPLIT = {
    SemanticSelectivitySplit.CALIBRATION: tuple(
        f"H3-CAL-{index:02d}" for index in range(1, 25)
    ),
    SemanticSelectivitySplit.CONFIRMATORY: tuple(
        f"H3-CONF-{index:02d}" for index in range(1, 25)
    ),
}
EXPECTED_SCENARIO_IDS = tuple(
    scenario_id
    for split in SemanticSelectivitySplit
    for scenario_id in EXPECTED_IDS_BY_SPLIT[split]
)


def _nonblank(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    if value != value.strip():
        raise ValueError(f"{field_name} must not have surrounding whitespace")
    return value


def _expected_split(scenario_id: str) -> SemanticSelectivitySplit:
    if scenario_id in EXPECTED_IDS_BY_SPLIT[SemanticSelectivitySplit.CALIBRATION]:
        return SemanticSelectivitySplit.CALIBRATION
    if scenario_id in EXPECTED_IDS_BY_SPLIT[SemanticSelectivitySplit.CONFIRMATORY]:
        return SemanticSelectivitySplit.CONFIRMATORY
    raise ValueError("scenario_id is outside the frozen H3 inventory")


def _expected_domain(scenario_id: str) -> SemanticSelectivityDomain:
    split = _expected_split(scenario_id)
    index = EXPECTED_IDS_BY_SPLIT[split].index(scenario_id)
    return DOMAIN_ORDER[index // 4]


class _DeterministicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically and reject nonfinite values."""

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


class SemanticSelectivityModelSpec(_DeterministicModel):
    """Pinned semantic judge identity."""

    model_id: StrictStr
    revision: StrictStr
    device: StrictStr

    @model_validator(mode="after")
    def identity_is_frozen(self) -> SemanticSelectivityModelSpec:
        if (self.model_id, self.revision, self.device) != (
            SEMANTIC_SELECTIVITY_MODEL_ID,
            SEMANTIC_SELECTIVITY_MODEL_REVISION,
            SEMANTIC_SELECTIVITY_DEVICE,
        ):
            raise ValueError("semantic selectivity model identity does not match the freeze")
        return self


class SemanticSelectivityScenario(_DeterministicModel):
    """One identity-free premise and its controlled hypothesis pair."""

    scenario_id: StrictStr
    split: SemanticSelectivitySplit
    domain: SemanticSelectivityDomain
    subject_name: StrictStr
    premise: StrictStr
    clean_hypothesis: StrictStr
    unsupported_hypothesis: StrictStr
    source_value: StrictStr
    replacement_value: StrictStr

    @field_validator(
        "scenario_id",
        "subject_name",
        "premise",
        "clean_hypothesis",
        "unsupported_hypothesis",
        "source_value",
        "replacement_value",
    )
    @classmethod
    def strings_are_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="semantic selectivity scenario field")

    @model_validator(mode="after")
    def pair_is_identity_free_and_exact(self) -> SemanticSelectivityScenario:
        if self.split is not _expected_split(self.scenario_id):
            raise ValueError("scenario split must match its frozen ID")
        if self.domain is not _expected_domain(self.scenario_id):
            raise ValueError("scenario domain must match its frozen balanced position")
        texts = (self.premise, self.clean_hypothesis, self.unsupported_hypothesis)
        if any(_FIRST_PERSON_PATTERN.search(text) for text in texts):
            raise ValueError("semantic selectivity text must not use first-person identity")
        if any(text.count(self.subject_name) != 1 for text in texts):
            raise ValueError("every scenario text must name its third-person subject exactly once")
        if self.source_value == self.replacement_value:
            raise ValueError("source and replacement values must differ")
        if self.premise.count(self.source_value) != 1:
            raise ValueError("source_value must occur exactly once in the premise")
        if self.clean_hypothesis.count(self.source_value) != 1:
            raise ValueError("source_value must occur exactly once in the clean hypothesis")
        if self.replacement_value in self.premise or self.replacement_value in (
            self.clean_hypothesis
        ):
            raise ValueError("replacement_value must be absent from premise and clean hypothesis")
        expected_unsupported = self.clean_hypothesis.replace(
            self.source_value,
            self.replacement_value,
            1,
        )
        if self.unsupported_hypothesis != expected_unsupported:
            raise ValueError("unsupported hypothesis must replace exactly one factual value")
        return self

class SemanticSelectivitySpec(_DeterministicModel):
    """The frozen 48-scenario H3 design without semantic observations."""

    schema_version: StrictStr
    probe_id: StrictStr
    hypothesis_id: StrictStr
    experiment_type: StrictStr
    model: SemanticSelectivityModelSpec
    threshold_grid: tuple[StrictUnitScore, ...]
    scenarios: tuple[SemanticSelectivityScenario, ...]

    @model_validator(mode="after")
    def inventory_is_exact(self) -> SemanticSelectivitySpec:
        if (
            self.schema_version != SEMANTIC_SELECTIVITY_SCHEMA_VERSION
            or self.probe_id != SEMANTIC_SELECTIVITY_PROBE_ID
            or self.hypothesis_id != SEMANTIC_SELECTIVITY_HYPOTHESIS_ID
            or self.experiment_type != SEMANTIC_SELECTIVITY_EXPERIMENT_TYPE
        ):
            raise ValueError("semantic selectivity probe identity does not match the freeze")
        if self.threshold_grid != SEMANTIC_SELECTIVITY_THRESHOLD_GRID:
            raise ValueError("semantic selectivity threshold grid does not match the freeze")
        if tuple(item.scenario_id for item in self.scenarios) != EXPECTED_SCENARIO_IDS:
            raise ValueError("semantic selectivity scenarios must use canonical IDs and order")
        if Counter((item.split, item.domain) for item in self.scenarios) != Counter(
            {
                (split, domain): 4
                for split in SemanticSelectivitySplit
                for domain in SemanticSelectivityDomain
            }
        ):
            raise ValueError("each split requires four scenarios in each frozen domain")
        if len({item.subject_name for item in self.scenarios}) != 48:
            raise ValueError("each semantic selectivity scenario requires a distinct subject")
        texts = [
            text
            for item in self.scenarios
            for text in (item.premise, item.clean_hypothesis, item.unsupported_hypothesis)
        ]
        if len(set(texts)) != 144:
            raise ValueError("all semantic selectivity premise and hypothesis text must be unique")
        values = [
            value.casefold()
            for item in self.scenarios
            for value in (item.source_value, item.replacement_value)
        ]
        if len(set(values)) != 96:
            raise ValueError(
                "all semantic selectivity source and replacement values must be unique"
            )
        return self


def baseline_decision(relation: SemanticRelation) -> SemanticSelectivityDecision:
    """Apply the frozen checker-equivalent relation interpretation without abstention."""

    if not isinstance(relation, SemanticRelation):
        raise SemanticSelectivityInputError("baseline decision requires a SemanticRelation")
    if relation is SemanticRelation.ENTAILMENT:
        return SemanticSelectivityDecision.NO_ALERT
    return SemanticSelectivityDecision.ALERT


def selective_decision(
    relation: SemanticRelation,
    score: float,
    threshold: float,
) -> SemanticSelectivityDecision:
    """Apply the frozen evaluation-only confidence-abstention policy."""

    if not isinstance(relation, SemanticRelation):
        raise SemanticSelectivityInputError("selective decision requires a SemanticRelation")
    if isinstance(score, bool) or not isinstance(score, float) or not 0.0 <= score <= 1.0:
        raise SemanticSelectivityInputError("selective decision requires a finite unit score")
    if threshold not in SEMANTIC_SELECTIVITY_THRESHOLD_GRID:
        raise SemanticSelectivityInputError("threshold is outside the frozen calibration grid")
    if relation is SemanticRelation.ENTAILMENT:
        return SemanticSelectivityDecision.NO_ALERT
    if score >= threshold:
        return SemanticSelectivityDecision.ALERT
    return SemanticSelectivityDecision.ABSTAIN


class SemanticSelectivityJudgment(_DeterministicModel):
    """One raw NLI judgment plus mechanically recomputable decisions."""

    scenario_id: StrictStr
    hypothesis_kind: SemanticSelectivityHypothesisKind
    relation: SemanticRelation
    score: StrictUnitScore
    usage: SemanticUsage
    baseline_decision: SemanticSelectivityDecision
    selected_threshold: StrictUnitScore | None
    selective_decision: SemanticSelectivityDecision | None

    @field_validator("scenario_id")
    @classmethod
    def scenario_id_is_nonblank(cls, value: str) -> str:
        return _nonblank(value, field_name="semantic selectivity judgment scenario_id")

    @field_validator("score", "selected_threshold", mode="before")
    @classmethod
    def scores_are_python_floats(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("semantic selectivity scores must be Python floats")
        return value

    @model_validator(mode="after")
    def decisions_recompute_from_raw_judgment(self) -> SemanticSelectivityJudgment:
        _expected_split(self.scenario_id)
        if self.baseline_decision is not baseline_decision(self.relation):
            raise ValueError("stored baseline decision does not match raw relation")
        if self.selected_threshold is None:
            if self.selective_decision is not None:
                raise ValueError("selective decision requires the selected threshold")
        else:
            if self.selected_threshold not in SEMANTIC_SELECTIVITY_THRESHOLD_GRID:
                raise ValueError("stored threshold is outside the frozen calibration grid")
            expected = selective_decision(
                self.relation,
                self.score,
                self.selected_threshold,
            )
            if self.selective_decision is not expected:
                raise ValueError("stored selective decision does not match relation and score")
        return self


class CalibrationThresholdEvaluation(_DeterministicModel):
    """Calibration counts for one frozen threshold candidate."""

    threshold: StrictUnitScore
    clean_alerts: StrictNonNegativeInt
    unsupported_alerts: StrictNonNegativeInt
    total_abstains: StrictNonNegativeInt
    eligible: StrictBool

    @field_validator("threshold", mode="before")
    @classmethod
    def threshold_is_a_python_float(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("calibration threshold must be a Python float")
        return value

    @model_validator(mode="after")
    def counts_and_eligibility_are_valid(self) -> CalibrationThresholdEvaluation:
        if self.threshold not in SEMANTIC_SELECTIVITY_THRESHOLD_GRID:
            raise ValueError("calibration threshold is outside the frozen grid")
        if self.clean_alerts > 24 or self.unsupported_alerts > 24:
            raise ValueError("calibration alert counts cannot exceed 24 per hypothesis kind")
        if self.total_abstains > 48:
            raise ValueError("calibration abstentions cannot exceed 48 judgments")
        if self.eligible is not (
            self.unsupported_alerts >= CALIBRATION_UNSUPPORTED_ALERT_MINIMUM
        ):
            raise ValueError("calibration eligibility must match the unsupported safety floor")
        return self


class CalibrationSelection(_DeterministicModel):
    """All six calibration evaluations and their deterministic selection."""

    status: CalibrationSelectionStatus
    selected_threshold: StrictUnitScore | None
    evaluations: tuple[CalibrationThresholdEvaluation, ...]

    @field_validator("selected_threshold", mode="before")
    @classmethod
    def selected_threshold_is_a_python_float(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("selected threshold must be a Python float or null")
        return value

    @model_validator(mode="after")
    def selection_follows_the_frozen_tie_break(self) -> CalibrationSelection:
        if tuple(item.threshold for item in self.evaluations) != (
            SEMANTIC_SELECTIVITY_THRESHOLD_GRID
        ):
            raise ValueError("calibration evaluations must cover the frozen grid in order")
        eligible = tuple(item for item in self.evaluations if item.eligible)
        if not eligible:
            if self.status is not CalibrationSelectionStatus.FAILED:
                raise ValueError("no eligible threshold requires FAILED selection")
            if self.selected_threshold is not None:
                raise ValueError("failed calibration cannot carry a selected threshold")
            return self
        chosen = min(
            eligible,
            key=lambda item: (
                item.clean_alerts,
                -item.unsupported_alerts,
                item.total_abstains,
                item.threshold,
            ),
        )
        if self.status is not CalibrationSelectionStatus.SELECTED:
            raise ValueError("an eligible threshold requires SELECTED status")
        if self.selected_threshold != chosen.threshold:
            raise ValueError("selected threshold does not follow the frozen tie-break")
        return self


def _validate_judgment_matrix(
    judgments: tuple[SemanticSelectivityJudgment, ...],
    splits: tuple[SemanticSelectivitySplit, ...],
) -> None:
    if not isinstance(judgments, tuple) or any(
        not isinstance(item, SemanticSelectivityJudgment) for item in judgments
    ):
        raise SemanticSelectivityInputError(
            "semantic selectivity accounting requires typed judgment tuples"
        )
    expected_ids = tuple(
        scenario_id for split in splits for scenario_id in EXPECTED_IDS_BY_SPLIT[split]
    )
    expected_keys = tuple(
        (scenario_id, kind)
        for scenario_id in expected_ids
        for kind in SemanticSelectivityHypothesisKind
    )
    actual_keys = tuple((item.scenario_id, item.hypothesis_kind) for item in judgments)
    if actual_keys != expected_keys:
        raise SemanticSelectivityInputError(
            "semantic selectivity judgments must match the frozen canonical matrix"
        )


def select_calibration_threshold(
    judgments: tuple[SemanticSelectivityJudgment, ...],
) -> CalibrationSelection:
    """Select only from calibration judgments using the frozen deterministic rule."""

    _validate_judgment_matrix(
        judgments,
        (SemanticSelectivitySplit.CALIBRATION,),
    )
    evaluations: list[CalibrationThresholdEvaluation] = []
    for threshold in SEMANTIC_SELECTIVITY_THRESHOLD_GRID:
        decisions = tuple(
            (
                item.hypothesis_kind,
                selective_decision(item.relation, item.score, threshold),
            )
            for item in judgments
        )
        clean_alerts = sum(
            kind is SemanticSelectivityHypothesisKind.CLEAN
            and decision is SemanticSelectivityDecision.ALERT
            for kind, decision in decisions
        )
        unsupported_alerts = sum(
            kind is SemanticSelectivityHypothesisKind.UNSUPPORTED
            and decision is SemanticSelectivityDecision.ALERT
            for kind, decision in decisions
        )
        abstains = sum(
            decision is SemanticSelectivityDecision.ABSTAIN
            for _, decision in decisions
        )
        evaluations.append(
            CalibrationThresholdEvaluation(
                threshold=threshold,
                clean_alerts=clean_alerts,
                unsupported_alerts=unsupported_alerts,
                total_abstains=abstains,
                eligible=unsupported_alerts >= CALIBRATION_UNSUPPORTED_ALERT_MINIMUM,
            )
        )
    eligible = tuple(item for item in evaluations if item.eligible)
    if not eligible:
        return CalibrationSelection(
            status=CalibrationSelectionStatus.FAILED,
            selected_threshold=None,
            evaluations=tuple(evaluations),
        )
    chosen = min(
        eligible,
        key=lambda item: (
            item.clean_alerts,
            -item.unsupported_alerts,
            item.total_abstains,
            item.threshold,
        ),
    )
    return CalibrationSelection(
        status=CalibrationSelectionStatus.SELECTED,
        selected_threshold=chosen.threshold,
        evaluations=tuple(evaluations),
    )


class SemanticSelectivitySummary(_DeterministicModel):
    """Confirmatory counts, gates, and the frozen three-way interpretation."""

    calibration_selection_status: CalibrationSelectionStatus
    selected_threshold: StrictUnitScore | None
    confirmatory_scenarios: StrictNonNegativeInt
    baseline_clean_alerts: StrictNonNegativeInt | None
    baseline_unsupported_alerts: StrictNonNegativeInt | None
    selective_clean_alerts: StrictNonNegativeInt | None
    selective_clean_no_alerts: StrictNonNegativeInt | None
    selective_clean_abstains: StrictNonNegativeInt | None
    selective_unsupported_alerts: StrictNonNegativeInt | None
    selective_unsupported_no_alerts: StrictNonNegativeInt | None
    selective_unsupported_abstains: StrictNonNegativeInt | None
    total_selective_abstains: StrictNonNegativeInt | None
    baseline_selectivity_challenge_gate: SemanticSelectivityGate | None
    unsupported_safety_gate: SemanticSelectivityGate | None
    clean_alert_reduction_gate: SemanticSelectivityGate | None
    coverage_gate: SemanticSelectivityGate | None
    interpretation: SemanticSelectivityInterpretation

    @field_validator("selected_threshold", mode="before")
    @classmethod
    def summary_threshold_is_a_python_float(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError("summary threshold must be a Python float or null")
        return value

    @model_validator(mode="after")
    def counts_gates_and_interpretation_are_mechanical(
        self,
    ) -> SemanticSelectivitySummary:
        count_fields = (
            self.baseline_clean_alerts,
            self.baseline_unsupported_alerts,
            self.selective_clean_alerts,
            self.selective_clean_no_alerts,
            self.selective_clean_abstains,
            self.selective_unsupported_alerts,
            self.selective_unsupported_no_alerts,
            self.selective_unsupported_abstains,
            self.total_selective_abstains,
        )
        gate_fields = (
            self.baseline_selectivity_challenge_gate,
            self.unsupported_safety_gate,
            self.clean_alert_reduction_gate,
            self.coverage_gate,
        )
        if self.calibration_selection_status is CalibrationSelectionStatus.FAILED:
            if self.selected_threshold is not None or self.confirmatory_scenarios != 0:
                raise ValueError("failed calibration cannot report confirmatory execution")
            if any(item is not None for item in (*count_fields, *gate_fields)):
                raise ValueError("failed calibration cannot report confirmatory counts or gates")
            if self.interpretation is not SemanticSelectivityInterpretation.DOES_NOT_SUPPORT_H3:
                raise ValueError("failed calibration must map to DOES_NOT_SUPPORT_H3")
            return self

        if self.selected_threshold not in SEMANTIC_SELECTIVITY_THRESHOLD_GRID:
            raise ValueError("selected summary threshold must come from the frozen grid")
        if self.confirmatory_scenarios != 24:
            raise ValueError("selected calibration requires all 24 confirmatory scenarios")
        if any(item is None for item in (*count_fields, *gate_fields)):
            raise ValueError("completed confirmation requires all counts and gates")

        baseline_clean = self.baseline_clean_alerts
        baseline_unsupported = self.baseline_unsupported_alerts
        clean_alerts = self.selective_clean_alerts
        clean_no_alerts = self.selective_clean_no_alerts
        clean_abstains = self.selective_clean_abstains
        unsupported_alerts = self.selective_unsupported_alerts
        unsupported_no_alerts = self.selective_unsupported_no_alerts
        unsupported_abstains = self.selective_unsupported_abstains
        total_abstains = self.total_selective_abstains
        if any(
            item is None
            for item in (
                baseline_clean,
                baseline_unsupported,
                clean_alerts,
                clean_no_alerts,
                clean_abstains,
                unsupported_alerts,
                unsupported_no_alerts,
                unsupported_abstains,
                total_abstains,
            )
        ):
            raise ValueError("completed confirmation requires integer counts")
        assert baseline_clean is not None
        assert baseline_unsupported is not None
        assert clean_alerts is not None
        assert clean_no_alerts is not None
        assert clean_abstains is not None
        assert unsupported_alerts is not None
        assert unsupported_no_alerts is not None
        assert unsupported_abstains is not None
        assert total_abstains is not None
        if baseline_clean > 24 or baseline_unsupported > 24:
            raise ValueError("baseline alert counts cannot exceed 24")
        if clean_alerts + clean_no_alerts + clean_abstains != 24:
            raise ValueError("selective clean decisions must partition 24 judgments")
        if unsupported_alerts + unsupported_no_alerts + unsupported_abstains != 24:
            raise ValueError("selective unsupported decisions must partition 24 judgments")
        if total_abstains != clean_abstains + unsupported_abstains:
            raise ValueError("total abstentions must equal both hypothesis-kind abstentions")

        baseline_pass = baseline_clean >= BASELINE_CLEAN_ALERT_MINIMUM
        safety_pass = (
            unsupported_alerts >= CONFIRMATORY_UNSUPPORTED_ALERT_MINIMUM
            and unsupported_alerts
            >= baseline_unsupported - CONFIRMATORY_UNSUPPORTED_MAXIMUM_DROP
        )
        reduction_pass = (
            clean_alerts <= baseline_clean - CLEAN_ALERT_MINIMUM_REDUCTION
            and clean_alerts * 2 <= baseline_clean
        )
        coverage_pass = total_abstains <= CONFIRMATORY_ABSTAIN_MAXIMUM
        expected_gates = (
            SemanticSelectivityGate.BASELINE_SELECTIVITY_CHALLENGE_PASS
            if baseline_pass
            else SemanticSelectivityGate.BASELINE_SELECTIVITY_CHALLENGE_FAIL,
            SemanticSelectivityGate.UNSUPPORTED_SAFETY_GATE_PASS
            if safety_pass
            else SemanticSelectivityGate.UNSUPPORTED_SAFETY_GATE_FAIL,
            SemanticSelectivityGate.CLEAN_ALERT_REDUCTION_GATE_PASS
            if reduction_pass
            else SemanticSelectivityGate.CLEAN_ALERT_REDUCTION_GATE_FAIL,
            SemanticSelectivityGate.COVERAGE_GATE_PASS
            if coverage_pass
            else SemanticSelectivityGate.COVERAGE_GATE_FAIL,
        )
        if gate_fields != expected_gates:
            raise ValueError("stored H3 gates do not match the frozen arithmetic")
        expected_interpretation = (
            SemanticSelectivityInterpretation.INCONCLUSIVE_BASELINE_TOO_EASY
            if not baseline_pass
            else SemanticSelectivityInterpretation.SUPPORTS_H3
            if safety_pass and reduction_pass and coverage_pass
            else SemanticSelectivityInterpretation.DOES_NOT_SUPPORT_H3
        )
        if self.interpretation is not expected_interpretation:
            raise ValueError("stored H3 interpretation does not match the frozen rule")
        return self


def summarize_semantic_selectivity(
    judgments: tuple[SemanticSelectivityJudgment, ...],
    selection: CalibrationSelection,
) -> SemanticSelectivitySummary:
    """Recompute confirmatory counts, gates, and interpretation from stored judgments."""

    if not isinstance(selection, CalibrationSelection):
        raise SemanticSelectivityInputError("summary requires a CalibrationSelection")
    if selection.status is CalibrationSelectionStatus.FAILED:
        _validate_judgment_matrix(
            judgments,
            (SemanticSelectivitySplit.CALIBRATION,),
        )
        if any(
            item.selected_threshold is not None or item.selective_decision is not None
            for item in judgments
        ):
            raise SemanticSelectivityInputError(
                "failed calibration judgments cannot carry selective decisions"
            )
        return SemanticSelectivitySummary(
            calibration_selection_status=CalibrationSelectionStatus.FAILED,
            selected_threshold=None,
            confirmatory_scenarios=0,
            baseline_clean_alerts=None,
            baseline_unsupported_alerts=None,
            selective_clean_alerts=None,
            selective_clean_no_alerts=None,
            selective_clean_abstains=None,
            selective_unsupported_alerts=None,
            selective_unsupported_no_alerts=None,
            selective_unsupported_abstains=None,
            total_selective_abstains=None,
            baseline_selectivity_challenge_gate=None,
            unsupported_safety_gate=None,
            clean_alert_reduction_gate=None,
            coverage_gate=None,
            interpretation=SemanticSelectivityInterpretation.DOES_NOT_SUPPORT_H3,
        )

    _validate_judgment_matrix(
        judgments,
        tuple(SemanticSelectivitySplit),
    )
    threshold = selection.selected_threshold
    if threshold is None:
        raise SemanticSelectivityInputError("selected calibration is missing its threshold")
    if any(
        item.selected_threshold != threshold or item.selective_decision is None
        for item in judgments
    ):
        raise SemanticSelectivityInputError(
            "completed judgments must use the one selected threshold"
        )
    confirmatory = tuple(
        item
        for item in judgments
        if _expected_split(item.scenario_id) is SemanticSelectivitySplit.CONFIRMATORY
    )
    clean = tuple(
        item
        for item in confirmatory
        if item.hypothesis_kind is SemanticSelectivityHypothesisKind.CLEAN
    )
    unsupported = tuple(
        item
        for item in confirmatory
        if item.hypothesis_kind is SemanticSelectivityHypothesisKind.UNSUPPORTED
    )
    baseline_clean_alerts = sum(
        item.baseline_decision is SemanticSelectivityDecision.ALERT for item in clean
    )
    baseline_unsupported_alerts = sum(
        item.baseline_decision is SemanticSelectivityDecision.ALERT
        for item in unsupported
    )
    clean_counts = Counter(item.selective_decision for item in clean)
    unsupported_counts = Counter(item.selective_decision for item in unsupported)
    clean_alerts = clean_counts[SemanticSelectivityDecision.ALERT]
    clean_no_alerts = clean_counts[SemanticSelectivityDecision.NO_ALERT]
    clean_abstains = clean_counts[SemanticSelectivityDecision.ABSTAIN]
    unsupported_alerts = unsupported_counts[SemanticSelectivityDecision.ALERT]
    unsupported_no_alerts = unsupported_counts[SemanticSelectivityDecision.NO_ALERT]
    unsupported_abstains = unsupported_counts[SemanticSelectivityDecision.ABSTAIN]
    total_abstains = clean_abstains + unsupported_abstains
    baseline_pass = baseline_clean_alerts >= BASELINE_CLEAN_ALERT_MINIMUM
    safety_pass = (
        unsupported_alerts >= CONFIRMATORY_UNSUPPORTED_ALERT_MINIMUM
        and unsupported_alerts
        >= baseline_unsupported_alerts - CONFIRMATORY_UNSUPPORTED_MAXIMUM_DROP
    )
    reduction_pass = (
        clean_alerts <= baseline_clean_alerts - CLEAN_ALERT_MINIMUM_REDUCTION
        and clean_alerts * 2 <= baseline_clean_alerts
    )
    coverage_pass = total_abstains <= CONFIRMATORY_ABSTAIN_MAXIMUM
    interpretation = (
        SemanticSelectivityInterpretation.INCONCLUSIVE_BASELINE_TOO_EASY
        if not baseline_pass
        else SemanticSelectivityInterpretation.SUPPORTS_H3
        if safety_pass and reduction_pass and coverage_pass
        else SemanticSelectivityInterpretation.DOES_NOT_SUPPORT_H3
    )
    return SemanticSelectivitySummary(
        calibration_selection_status=CalibrationSelectionStatus.SELECTED,
        selected_threshold=threshold,
        confirmatory_scenarios=24,
        baseline_clean_alerts=baseline_clean_alerts,
        baseline_unsupported_alerts=baseline_unsupported_alerts,
        selective_clean_alerts=clean_alerts,
        selective_clean_no_alerts=clean_no_alerts,
        selective_clean_abstains=clean_abstains,
        selective_unsupported_alerts=unsupported_alerts,
        selective_unsupported_no_alerts=unsupported_no_alerts,
        selective_unsupported_abstains=unsupported_abstains,
        total_selective_abstains=total_abstains,
        baseline_selectivity_challenge_gate=(
            SemanticSelectivityGate.BASELINE_SELECTIVITY_CHALLENGE_PASS
            if baseline_pass
            else SemanticSelectivityGate.BASELINE_SELECTIVITY_CHALLENGE_FAIL
        ),
        unsupported_safety_gate=(
            SemanticSelectivityGate.UNSUPPORTED_SAFETY_GATE_PASS
            if safety_pass
            else SemanticSelectivityGate.UNSUPPORTED_SAFETY_GATE_FAIL
        ),
        clean_alert_reduction_gate=(
            SemanticSelectivityGate.CLEAN_ALERT_REDUCTION_GATE_PASS
            if reduction_pass
            else SemanticSelectivityGate.CLEAN_ALERT_REDUCTION_GATE_FAIL
        ),
        coverage_gate=(
            SemanticSelectivityGate.COVERAGE_GATE_PASS
            if coverage_pass
            else SemanticSelectivityGate.COVERAGE_GATE_FAIL
        ),
        interpretation=interpretation,
    )


class SemanticSelectivityExecutionResult(_DeterministicModel):
    """One deterministic one-shot result whose full chain is recomputable."""

    schema_version: StrictStr
    probe_id: StrictStr
    hypothesis_id: StrictStr
    fixture_sha256: StrictStr
    judge_id: StrictStr
    judge_version: StrictStr
    calibration_selection: CalibrationSelection
    judgments: tuple[SemanticSelectivityJudgment, ...]
    summary: SemanticSelectivitySummary

    @field_validator("judgments")
    @classmethod
    def judgments_are_canonical(
        cls,
        value: tuple[SemanticSelectivityJudgment, ...],
    ) -> tuple[SemanticSelectivityJudgment, ...]:
        return tuple(
            sorted(
                value,
                key=lambda item: (item.scenario_id, item.hypothesis_kind.value),
            )
        )

    @model_validator(mode="after")
    def result_recomputes_from_raw_judgments(
        self,
    ) -> SemanticSelectivityExecutionResult:
        if (
            self.schema_version != SEMANTIC_SELECTIVITY_SCHEMA_VERSION
            or self.probe_id != SEMANTIC_SELECTIVITY_PROBE_ID
            or self.hypothesis_id != SEMANTIC_SELECTIVITY_HYPOTHESIS_ID
            or self.fixture_sha256 != SEMANTIC_SELECTIVITY_FIXTURE_SHA256
        ):
            raise ValueError("semantic selectivity result identity does not match the freeze")
        if (self.judge_id, self.judge_version) != (
            f"hf-nli:{SEMANTIC_SELECTIVITY_MODEL_ID}",
            SEMANTIC_SELECTIVITY_MODEL_REVISION,
        ):
            raise ValueError("semantic selectivity result judge does not match pinned MiniLM")
        calibration = tuple(
            item
            for item in self.judgments
            if _expected_split(item.scenario_id) is SemanticSelectivitySplit.CALIBRATION
        )
        recomputed_selection = select_calibration_threshold(calibration)
        if self.calibration_selection != recomputed_selection:
            raise ValueError("stored calibration selection does not match raw judgments")
        splits = (
            (SemanticSelectivitySplit.CALIBRATION,)
            if self.calibration_selection.status is CalibrationSelectionStatus.FAILED
            else tuple(SemanticSelectivitySplit)
        )
        _validate_judgment_matrix(self.judgments, splits)
        expected_threshold = self.calibration_selection.selected_threshold
        if any(item.selected_threshold != expected_threshold for item in self.judgments):
            raise ValueError("all stored decisions must use the selected threshold")
        recomputed_summary = summarize_semantic_selectivity(
            self.judgments,
            self.calibration_selection,
        )
        if self.summary != recomputed_summary:
            raise ValueError("stored semantic selectivity summary does not recompute")
        return self


def validate_semantic_selectivity_model_identity() -> None:
    """Fail closed if execution constants no longer match the freeze."""

    if (
        SEMANTIC_SELECTIVITY_MODEL_ID != "cross-encoder/nli-MiniLM2-L6-H768"
        or SEMANTIC_SELECTIVITY_MODEL_REVISION
        != "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
        or SEMANTIC_SELECTIVITY_DEVICE != "cpu"
    ):
        raise SemanticSelectivityInputError(
            "semantic selectivity model identity does not match the freeze"
        )


def preflight_semantic_selectivity(
    path: Path = SEMANTIC_SELECTIVITY_FIXTURE_PATH,
) -> SemanticSelectivitySpec:
    """Verify fixture bytes and schema without constructing or calling MiniLM."""

    if not isinstance(path, Path):
        raise SemanticSelectivityInputError("semantic selectivity path must be pathlib.Path")
    try:
        fixture_bytes = path.read_bytes()
    except OSError as error:
        raise SemanticSelectivityInputError(
            "could not read frozen semantic selectivity fixture"
        ) from error
    if hashlib.sha256(fixture_bytes).hexdigest() != SEMANTIC_SELECTIVITY_FIXTURE_SHA256:
        raise SemanticSelectivityInputError("frozen semantic selectivity fixture SHA mismatch")
    try:
        spec = SemanticSelectivitySpec.model_validate_json(fixture_bytes)
    except ValidationError as error:
        raise SemanticSelectivityInputError(
            "invalid frozen semantic selectivity fixture"
        ) from error
    validate_semantic_selectivity_model_identity()
    return spec


def execute_semantic_selectivity(
    *,
    spec: SemanticSelectivitySpec,
    semantic_judge: SemanticJudge,
) -> SemanticSelectivityExecutionResult:
    """Run calibration, select mechanically, then immediately run confirmation if eligible."""

    if not isinstance(spec, SemanticSelectivitySpec):
        raise SemanticSelectivityInputError("execution requires a SemanticSelectivitySpec")
    validate_semantic_selectivity_model_identity()
    judge_id, judge_version = semantic_judge_identity(semantic_judge)
    if (judge_id, judge_version) != (
        f"hf-nli:{SEMANTIC_SELECTIVITY_MODEL_ID}",
        SEMANTIC_SELECTIVITY_MODEL_REVISION,
    ):
        raise SemanticSelectivityInputError("semantic judge identity does not match the freeze")

    def observe(
        scenarios: tuple[SemanticSelectivityScenario, ...],
        threshold: float | None,
    ) -> tuple[SemanticSelectivityJudgment, ...]:
        rows: list[SemanticSelectivityJudgment] = []
        for scenario in scenarios:
            for kind in SemanticSelectivityHypothesisKind:
                response = semantic_judge.judge(
                    premise=scenario.premise,
                    hypothesis=(
                        scenario.clean_hypothesis
                        if kind is SemanticSelectivityHypothesisKind.CLEAN
                        else scenario.unsupported_hypothesis
                    ),
                )
                if not isinstance(response, SemanticJudgment):
                    raise SemanticSelectivityInputError(
                        "semantic judge returned an invalid judgment"
                    )
                rows.append(
                    SemanticSelectivityJudgment(
                        scenario_id=scenario.scenario_id,
                        hypothesis_kind=kind,
                        relation=response.relation,
                        score=response.score,
                        usage=response.usage,
                        baseline_decision=baseline_decision(response.relation),
                        selected_threshold=threshold,
                        selective_decision=(
                            None
                            if threshold is None
                            else selective_decision(
                                response.relation,
                                response.score,
                                threshold,
                            )
                        ),
                    )
                )
        return tuple(rows)

    calibration_scenarios = tuple(
        item
        for item in spec.scenarios
        if item.split is SemanticSelectivitySplit.CALIBRATION
    )
    calibration_without_threshold = observe(calibration_scenarios, None)
    selection = select_calibration_threshold(calibration_without_threshold)
    if selection.status is CalibrationSelectionStatus.FAILED:
        judgments = calibration_without_threshold
    else:
        threshold = selection.selected_threshold
        if threshold is None:
            raise SemanticSelectivityInputError("selected calibration is missing its threshold")
        calibration_with_threshold = tuple(
            SemanticSelectivityJudgment(
                scenario_id=item.scenario_id,
                hypothesis_kind=item.hypothesis_kind,
                relation=item.relation,
                score=item.score,
                usage=item.usage,
                baseline_decision=item.baseline_decision,
                selected_threshold=threshold,
                selective_decision=selective_decision(
                    item.relation,
                    item.score,
                    threshold,
                ),
            )
            for item in calibration_without_threshold
        )
        confirmatory_scenarios = tuple(
            item
            for item in spec.scenarios
            if item.split is SemanticSelectivitySplit.CONFIRMATORY
        )
        judgments = (
            *calibration_with_threshold,
            *observe(confirmatory_scenarios, threshold),
        )
    summary = summarize_semantic_selectivity(judgments, selection)
    return SemanticSelectivityExecutionResult(
        schema_version=SEMANTIC_SELECTIVITY_SCHEMA_VERSION,
        probe_id=SEMANTIC_SELECTIVITY_PROBE_ID,
        hypothesis_id=SEMANTIC_SELECTIVITY_HYPOTHESIS_ID,
        fixture_sha256=SEMANTIC_SELECTIVITY_FIXTURE_SHA256,
        judge_id=judge_id,
        judge_version=judge_version,
        calibration_selection=selection,
        judgments=judgments,
        summary=summary,
    )
