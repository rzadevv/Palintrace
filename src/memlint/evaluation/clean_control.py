"""Curated-clean control accounting and per-defect benchmark summaries."""

from __future__ import annotations

from collections import defaultdict

from memlint.checkers import CheckerResult
from memlint.evaluation.benchmark import (
    STATIC_BENCHMARK_DEFECTS,
    BenchmarkCaseKind,
    BenchmarkFixture,
    BenchmarkSplit,
    CleanControlBenchmarkCase,
)
from memlint.evaluation.execution_models import (
    CleanControlEvaluation,
    StaticDefectBenchmarkSummary,
)
from memlint.evaluation.models import EvaluationInputError, MutationTrialEvaluation
from memlint.models import NormalizedStore
from memlint.mutations import BaseStoreStatus
from memlint.taxonomy import DefectClass


def evaluate_clean_control(
    *,
    case: CleanControlBenchmarkCase,
    fixture: BenchmarkFixture,
    store: NormalizedStore,
    result: CheckerResult,
) -> CleanControlEvaluation:
    """Account for findings on one explicitly curated-clean unmutated fixture."""

    if not isinstance(case, CleanControlBenchmarkCase):
        raise EvaluationInputError("case must be a CleanControlBenchmarkCase")
    if not isinstance(fixture, BenchmarkFixture):
        raise EvaluationInputError("fixture must be a BenchmarkFixture")
    if not isinstance(store, NormalizedStore):
        raise EvaluationInputError("store must be a NormalizedStore")
    if not isinstance(result, CheckerResult):
        raise EvaluationInputError("result must be a CheckerResult")
    if case.split is not BenchmarkSplit.HELD_OUT:
        raise EvaluationInputError("clean-control evaluation requires a held-out case")
    if case.kind is not BenchmarkCaseKind.CLEAN_CONTROL:
        raise EvaluationInputError("clean-control evaluation requires clean_control kind")
    if fixture.fixture_id != case.base_fixture_id:
        raise EvaluationInputError("clean-control fixture does not match the case")
    if fixture.base_store_status is not BaseStoreStatus.CURATED_CLEAN:
        raise EvaluationInputError("clean controls require an explicitly curated-clean fixture")
    if result.defect_class is not case.defect_class:
        raise EvaluationInputError("clean-control result defect class does not match the case")

    store_ids = {memory.id for memory in store.memories}
    for finding in result.findings:
        unknown_ids = tuple(sorted(set(finding.memory_ids) - store_ids))
        if unknown_ids:
            raise EvaluationInputError(
                "clean-control finding references IDs absent from the fixture store: "
                + ", ".join(unknown_ids)
            )

    finding_ids = tuple(finding.finding_id for finding in result.findings)
    return CleanControlEvaluation(
        case_id=case.case_id,
        defect_class=case.defect_class,
        base_fixture_id=case.base_fixture_id,
        checker_id=result.checker_id,
        checker_version=result.checker_version,
        alert_present=bool(finding_ids),
        finding_ids=finding_ids,
        findings_emitted=len(finding_ids),
    )


def summarize_static_defect_benchmark(
    *,
    mutation_trials: tuple[MutationTrialEvaluation, ...],
    clean_controls: tuple[CleanControlEvaluation, ...],
) -> tuple[StaticDefectBenchmarkSummary, ...]:
    """Report descriptive positive and clean-control accounting per defect class."""

    if not isinstance(mutation_trials, tuple) or not mutation_trials:
        raise EvaluationInputError("static benchmark summary requires mutation trials")
    if not isinstance(clean_controls, tuple) or not clean_controls:
        raise EvaluationInputError("static benchmark summary requires clean controls")
    if any(not isinstance(item, MutationTrialEvaluation) for item in mutation_trials):
        raise EvaluationInputError("mutation_trials must contain only trial evaluations")
    if any(not isinstance(item, CleanControlEvaluation) for item in clean_controls):
        raise EvaluationInputError("clean_controls must contain only clean-control evaluations")

    trial_groups: dict[DefectClass, list[MutationTrialEvaluation]] = defaultdict(list)
    control_groups: dict[DefectClass, list[CleanControlEvaluation]] = defaultdict(list)
    for trial in mutation_trials:
        trial_groups[trial.defect_class].append(trial)
    for control in clean_controls:
        control_groups[control.defect_class].append(control)

    required = set(STATIC_BENCHMARK_DEFECTS)
    if set(trial_groups) != required or set(control_groups) != required:
        raise EvaluationInputError(
            "benchmark v0.1 static summaries require all five implemented defect classes"
        )

    summaries: list[StaticDefectBenchmarkSummary] = []
    for defect_class in sorted(required, key=lambda item: item.value):
        trials = trial_groups[defect_class]
        controls = control_groups[defect_class]
        detected = sum(trial.injected_positive_detected for trial in trials)
        controls_with_alert = sum(control.alert_present for control in controls)
        summaries.append(
            StaticDefectBenchmarkSummary(
                defect_class=defect_class,
                positive_trials=len(trials),
                positive_trials_detected=detected,
                positive_trials_missed=len(trials) - detected,
                injected_positive_recall=detected / len(trials),
                clean_controls=len(controls),
                clean_controls_with_alert=controls_with_alert,
                clean_control_alert_rate=controls_with_alert / len(controls),
                verified_clean_alerts=(
                    sum(
                        len(trial.verified_clean_alert_finding_ids)
                        for trial in trials
                    )
                    + sum(control.findings_emitted for control in controls)
                ),
                unknown_natural_alerts=sum(
                    len(trial.unknown_natural_alert_finding_ids) for trial in trials
                ),
                mutation_context_alerts=sum(
                    len(trial.mutation_context_alert_finding_ids) for trial in trials
                ),
                duplicate_positive_findings=sum(
                    trial.duplicate_positive_findings for trial in trials
                ),
            )
        )
    return tuple(summaries)
