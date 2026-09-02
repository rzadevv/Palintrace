"""Gold-safe accounting for controlled static mutation trials."""

from __future__ import annotations

from palintrace.checkers import CheckerResult, Finding
from palintrace.evaluation.models import (
    EvaluationInputError,
    MutationEvaluationSummary,
    MutationTrialEvaluation,
)
from palintrace.models import NormalizedStore, TranscriptSet
from palintrace.mutations import BaseStoreStatus, GoldLabelUnit, MutationManifest
from palintrace.mutations.base import semantic_store_digest, transcript_set_digest
from palintrace.taxonomy import DefectClass


def _canonical_evaluation_unit(finding: Finding) -> tuple[str, ...]:
    return tuple(sorted(finding.memory_ids))


def _validate_static_trial_inputs(
    *,
    base_store: NormalizedStore,
    mutated_store: NormalizedStore,
    manifest: MutationManifest,
    result: CheckerResult,
    transcripts: TranscriptSet | None,
) -> None:
    if not isinstance(base_store, NormalizedStore):
        raise EvaluationInputError("base_store must be a NormalizedStore")
    if not isinstance(mutated_store, NormalizedStore):
        raise EvaluationInputError("mutated_store must be a NormalizedStore")
    if not isinstance(manifest, MutationManifest):
        raise EvaluationInputError("manifest must be a MutationManifest")
    if not isinstance(result, CheckerResult):
        raise EvaluationInputError("result must be a CheckerResult")
    if transcripts is not None and not isinstance(transcripts, TranscriptSet):
        raise EvaluationInputError("transcripts must be a TranscriptSet or null")

    if (
        manifest.defect_class is DefectClass.RETRIEVAL_SHADOWING
        or manifest.requires_runtime_validation
        or not manifest.gold_label.observed_positive
        or manifest.gold_label.unit is GoldLabelUnit.RETRIEVAL_CASE
    ):
        raise EvaluationInputError(
            "runtime retrieval challenges are not eligible for static mutation evaluation"
        )

    if semantic_store_digest(base_store) != manifest.base_store_digest:
        raise EvaluationInputError("base store digest does not match the mutation manifest")
    if semantic_store_digest(mutated_store) != manifest.mutated_store_digest:
        raise EvaluationInputError("mutated store digest does not match the mutation manifest")
    if transcript_set_digest(transcripts) != manifest.transcript_digest:
        raise EvaluationInputError("transcript digest does not match the mutation manifest")
    if result.defect_class is not manifest.defect_class:
        raise EvaluationInputError("checker result defect class does not match the manifest")

    mutated_ids = {memory.id for memory in mutated_store.memories}
    expected_arity = 2 if manifest.gold_label.unit is GoldLabelUnit.MEMORY_PAIR else 1
    for finding in result.findings:
        unknown_ids = tuple(sorted(set(finding.memory_ids) - mutated_ids))
        if unknown_ids:
            raise EvaluationInputError(
                "finding references IDs absent from the mutated store: "
                + ", ".join(unknown_ids)
            )
        if len(finding.memory_ids) != expected_arity:
            raise EvaluationInputError(
                f"{manifest.gold_label.unit.value} findings require exactly "
                f"{expected_arity} memory ID(s)"
            )


def evaluate_mutation_trial(
    *,
    base_store: NormalizedStore,
    mutated_store: NormalizedStore,
    manifest: MutationManifest,
    result: CheckerResult,
    transcripts: TranscriptSet | None = None,
) -> MutationTrialEvaluation:
    """Evaluate predictions only after validating one controlled static mutation."""

    _validate_static_trial_inputs(
        base_store=base_store,
        mutated_store=mutated_store,
        manifest=manifest,
        result=result,
        transcripts=transcripts,
    )

    gold_unit = tuple(sorted(manifest.gold_label.memory_ids))
    base_memory_ids = {memory.id for memory in base_store.memories}
    mutation_context_ids = set(manifest.created_memory_ids) | set(
        manifest.modified_memory_ids
    )
    gold_matching: list[str] = []
    verified_clean: list[str] = []
    unknown_natural: list[str] = []
    mutation_context: list[str] = []

    for finding in result.findings:
        if _canonical_evaluation_unit(finding) == gold_unit:
            gold_matching.append(finding.finding_id)
            continue

        finding_memory_ids = set(finding.memory_ids)
        if finding_memory_ids & mutation_context_ids:
            mutation_context.append(finding.finding_id)
        elif finding_memory_ids <= base_memory_ids:
            if manifest.base_store_status is BaseStoreStatus.CURATED_CLEAN:
                verified_clean.append(finding.finding_id)
            else:
                unknown_natural.append(finding.finding_id)
        else:
            raise EvaluationInputError(
                "non-gold finding cannot be reconciled with base or mutation-context IDs"
            )

    return MutationTrialEvaluation(
        mutation_id=manifest.mutation_id,
        defect_class=manifest.defect_class,
        subtype=manifest.subtype,
        gold_unit=manifest.gold_label.unit,
        base_store_status=manifest.base_store_status,
        checker_id=result.checker_id,
        checker_version=result.checker_version,
        injected_positive_detected=bool(gold_matching),
        gold_matching_finding_ids=tuple(gold_matching),
        duplicate_positive_findings=max(0, len(gold_matching) - 1),
        verified_clean_alert_finding_ids=tuple(verified_clean),
        unknown_natural_alert_finding_ids=tuple(unknown_natural),
        mutation_context_alert_finding_ids=tuple(mutation_context),
        total_findings=len(result.findings),
    )


def summarize_mutation_trials(
    trials: tuple[MutationTrialEvaluation, ...],
) -> MutationEvaluationSummary:
    """Aggregate only safe injected-positive and explicitly separated alert counts."""

    if not isinstance(trials, tuple) or not trials:
        raise EvaluationInputError("mutation trial summary requires a nonempty tuple")
    if any(not isinstance(trial, MutationTrialEvaluation) for trial in trials):
        raise EvaluationInputError(
            "mutation trial summary accepts only MutationTrialEvaluation values"
        )

    trial_count = len(trials)
    detected = sum(trial.injected_positive_detected for trial in trials)
    gold_matching_findings = sum(len(trial.gold_matching_finding_ids) for trial in trials)
    return MutationEvaluationSummary(
        trials=trial_count,
        injected_positives=trial_count,
        injected_positives_detected=detected,
        injected_positives_missed=trial_count - detected,
        injected_positive_recall=detected / trial_count,
        gold_matching_findings=gold_matching_findings,
        duplicate_positive_findings=sum(
            trial.duplicate_positive_findings for trial in trials
        ),
        verified_clean_alerts=sum(
            len(trial.verified_clean_alert_finding_ids) for trial in trials
        ),
        unknown_natural_alerts=sum(
            len(trial.unknown_natural_alert_finding_ids) for trial in trials
        ),
        mutation_context_alerts=sum(
            len(trial.mutation_context_alert_finding_ids) for trial in trials
        ),
        total_findings=sum(trial.total_findings for trial in trials),
    )
