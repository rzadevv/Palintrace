from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from memlint.checkers import (
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
    OrphanedProvenanceChecker,
    PrincipalBoundaryRule,
    PrivacyScopeViolationChecker,
    RedundancyBloatChecker,
    ScopeDimension,
    ScopeIsolationPolicy,
    StaleActiveChecker,
    UnsupportedClaimChecker,
)
from memlint.evaluation import (
    EvaluationInputError,
    MutationEvaluationSummary,
    MutationScientificLabel,
    MutationTrialEvaluation,
    evaluate_mutation_trial,
    summarize_mutation_trials,
)
from memlint.models import NormalizedMemory, NormalizedStore, TranscriptSet
from memlint.mutations import (
    BaseStoreStatus,
    DistractorFamily,
    GoldLabelUnit,
    MutationManifest,
    MutationRequest,
    MutationResult,
    mutate,
)
from memlint.semantics import (
    SemanticJudgment,
    SemanticRelation,
    SemanticUsage,
)
from memlint.serialization import load_store, load_transcripts
from memlint.taxonomy import DefectClass


@pytest.fixture
def base_store() -> NormalizedStore:
    return load_store("examples/mutation-store.json")


@pytest.fixture
def transcripts() -> TranscriptSet:
    return load_transcripts("examples/mutation-transcripts.json")


def _finding(
    finding_id: str,
    defect_class: DefectClass,
    memory_ids: Sequence[str],
    *,
    confidence: float = 1.0,
) -> Finding:
    return Finding(
        finding_id=finding_id,
        defect_class=defect_class,
        memory_ids=tuple(memory_ids),
        confidence=confidence,
        evidence=(
            EvidenceItem(
                kind="evaluation-test-prediction",
                message="Synthetic checker output for accounting contract tests.",
                data={"prediction_reference": finding_id},
            ),
        ),
    )


def _result(
    defect_class: DefectClass,
    findings: Sequence[Finding] = (),
    *,
    checker_id: str = "alternative-candidate",
    checker_version: str = "revision-test",
) -> CheckerResult:
    finding_tuple = tuple(findings)
    return CheckerResult(
        checker_id=checker_id,
        checker_version=checker_version,
        defect_class=defect_class,
        findings=finding_tuple,
        stats=CheckerStats(
            memories_scanned=0,
            findings_emitted=len(finding_tuple),
        ),
    )


def _orphan_mutation(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
    *,
    status: BaseStoreStatus = BaseStoreStatus.UNKNOWN,
) -> MutationResult:
    return mutate(
        base_store,
        MutationRequest(
            defect_class=DefectClass.ORPHANED_PROVENANCE,
            subtype="missing_transcript",
            target_memory_id="preference-python",
            base_store_status=status,
        ),
        transcripts,
    )


def _redundancy_mutation(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
    *,
    status: BaseStoreStatus = BaseStoreStatus.UNKNOWN,
) -> MutationResult:
    return mutate(
        base_store,
        MutationRequest(
            defect_class=DefectClass.REDUNDANCY_BLOAT,
            target_memory_id="preference-python",
            base_store_status=status,
        ),
        transcripts,
    )


def _evaluate(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
    mutation_result: MutationResult,
    checker_result: CheckerResult,
) -> MutationTrialEvaluation:
    return evaluate_mutation_trial(
        base_store=base_store,
        mutated_store=mutation_result.mutated_store,
        manifest=mutation_result.manifest,
        result=checker_result,
        transcripts=transcripts,
    )


def _store_with_extra(store: NormalizedStore, memory_id: str) -> NormalizedStore:
    return NormalizedStore(
        schema_version=store.schema_version,
        adapter=store.adapter,
        exported_at=store.exported_at,
        memories=(*store.memories, NormalizedMemory(id=memory_id, content="Extra memory.")),
    )


def test_exact_three_scientific_labels_exclude_mutation_context() -> None:
    assert tuple(MutationScientificLabel) == (
        MutationScientificLabel.INJECTED_POSITIVE,
        MutationScientificLabel.VERIFIED_CLEAN,
        MutationScientificLabel.UNKNOWN_NATURAL,
    )
    assert [label.value for label in MutationScientificLabel] == [
        "injected_positive",
        "verified_clean",
        "unknown_natural",
    ]
    assert "MUTATION_CONTEXT" not in MutationScientificLabel.__members__


def test_static_evaluator_validates_both_store_digests_and_transcript_digest(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    mutation_result = _orphan_mutation(base_store, transcripts)
    checker_result = _result(DefectClass.ORPHANED_PROVENANCE)

    with pytest.raises(EvaluationInputError, match="base store digest"):
        evaluate_mutation_trial(
            base_store=_store_with_extra(base_store, "unexpected-base"),
            mutated_store=mutation_result.mutated_store,
            manifest=mutation_result.manifest,
            result=checker_result,
            transcripts=transcripts,
        )
    with pytest.raises(EvaluationInputError, match="mutated store digest"):
        evaluate_mutation_trial(
            base_store=base_store,
            mutated_store=_store_with_extra(
                mutation_result.mutated_store,
                "unexpected-mutated",
            ),
            manifest=mutation_result.manifest,
            result=checker_result,
            transcripts=transcripts,
        )
    with pytest.raises(EvaluationInputError, match="transcript digest"):
        evaluate_mutation_trial(
            base_store=base_store,
            mutated_store=mutation_result.mutated_store,
            manifest=mutation_result.manifest,
            result=checker_result,
            transcripts=None,
        )


def test_retrieval_challenge_is_rejected_by_static_evaluator(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    retrieval_mutation = mutate(
        base_store,
        MutationRequest(
            defect_class=DefectClass.RETRIEVAL_SHADOWING,
            target_memory_id="editor-neovim",
            query="Which editor does the user prefer?",
            distractor_family=DistractorFamily.EDITOR,
        ),
        transcripts,
    )
    manifest = retrieval_mutation.manifest
    assert manifest.requires_runtime_validation is True
    assert manifest.gold_label.observed_positive is False
    assert manifest.gold_label.unit is GoldLabelUnit.RETRIEVAL_CASE

    with pytest.raises(EvaluationInputError, match="not eligible for static"):
        _evaluate(
            base_store,
            transcripts,
            retrieval_mutation,
            _result(DefectClass.RETRIEVAL_SHADOWING),
        )


def test_result_defect_class_must_match_but_checker_id_may_differ(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    mutation_result = _orphan_mutation(base_store, transcripts)
    gold_ids = mutation_result.manifest.gold_label.memory_ids

    with pytest.raises(EvaluationInputError, match="defect class"):
        _evaluate(
            base_store,
            transcripts,
            mutation_result,
            _result(DefectClass.STALE_ACTIVE),
        )

    trial = _evaluate(
        base_store,
        transcripts,
        mutation_result,
        _result(
            DefectClass.ORPHANED_PROVENANCE,
            (_finding("gold", DefectClass.ORPHANED_PROVENANCE, gold_ids),),
            checker_id="experimental-provenance-baseline",
        ),
    )
    assert trial.checker_id == "experimental-provenance-baseline"
    assert trial.injected_positive_detected is True


def test_every_finding_id_must_exist_in_mutated_snapshot(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    mutation_result = _orphan_mutation(base_store, transcripts)
    result = _result(
        DefectClass.ORPHANED_PROVENANCE,
        (_finding("absent", DefectClass.ORPHANED_PROVENANCE, ("not-in-store",)),),
    )

    with pytest.raises(EvaluationInputError, match="absent from the mutated store"):
        _evaluate(base_store, transcripts, mutation_result, result)


def test_memory_and_memory_pair_finding_arities_are_enforced(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    memory_mutation = _orphan_mutation(base_store, transcripts)
    pair_mutation = _redundancy_mutation(base_store, transcripts)

    with pytest.raises(EvaluationInputError, match="memory findings require exactly 1"):
        _evaluate(
            base_store,
            transcripts,
            memory_mutation,
            _result(
                DefectClass.ORPHANED_PROVENANCE,
                (
                    _finding(
                        "wrong-arity-memory",
                        DefectClass.ORPHANED_PROVENANCE,
                        ("employment-aster", "editor-neovim"),
                    ),
                ),
            ),
        )
    with pytest.raises(EvaluationInputError, match="memory_pair findings require exactly 2"):
        _evaluate(
            base_store,
            transcripts,
            pair_mutation,
            _result(
                DefectClass.REDUNDANCY_BLOAT,
                (
                    _finding(
                        "wrong-arity-pair",
                        DefectClass.REDUNDANCY_BLOAT,
                        ("employment-aster",),
                    ),
                ),
            ),
        )


def test_pair_matching_is_order_independent_and_exact(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    mutation_result = _redundancy_mutation(base_store, transcripts)
    gold_ids = mutation_result.manifest.gold_label.memory_ids
    reversed_gold = tuple(reversed(gold_ids))
    trial = _evaluate(
        base_store,
        transcripts,
        mutation_result,
        _result(
            DefectClass.REDUNDANCY_BLOAT,
            (_finding("reversed-gold", DefectClass.REDUNDANCY_BLOAT, reversed_gold),),
        ),
    )

    assert trial.injected_positive_detected is True
    assert trial.gold_matching_finding_ids == ("reversed-gold",)


def test_wrong_pair_gets_no_partial_credit_and_uses_unknown_natural_bucket(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    mutation_result = _redundancy_mutation(base_store, transcripts)
    wrong_base_pair = ("employment-aster", "preference-python")
    assert set(wrong_base_pair) & set(mutation_result.manifest.gold_label.memory_ids)

    trial = _evaluate(
        base_store,
        transcripts,
        mutation_result,
        _result(
            DefectClass.REDUNDANCY_BLOAT,
            (_finding("partial-pair", DefectClass.REDUNDANCY_BLOAT, wrong_base_pair),),
        ),
    )

    assert trial.injected_positive_detected is False
    assert trial.gold_matching_finding_ids == ()
    assert trial.unknown_natural_alert_finding_ids == ("partial-pair",)


def test_wrong_pair_with_created_id_is_mutation_context_not_partial_credit(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    mutation_result = _redundancy_mutation(
        base_store,
        transcripts,
        status=BaseStoreStatus.CURATED_CLEAN,
    )
    created_id = mutation_result.manifest.created_memory_ids[0]
    wrong_pair = (created_id, "employment-aster")

    trial = _evaluate(
        base_store,
        transcripts,
        mutation_result,
        _result(
            DefectClass.REDUNDANCY_BLOAT,
            (_finding("created-context", DefectClass.REDUNDANCY_BLOAT, wrong_pair),),
        ),
    )

    assert trial.injected_positive_detected is False
    assert trial.mutation_context_alert_finding_ids == ("created-context",)
    assert trial.verified_clean_alert_finding_ids == ()
    assert trial.unknown_natural_alert_finding_ids == ()


def test_declared_modified_context_takes_precedence_over_curated_clean(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    mutation_result = _orphan_mutation(
        base_store,
        transcripts,
        status=BaseStoreStatus.CURATED_CLEAN,
    )
    manifest_payload = mutation_result.manifest.model_dump()
    manifest_payload["modified_memory_ids"] = (
        *mutation_result.manifest.modified_memory_ids,
        "editor-neovim",
    )
    manifest = MutationManifest.model_validate(manifest_payload)
    result = _result(
        DefectClass.ORPHANED_PROVENANCE,
        (
            _finding(
                "modified-context",
                DefectClass.ORPHANED_PROVENANCE,
                ("editor-neovim",),
            ),
        ),
    )

    trial = evaluate_mutation_trial(
        base_store=base_store,
        mutated_store=mutation_result.mutated_store,
        manifest=manifest,
        result=result,
        transcripts=transcripts,
    )

    assert trial.mutation_context_alert_finding_ids == ("modified-context",)
    assert trial.verified_clean_alert_finding_ids == ()


@pytest.mark.parametrize(
    ("status", "expected_bucket"),
    [
        (BaseStoreStatus.UNKNOWN, "unknown_natural_alert_finding_ids"),
        (BaseStoreStatus.CURATED_CLEAN, "verified_clean_alert_finding_ids"),
    ],
)
def test_base_only_non_gold_alert_label_depends_only_on_explicit_base_status(
    status: BaseStoreStatus,
    expected_bucket: str,
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    mutation_result = _orphan_mutation(base_store, transcripts, status=status)
    result = _result(
        DefectClass.ORPHANED_PROVENANCE,
        (
            _finding(
                "base-only-alert",
                DefectClass.ORPHANED_PROVENANCE,
                ("editor-neovim",),
            ),
        ),
    )

    trial = _evaluate(base_store, transcripts, mutation_result, result)

    assert getattr(trial, expected_bucket) == ("base-only-alert",)
    other_bucket = (
        trial.verified_clean_alert_finding_ids
        if status is BaseStoreStatus.UNKNOWN
        else trial.unknown_natural_alert_finding_ids
    )
    assert other_bucket == ()


def test_duplicate_gold_findings_count_one_detection_without_false_alerts(
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    mutation_result = _orphan_mutation(base_store, transcripts)
    gold_ids = mutation_result.manifest.gold_label.memory_ids
    result = _result(
        DefectClass.ORPHANED_PROVENANCE,
        tuple(
            _finding(
                finding_id,
                DefectClass.ORPHANED_PROVENANCE,
                gold_ids,
                confidence=confidence,
            )
            for finding_id, confidence in (
                ("gold-a", 0.01),
                ("gold-b", 0.5),
                ("gold-c", 1.0),
            )
        ),
    )

    trial = _evaluate(base_store, transcripts, mutation_result, result)
    summary = summarize_mutation_trials((trial,))

    assert trial.injected_positive_detected is True
    assert trial.gold_matching_finding_ids == ("gold-a", "gold-b", "gold-c")
    assert trial.duplicate_positive_findings == 2
    assert trial.verified_clean_alert_finding_ids == ()
    assert trial.unknown_natural_alert_finding_ids == ()
    assert trial.mutation_context_alert_finding_ids == ()
    assert summary.injected_positives == 1
    assert summary.injected_positives_detected == 1
    assert summary.injected_positive_recall == 1.0
    assert summary.gold_matching_findings == 3
    assert summary.duplicate_positive_findings == 2


@pytest.mark.parametrize("confidence", [0.0, 0.17, 1.0])
def test_confidence_does_not_affect_exact_gold_matching(
    confidence: float,
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    mutation_result = _orphan_mutation(base_store, transcripts)
    trial = _evaluate(
        base_store,
        transcripts,
        mutation_result,
        _result(
            DefectClass.ORPHANED_PROVENANCE,
            (
                _finding(
                    "gold",
                    DefectClass.ORPHANED_PROVENANCE,
                    mutation_result.manifest.gold_label.memory_ids,
                    confidence=confidence,
                ),
            ),
        ),
    )
    assert trial.injected_positive_detected is True


def _trial_model(
    *,
    mutation_id: str,
    detected: bool,
    gold_ids: tuple[str, ...] = (),
    verified_ids: tuple[str, ...] = (),
    unknown_ids: tuple[str, ...] = (),
    context_ids: tuple[str, ...] = (),
) -> MutationTrialEvaluation:
    return MutationTrialEvaluation(
        mutation_id=mutation_id,
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        subtype="test-subtype",
        gold_unit=GoldLabelUnit.MEMORY,
        base_store_status=BaseStoreStatus.UNKNOWN,
        checker_id="test-checker",
        checker_version="1",
        injected_positive_detected=detected,
        gold_matching_finding_ids=gold_ids,
        duplicate_positive_findings=max(0, len(gold_ids) - 1),
        verified_clean_alert_finding_ids=verified_ids,
        unknown_natural_alert_finding_ids=unknown_ids,
        mutation_context_alert_finding_ids=context_ids,
        total_findings=(
            len(gold_ids) + len(verified_ids) + len(unknown_ids) + len(context_ids)
        ),
    )


def test_trial_model_is_self_validating_and_canonical() -> None:
    valid = _trial_model(
        mutation_id="mutation",
        detected=True,
        gold_ids=("gold-b", "gold-a"),
        unknown_ids=("unknown-b", "unknown-a"),
    )
    assert valid.gold_matching_finding_ids == ("gold-a", "gold-b")
    assert valid.unknown_natural_alert_finding_ids == ("unknown-a", "unknown-b")

    invalid_payloads = []
    total_payload = valid.model_dump()
    total_payload["total_findings"] = 99
    invalid_payloads.append(total_payload)
    overlap_payload = valid.model_dump()
    overlap_payload["verified_clean_alert_finding_ids"] = ("gold-a",)
    overlap_payload["total_findings"] = 5
    invalid_payloads.append(overlap_payload)
    detected_payload = valid.model_dump()
    detected_payload["injected_positive_detected"] = False
    invalid_payloads.append(detected_payload)
    duplicate_payload = valid.model_dump()
    duplicate_payload["duplicate_positive_findings"] = 0
    invalid_payloads.append(duplicate_payload)

    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            MutationTrialEvaluation.model_validate(payload)


def test_trial_serialization_is_deterministic_and_excludes_sensitive_payloads() -> None:
    trial = _trial_model(
        mutation_id="mutation",
        detected=True,
        gold_ids=("gold",),
        unknown_ids=("unknown",),
    )

    first = trial.to_json()
    second = trial.to_json()
    assert first == second
    assert MutationTrialEvaluation.model_validate_json(first) == trial
    assert set(MutationTrialEvaluation.model_fields).isdisjoint(
        {
            "memory_content",
            "transcript_text",
            "raw",
            "parameters",
            "replace_from",
            "replace_to",
            "distractor_ids",
            "query",
        }
    )


def test_mutation_summary_counts_safe_buckets_and_recall_without_precision_metrics() -> None:
    trials = (
        _trial_model(
            mutation_id="m1",
            detected=True,
            gold_ids=("gold-1",),
            unknown_ids=("unknown-1",),
        ),
        _trial_model(
            mutation_id="m2",
            detected=False,
            verified_ids=("verified-1",),
        ),
        _trial_model(
            mutation_id="m3",
            detected=True,
            gold_ids=("gold-2a", "gold-2b"),
            context_ids=("context-1",),
        ),
    )

    summary = summarize_mutation_trials(trials)

    assert summary.trials == 3
    assert summary.injected_positives == 3
    assert summary.injected_positives_detected == 2
    assert summary.injected_positives_missed == 1
    assert summary.injected_positive_recall == 2 / 3
    assert summary.gold_matching_findings == 3
    assert summary.duplicate_positive_findings == 1
    assert summary.verified_clean_alerts == 1
    assert summary.unknown_natural_alerts == 1
    assert summary.mutation_context_alerts == 1
    assert summary.total_findings == 6
    assert set(MutationEvaluationSummary.model_fields).isdisjoint(
        {"precision", "f1", "accuracy", "specificity", "false_positive_rate"}
    )
    assert summary.to_json() == summarize_mutation_trials(trials).to_json()
    assert MutationEvaluationSummary.model_validate_json(summary.to_json()) == summary


def test_mutation_summary_requires_nonempty_typed_trials() -> None:
    with pytest.raises(EvaluationInputError, match="nonempty"):
        summarize_mutation_trials(())
    with pytest.raises(EvaluationInputError, match="only MutationTrialEvaluation"):
        summarize_mutation_trials((object(),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_recall",
    [float("nan"), float("inf"), -0.01, 1.01, 1, True, "0.5"],
)
def test_injected_positive_recall_is_a_strict_finite_unit_ratio(
    invalid_recall: object,
) -> None:
    payload = summarize_mutation_trials(
        (_trial_model(mutation_id="m1", detected=True, gold_ids=("gold",)),)
    ).model_dump()
    payload["injected_positive_recall"] = invalid_recall

    with pytest.raises(ValidationError):
        MutationEvaluationSummary.model_validate(payload)


class _OfflineFakeJudge:
    judge_id = "evaluation-test-judge"
    judge_version = "1"

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        del premise
        relation = (
            SemanticRelation.NEUTRAL
            if hypothesis == "User prefers Rust."
            else SemanticRelation.ENTAILMENT
        )
        return SemanticJudgment(
            relation=relation,
            score=0.8,
            usage=SemanticUsage(model_calls=1, input_tokens=1, output_tokens=0),
        )


@pytest.mark.parametrize(
    "defect_class",
    [
        DefectClass.ORPHANED_PROVENANCE,
        DefectClass.REDUNDANCY_BLOAT,
        DefectClass.STALE_ACTIVE,
        DefectClass.PRIVACY_SCOPE_VIOLATION,
        DefectClass.UNSUPPORTED_CLAIM,
    ],
)
def test_implemented_part_two_checker_results_are_compatible_with_evaluator(
    defect_class: DefectClass,
    base_store: NormalizedStore,
    transcripts: TranscriptSet,
) -> None:
    if defect_class is DefectClass.ORPHANED_PROVENANCE:
        request = MutationRequest(
            defect_class=defect_class,
            subtype="missing_transcript",
            target_memory_id="preference-python",
        )
        mutation_result = mutate(base_store, request, transcripts)
        checker_result = OrphanedProvenanceChecker().check(
            mutation_result.mutated_store,
            transcripts=transcripts,
        )
    elif defect_class is DefectClass.REDUNDANCY_BLOAT:
        request = MutationRequest(
            defect_class=defect_class,
            target_memory_id="preference-python",
        )
        mutation_result = mutate(base_store, request, transcripts)
        checker_result = RedundancyBloatChecker().check(mutation_result.mutated_store)
    elif defect_class is DefectClass.STALE_ACTIVE:
        request = MutationRequest(
            defect_class=defect_class,
            target_memory_id="employment-aster",
            replace_from="Aster Labs",
            replace_to="Beacon Works",
        )
        mutation_result = mutate(base_store, request, transcripts)
        checker_result = StaleActiveChecker().check(mutation_result.mutated_store)
    elif defect_class is DefectClass.PRIVACY_SCOPE_VIOLATION:
        request = MutationRequest(
            defect_class=defect_class,
            subtype="cross_user_copy",
            target_memory_id="preference-python",
            destination_user_id="user-b",
        )
        mutation_result = mutate(base_store, request, transcripts)
        policy = ScopeIsolationPolicy(
            rules=(
                PrincipalBoundaryRule(
                    dimension=ScopeDimension.USER_ID,
                    authoritative_source_principal="user-a",
                    prohibited_destination_principals=("user-b",),
                ),
            )
        )
        checker_result = PrivacyScopeViolationChecker(policy).check(
            mutation_result.mutated_store
        )
    else:
        request = MutationRequest(
            defect_class=defect_class,
            target_memory_id="preference-python",
            replace_from="Python",
            replace_to="Rust",
        )
        mutation_result = mutate(base_store, request, transcripts)
        checker_result = UnsupportedClaimChecker(_OfflineFakeJudge()).check(
            mutation_result.mutated_store,
            transcripts=transcripts,
        )

    trial = _evaluate(base_store, transcripts, mutation_result, checker_result)

    assert trial.defect_class is defect_class
    assert trial.injected_positive_detected is True
    assert len(trial.gold_matching_finding_ids) >= 1
