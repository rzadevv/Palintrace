from __future__ import annotations

import pytest
from pydantic import ValidationError

from memlint.checkers import CheckerResult, CheckerStats, EvidenceItem, Finding
from memlint.evaluation import (
    BenchmarkCaseKind,
    BenchmarkFixture,
    BenchmarkSplit,
    CleanControlBenchmarkCase,
    CleanControlEvaluation,
    EvaluationInputError,
    MutationTrialEvaluation,
    StaticDefectBenchmarkSummary,
    evaluate_clean_control,
    summarize_static_defect_benchmark,
)
from memlint.models import NormalizedMemory, NormalizedStore
from memlint.mutations import BaseStoreStatus, GoldLabelUnit
from memlint.taxonomy import DefectClass

IMPLEMENTED = (
    DefectClass.ORPHANED_PROVENANCE,
    DefectClass.REDUNDANCY_BLOAT,
    DefectClass.STALE_ACTIVE,
    DefectClass.PRIVACY_SCOPE_VIOLATION,
    DefectClass.UNSUPPORTED_CLAIM,
)


def _fixture(*, status: BaseStoreStatus = BaseStoreStatus.CURATED_CLEAN) -> BenchmarkFixture:
    return BenchmarkFixture(
        fixture_id="TOY",
        store_path="tests/toy-store.json",
        base_store_status=status,
    )


def _case(
    defect_class: DefectClass = DefectClass.ORPHANED_PROVENANCE,
    *,
    split: BenchmarkSplit = BenchmarkSplit.HELD_OUT,
) -> CleanControlBenchmarkCase:
    return CleanControlBenchmarkCase(
        case_id=f"CC-{defect_class.value}",
        kind=BenchmarkCaseKind.CLEAN_CONTROL,
        split=split,
        defect_class=defect_class,
        base_fixture_id="TOY",
        scope_policy_fixture_id=(
            "TOY" if defect_class is DefectClass.PRIVACY_SCOPE_VIOLATION else None
        ),
    )


def _store() -> NormalizedStore:
    return NormalizedStore(
        schema_version="0.1",
        adapter="toy",
        memories=(
            NormalizedMemory(id="m1", content="Alpha fact."),
            NormalizedMemory(id="m2", content="Beta fact."),
        ),
    )


def _finding(
    finding_id: str,
    defect_class: DefectClass,
    memory_id: str = "m1",
) -> Finding:
    return Finding(
        finding_id=finding_id,
        defect_class=defect_class,
        memory_ids=(memory_id,),
        confidence=1.0,
        evidence=(
            EvidenceItem(
                kind="toy",
                message="Toy clean-control finding.",
                data={"reference": finding_id},
            ),
        ),
    )


def _result(
    defect_class: DefectClass,
    findings: tuple[Finding, ...] = (),
) -> CheckerResult:
    return CheckerResult(
        checker_id=defect_class.value,
        checker_version="1.0",
        defect_class=defect_class,
        findings=findings,
        stats=CheckerStats(
            memories_scanned=2,
            findings_emitted=len(findings),
        ),
    )


def _trial(
    defect_class: DefectClass,
    *,
    detected: bool = True,
    verified_ids: tuple[str, ...] = (),
    unknown_ids: tuple[str, ...] = (),
    context_ids: tuple[str, ...] = (),
    duplicate_positive_findings: int = 0,
) -> MutationTrialEvaluation:
    gold_ids = tuple(f"gold-{index}" for index in range(duplicate_positive_findings + 1))
    if not detected:
        gold_ids = ()
        duplicate_positive_findings = 0
    return MutationTrialEvaluation(
        mutation_id=f"mutation-{defect_class.value}",
        defect_class=defect_class,
        subtype="toy",
        gold_unit=(
            GoldLabelUnit.MEMORY_PAIR
            if defect_class is DefectClass.REDUNDANCY_BLOAT
            else GoldLabelUnit.MEMORY
        ),
        base_store_status=BaseStoreStatus.CURATED_CLEAN,
        checker_id=defect_class.value,
        checker_version="1.0",
        injected_positive_detected=detected,
        gold_matching_finding_ids=gold_ids,
        duplicate_positive_findings=duplicate_positive_findings,
        verified_clean_alert_finding_ids=verified_ids,
        unknown_natural_alert_finding_ids=unknown_ids,
        mutation_context_alert_finding_ids=context_ids,
        total_findings=(
            len(gold_ids) + len(verified_ids) + len(unknown_ids) + len(context_ids)
        ),
    )


def _control(
    defect_class: DefectClass,
    *,
    case_suffix: str = "1",
    finding_ids: tuple[str, ...] = (),
) -> CleanControlEvaluation:
    return CleanControlEvaluation(
        case_id=f"CC-{defect_class.value}-{case_suffix}",
        defect_class=defect_class,
        base_fixture_id="TOY",
        checker_id=defect_class.value,
        checker_version="1.0",
        alert_present=bool(finding_ids),
        finding_ids=finding_ids,
        findings_emitted=len(finding_ids),
    )


def test_clean_control_zero_one_and_multiple_findings() -> None:
    case = _case()
    fixture = _fixture()
    store = _store()

    empty = evaluate_clean_control(
        case=case,
        fixture=fixture,
        store=store,
        result=_result(case.defect_class),
    )
    assert empty.alert_present is False
    assert empty.finding_ids == ()
    assert empty.findings_emitted == 0

    findings = (
        _finding("finding-b", case.defect_class, "m2"),
        _finding("finding-a", case.defect_class, "m1"),
    )
    alert = evaluate_clean_control(
        case=case,
        fixture=fixture,
        store=store,
        result=_result(case.defect_class, findings),
    )
    assert alert.alert_present is True
    assert alert.finding_ids == ("finding-a", "finding-b")
    assert alert.findings_emitted == 2
    assert alert.to_json() == CleanControlEvaluation.model_validate_json(
        alert.to_json()
    ).to_json()


def test_clean_control_requires_heldout_curated_matching_inputs() -> None:
    store = _store()
    result = _result(DefectClass.ORPHANED_PROVENANCE)
    with pytest.raises(EvaluationInputError, match="held-out"):
        evaluate_clean_control(
            case=_case(split=BenchmarkSplit.DEVELOPMENT),
            fixture=_fixture(),
            store=store,
            result=result,
        )
    with pytest.raises(EvaluationInputError, match="curated-clean"):
        evaluate_clean_control(
            case=_case(),
            fixture=_fixture(status=BaseStoreStatus.UNKNOWN),
            store=store,
            result=result,
        )
    mismatched_fixture = _fixture().model_copy(update={"fixture_id": "OTHER"})
    with pytest.raises(EvaluationInputError, match="does not match"):
        evaluate_clean_control(
            case=_case(),
            fixture=mismatched_fixture,
            store=store,
            result=result,
        )
    with pytest.raises(EvaluationInputError, match="defect class"):
        evaluate_clean_control(
            case=_case(),
            fixture=_fixture(),
            store=store,
            result=_result(DefectClass.STALE_ACTIVE),
        )


def test_clean_control_rejects_unknown_finding_memory_id() -> None:
    defect = DefectClass.ORPHANED_PROVENANCE
    with pytest.raises(EvaluationInputError, match="absent"):
        evaluate_clean_control(
            case=_case(defect),
            fixture=_fixture(),
            store=_store(),
            result=_result(defect, (_finding("outside", defect, "unknown"),)),
        )


def test_clean_control_model_is_self_validating() -> None:
    valid = _control(DefectClass.STALE_ACTIVE, finding_ids=("f1",))
    with pytest.raises(ValidationError, match="alert_present"):
        CleanControlEvaluation.model_validate(
            {**valid.model_dump(), "alert_present": False}
        )
    with pytest.raises(ValidationError, match="findings_emitted"):
        CleanControlEvaluation.model_validate(
            {**valid.model_dump(), "findings_emitted": 2}
        )


def test_per_defect_summary_uses_case_level_clean_control_rate() -> None:
    trials = tuple(
        _trial(
            defect,
            detected=defect is not DefectClass.STALE_ACTIVE,
            verified_ids=("verified-from-mutation",)
            if defect is DefectClass.ORPHANED_PROVENANCE
            else (),
            unknown_ids=("unknown",)
            if defect is DefectClass.ORPHANED_PROVENANCE
            else (),
            context_ids=("context",)
            if defect is DefectClass.ORPHANED_PROVENANCE
            else (),
            duplicate_positive_findings=1
            if defect is DefectClass.ORPHANED_PROVENANCE
            else 0,
        )
        for defect in IMPLEMENTED
    )
    controls = tuple(
        [
            _control(
                DefectClass.ORPHANED_PROVENANCE,
                case_suffix="1",
                finding_ids=("clean-a", "clean-b"),
            ),
            _control(DefectClass.ORPHANED_PROVENANCE, case_suffix="2"),
        ]
        + [
            _control(defect)
            for defect in IMPLEMENTED
            if defect is not DefectClass.ORPHANED_PROVENANCE
        ]
    )

    summaries = summarize_static_defect_benchmark(
        mutation_trials=trials,
        clean_controls=controls,
    )
    assert tuple(item.defect_class for item in summaries) == tuple(
        sorted(IMPLEMENTED, key=lambda item: item.value)
    )
    orphaned = next(
        item
        for item in summaries
        if item.defect_class is DefectClass.ORPHANED_PROVENANCE
    )
    assert orphaned.positive_trials == 1
    assert orphaned.injected_positive_recall == 1.0
    assert orphaned.clean_controls == 2
    assert orphaned.clean_controls_with_alert == 1
    assert orphaned.clean_control_alert_rate == 0.5
    assert orphaned.verified_clean_alerts == 3
    assert orphaned.unknown_natural_alerts == 1
    assert orphaned.mutation_context_alerts == 1
    assert orphaned.duplicate_positive_findings == 1


def test_summary_requires_exact_five_classes_and_has_no_misleading_metrics() -> None:
    trials = tuple(_trial(defect) for defect in IMPLEMENTED)
    controls = tuple(_control(defect) for defect in IMPLEMENTED)
    with pytest.raises(EvaluationInputError, match="all five"):
        summarize_static_defect_benchmark(
            mutation_trials=trials[:-1],
            clean_controls=controls,
        )
    forbidden = {
        "precision",
        "f1",
        "accuracy",
        "specificity",
        "false_positive_rate",
    }
    assert forbidden.isdisjoint(StaticDefectBenchmarkSummary.model_fields)
