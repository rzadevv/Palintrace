from __future__ import annotations

from pathlib import Path

import pytest

from palintrace.audit import (
    AuditReport,
    SkippedChecker,
    SkipReason,
    run_aggregate_audit,
)
from palintrace.checker_requirements import (
    _CHECKER_REQUIREMENTS,
    PUBLIC_CHECKER_IDS,
    _CheckerRequirement,
)
from palintrace.checkers import (
    CheckerError,
    CheckerResult,
    PrincipalBoundaryRule,
    RedundancyBloatChecker,
    ScopeDimension,
    ScopeIsolationPolicy,
)
from palintrace.models import (
    MemoryScope,
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    Transcript,
    TranscriptSet,
    TranscriptTurn,
)
from palintrace.semantics import (
    SemanticJudgeError,
    SemanticJudgment,
    SemanticRelation,
)


class _FakeJudge:
    judge_id = "test-nli:aggregate"
    judge_version = "revision-1"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        self.calls.append((premise, hypothesis))
        if self.error is not None:
            raise self.error
        return SemanticJudgment(relation=SemanticRelation.ENTAILMENT, score=1.0)


def _store() -> NormalizedStore:
    return NormalizedStore(
        adapter="test",
        memories=(
            NormalizedMemory(
                id="m1",
                content="User prefers Python.",
                source_refs=(SourceRef(transcript_id="t1", turn_idx=0),),
                provenance_status=ProvenanceStatus.DECLARED,
                scope=MemoryScope(user_id="user-a"),
                active=True,
            ),
        ),
    )


def _transcripts() -> TranscriptSet:
    return TranscriptSet(
        transcripts=(
            Transcript(
                id="t1",
                turns=(
                    TranscriptTurn(index=0, role="user", content="User prefers Python."),
                ),
            ),
        )
    )


def _policy() -> ScopeIsolationPolicy:
    return ScopeIsolationPolicy(
        rules=(
            PrincipalBoundaryRule(
                dimension=ScopeDimension.USER_ID,
                authoritative_source_principal="user-a",
                prohibited_destination_principals=("user-b",),
            ),
        )
    )


def _ids(items: tuple[CheckerResult, ...] | tuple[SkippedChecker, ...]) -> tuple[str, ...]:
    return tuple(item.checker_id for item in items)


def _skip_reasons(report: AuditReport, checker_id: str) -> tuple[SkipReason, ...]:
    return next(item.reasons for item in report.skipped if item.checker_id == checker_id)


def test_requirement_metadata_is_exact_and_immutable() -> None:
    assert PUBLIC_CHECKER_IDS == (
        "orphaned_provenance",
        "redundancy_bloat",
        "stale_active",
        "privacy_scope_violation",
        "unsupported_claim",
    )
    assert dict(_CHECKER_REQUIREMENTS) == {
        "orphaned_provenance": _CheckerRequirement(requires_transcripts=True),
        "redundancy_bloat": _CheckerRequirement(),
        "stale_active": _CheckerRequirement(),
        "privacy_scope_violation": _CheckerRequirement(requires_scope_policy=True),
        "unsupported_claim": _CheckerRequirement(
            requires_transcripts=True,
            requires_semantic_judge=True,
        ),
    }
    with pytest.raises(TypeError):
        _CHECKER_REQUIREMENTS["other"] = _CheckerRequirement()  # type: ignore[index]


def test_all_dependencies_execute_all_checkers_in_canonical_order() -> None:
    judge = _FakeJudge()

    report = run_aggregate_audit(
        _store(),
        transcripts=_transcripts(),
        scope_policy=_policy(),
        semantic_judge=judge,
    )

    assert _ids(report.results) == PUBLIC_CHECKER_IDS
    assert report.skipped == ()
    assert judge.calls == [("User prefers Python.", "User prefers Python.")]
    assert all(isinstance(result, CheckerResult) for result in report.results)


def test_missing_transcripts_skips_only_transcript_dependent_checkers() -> None:
    report = run_aggregate_audit(
        _store(),
        scope_policy=_policy(),
        semantic_judge=_FakeJudge(),
    )

    assert _ids(report.results) == (
        "redundancy_bloat",
        "stale_active",
        "privacy_scope_violation",
    )
    assert _ids(report.skipped) == ("orphaned_provenance", "unsupported_claim")
    assert _skip_reasons(report, "orphaned_provenance") == (
        SkipReason.MISSING_TRANSCRIPTS,
    )
    assert _skip_reasons(report, "unsupported_claim") == (SkipReason.MISSING_TRANSCRIPTS,)


def test_missing_policy_skips_only_privacy_checker() -> None:
    report = run_aggregate_audit(
        _store(),
        transcripts=_transcripts(),
        semantic_judge=_FakeJudge(),
    )

    assert _ids(report.results) == tuple(
        checker_id for checker_id in PUBLIC_CHECKER_IDS if checker_id != "privacy_scope_violation"
    )
    assert _ids(report.skipped) == ("privacy_scope_violation",)
    assert _skip_reasons(report, "privacy_scope_violation") == (
        SkipReason.MISSING_SCOPE_POLICY,
    )


def test_missing_semantic_judge_skips_only_unsupported_claim() -> None:
    report = run_aggregate_audit(
        _store(),
        transcripts=_transcripts(),
        scope_policy=_policy(),
    )

    assert _ids(report.results) == PUBLIC_CHECKER_IDS[:-1]
    assert _ids(report.skipped) == ("unsupported_claim",)
    assert _skip_reasons(report, "unsupported_claim") == (
        SkipReason.MISSING_SEMANTIC_CONFIGURATION,
    )


def test_multiple_missing_requirements_use_canonical_reason_order() -> None:
    report = run_aggregate_audit(_store(), scope_policy=_policy())

    assert _skip_reasons(report, "unsupported_claim") == (
        SkipReason.MISSING_TRANSCRIPTS,
        SkipReason.MISSING_SEMANTIC_CONFIGURATION,
    )


def test_no_optional_dependencies_executes_only_structural_checkers() -> None:
    report = run_aggregate_audit(_store())

    assert _ids(report.results) == ("redundancy_bloat", "stale_active")
    assert _ids(report.skipped) == (
        "orphaned_provenance",
        "privacy_scope_violation",
        "unsupported_claim",
    )
    assert _skip_reasons(report, "orphaned_provenance") == (
        SkipReason.MISSING_TRANSCRIPTS,
    )
    assert _skip_reasons(report, "privacy_scope_violation") == (
        SkipReason.MISSING_SCOPE_POLICY,
    )
    assert _skip_reasons(report, "unsupported_claim") == (
        SkipReason.MISSING_TRANSCRIPTS,
        SkipReason.MISSING_SEMANTIC_CONFIGURATION,
    )


def test_checker_exceptions_propagate_instead_of_becoming_skips() -> None:
    judge = _FakeJudge(SemanticJudgeError("semantic backend failed"))

    with pytest.raises(CheckerError, match="semantic judgment failed"):
        run_aggregate_audit(
            _store(),
            transcripts=_transcripts(),
            scope_policy=_policy(),
            semantic_judge=judge,
        )


def test_completed_result_content_is_not_rewritten() -> None:
    expected = RedundancyBloatChecker().check(_store())
    report = run_aggregate_audit(_store())
    actual = next(result for result in report.results if result.checker_id == "redundancy_bloat")

    assert actual == expected
    assert actual.to_json() == expected.to_json()


def test_aggregate_module_does_not_cross_architecture_boundaries() -> None:
    source = (Path(__file__).parents[1] / "src/palintrace/audit.py").read_text(encoding="utf-8")

    assert "palintrace.evaluation" not in source
    assert "palintrace.mutations" not in source
    assert "palintrace.retrieval" not in source
    assert ".raw" not in source
