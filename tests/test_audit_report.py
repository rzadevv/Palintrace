from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from palintrace.audit import (
    AUDIT_REPORT_SCHEMA_VERSION,
    AuditReport,
    SkippedChecker,
    SkipReason,
)
from palintrace.checker_requirements import PUBLIC_CHECKER_IDS
from palintrace.checkers import CheckerResult, CheckerStats
from palintrace.taxonomy import DefectClass

_DEFECT_CLASSES = {
    "orphaned_provenance": DefectClass.ORPHANED_PROVENANCE,
    "redundancy_bloat": DefectClass.REDUNDANCY_BLOAT,
    "stale_active": DefectClass.STALE_ACTIVE,
    "privacy_scope_violation": DefectClass.PRIVACY_SCOPE_VIOLATION,
    "unsupported_claim": DefectClass.UNSUPPORTED_CLAIM,
}


def _result(checker_id: str) -> CheckerResult:
    return CheckerResult(
        checker_id=checker_id,
        checker_version="1.0",
        defect_class=_DEFECT_CLASSES[checker_id],
        stats=CheckerStats(memories_scanned=0, findings_emitted=0),
    )


def _custom_result(checker_id: str) -> CheckerResult:
    return CheckerResult(
        checker_id=checker_id,
        checker_version="1.0",
        rule_id="memory.test.unknown",
        rule_version="1.0.0",
        severity="info",
        defect_class=DefectClass.REDUNDANCY_BLOAT,
        stats=CheckerStats(memories_scanned=0, findings_emitted=0),
    )


def _skipped(checker_id: str, *reasons: SkipReason) -> SkippedChecker:
    return SkippedChecker(checker_id=checker_id, reasons=reasons)


def _valid_report() -> AuditReport:
    return AuditReport(
        results=(
            _result("stale_active"),
            _result("orphaned_provenance"),
            _result("redundancy_bloat"),
        ),
        skipped=(
            _skipped("unsupported_claim", SkipReason.MISSING_SEMANTIC_CONFIGURATION),
            _skipped("privacy_scope_violation", SkipReason.MISSING_SCOPE_POLICY),
        ),
    )


def test_audit_report_schema_version_is_independent_and_exact() -> None:
    assert AUDIT_REPORT_SCHEMA_VERSION == "0.1"
    assert _valid_report().schema_version == "0.1"

    with pytest.raises(ValidationError, match="unsupported audit report schema_version"):
        AuditReport(
            schema_version="0.2",
            results=tuple(_result(checker_id) for checker_id in PUBLIC_CHECKER_IDS),
        )


def test_audit_report_requires_a_complete_checker_partition() -> None:
    with pytest.raises(ValidationError, match="missing public checker IDs"):
        AuditReport(results=tuple(_result(checker_id) for checker_id in PUBLIC_CHECKER_IDS[:-1]))

    with pytest.raises(ValidationError, match="unknown public checker IDs"):
        AuditReport(
            results=(
                *tuple(_result(checker_id) for checker_id in PUBLIC_CHECKER_IDS),
                _custom_result("unknown_checker"),
            )
        )


def test_audit_report_rejects_duplicate_and_overlapping_checker_ids() -> None:
    results = tuple(_result(checker_id) for checker_id in PUBLIC_CHECKER_IDS)
    with pytest.raises(ValidationError, match="completed checker IDs must be unique"):
        AuditReport(results=(*results, _result("stale_active")))

    with pytest.raises(ValidationError, match="skipped checker IDs must be unique"):
        AuditReport(
            results=tuple(_result(checker_id) for checker_id in PUBLIC_CHECKER_IDS[1:]),
            skipped=(
                _skipped("orphaned_provenance", SkipReason.MISSING_TRANSCRIPTS),
                _skipped("orphaned_provenance", SkipReason.MISSING_TRANSCRIPTS),
            ),
        )

    with pytest.raises(ValidationError, match="both completed and skipped"):
        AuditReport(
            results=results,
            skipped=(_skipped("orphaned_provenance", SkipReason.MISSING_TRANSCRIPTS),),
        )


def test_audit_report_canonicalizes_result_and_skip_order() -> None:
    report = _valid_report()

    assert tuple(result.checker_id for result in report.results) == PUBLIC_CHECKER_IDS[:3]
    assert tuple(item.checker_id for item in report.skipped) == PUBLIC_CHECKER_IDS[3:]


def test_skipped_checker_validates_identity_reasons_and_order() -> None:
    skipped = SkippedChecker(
        checker_id="unsupported_claim",
        reasons=(
            SkipReason.MISSING_SEMANTIC_CONFIGURATION,
            SkipReason.MISSING_SCOPE_POLICY,
            SkipReason.MISSING_TRANSCRIPTS,
        ),
    )
    assert skipped.reasons == tuple(SkipReason)

    with pytest.raises(ValidationError, match="unknown public checker_id"):
        _skipped("unknown_checker", SkipReason.MISSING_TRANSCRIPTS)
    with pytest.raises(ValidationError, match="must not be empty"):
        SkippedChecker(checker_id="orphaned_provenance", reasons=())
    with pytest.raises(ValidationError, match="must be unique"):
        _skipped(
            "orphaned_provenance",
            SkipReason.MISSING_TRANSCRIPTS,
            SkipReason.MISSING_TRANSCRIPTS,
        )


def test_models_are_frozen_and_forbid_extra_fields() -> None:
    report = _valid_report()

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AuditReport.model_validate(
            {
                **report.model_dump(mode="json"),
                "generated_at": "2026-09-05T00:00:00Z",
            }
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SkippedChecker.model_validate(
            {
                "checker_id": "orphaned_provenance",
                "reasons": ["missing_transcripts"],
                "message": "missing input",
            }
        )
    with pytest.raises(ValidationError, match="Instance is frozen"):
        report.results = ()


def test_audit_report_json_is_deterministic_and_contains_no_runtime_fields() -> None:
    first = _valid_report().to_json()
    second = _valid_report().to_json()

    assert first == second
    assert first.endswith("\n")
    assert not _valid_report().to_json(indent=None).endswith("\n")
    payload = json.loads(first)
    assert set(payload) == {"results", "schema_version", "skipped"}
    assert not {
        "timestamp",
        "generated_at",
        "duration",
        "hostname",
        "cwd",
    } & set(payload)


def test_audit_report_file_output_matches_returned_text(tmp_path: Path) -> None:
    output = tmp_path / "audit-report.json"

    text = _valid_report().to_json(output)

    assert output.read_text(encoding="utf-8") == text


def test_audit_report_supports_an_all_skipped_partition() -> None:
    report = AuditReport(
        skipped=tuple(
            _skipped(checker_id, SkipReason.MISSING_TRANSCRIPTS)
            for checker_id in reversed(PUBLIC_CHECKER_IDS)
        )
    )

    assert report.results == ()
    assert tuple(item.checker_id for item in report.skipped) == PUBLIC_CHECKER_IDS
