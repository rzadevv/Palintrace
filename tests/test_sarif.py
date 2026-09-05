from __future__ import annotations

import json
from pathlib import Path

import pytest

from palintrace import __version__
from palintrace.audit import AuditReport, SkippedChecker, SkipReason
from palintrace.checker_requirements import PUBLIC_CHECKER_IDS
from palintrace.checkers import CheckerResult, CheckerStats, EvidenceItem, Finding
from palintrace.sarif import render_audit_report_sarif, render_sarif
from palintrace.taxonomy import DefectClass

SCHEMA_URI = (
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/os/schemas/"
    "sarif-schema-2.1.0.json"
)


def _finding(
    finding_id: str = "finding-one",
    *,
    memory_ids: tuple[str, ...] = ("memory-b", "memory-a"),
    confidence: float = 0.25,
    defect_class: DefectClass = DefectClass.ORPHANED_PROVENANCE,
) -> Finding:
    return Finding(
        finding_id=finding_id,
        defect_class=defect_class,
        memory_ids=memory_ids,
        confidence=confidence,
        evidence=(
            EvidenceItem(kind="second", message="Second message.", data={"index": 2}),
            EvidenceItem(kind="first", message="First message.", data={"index": 1}),
        ),
    )


def _result(
    *findings: Finding,
    checker_id: str = "alternate-provenance",
    rule_id: str = "memory.provenance.orphaned",
    severity: str = "warning",
) -> CheckerResult:
    return CheckerResult(
        checker_id=checker_id,
        checker_version="test-1",
        rule_id=rule_id,
        rule_version="1.2.3",
        severity=severity,
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        findings=findings,
        stats=CheckerStats(memories_scanned=2, findings_emitted=len(findings)),
    )


def _document(result: CheckerResult) -> dict[str, object]:
    return json.loads(render_sarif(result))


_BUILTIN_DEFECTS = {
    "orphaned_provenance": DefectClass.ORPHANED_PROVENANCE,
    "redundancy_bloat": DefectClass.REDUNDANCY_BLOAT,
    "stale_active": DefectClass.STALE_ACTIVE,
    "privacy_scope_violation": DefectClass.PRIVACY_SCOPE_VIOLATION,
    "unsupported_claim": DefectClass.UNSUPPORTED_CLAIM,
}


def _builtin_result(checker_id: str, *findings: Finding) -> CheckerResult:
    return CheckerResult(
        checker_id=checker_id,
        checker_version="1.0",
        defect_class=_BUILTIN_DEFECTS[checker_id],
        findings=findings,
        stats=CheckerStats(memories_scanned=2, findings_emitted=len(findings)),
    )


def _aggregate_report() -> AuditReport:
    return AuditReport(
        results=(
            _builtin_result(
                "stale_active",
                _finding(
                    "finding-stale",
                    memory_ids=("memory-stale",),
                    confidence=1.0,
                    defect_class=DefectClass.STALE_ACTIVE,
                ),
            ),
            _builtin_result("orphaned_provenance"),
            _builtin_result(
                "redundancy_bloat",
                _finding(
                    "finding-redundancy-b",
                    memory_ids=("memory-b",),
                    confidence=0.2,
                    defect_class=DefectClass.REDUNDANCY_BLOAT,
                ),
                _finding(
                    "finding-redundancy-a",
                    memory_ids=("memory-a",),
                    confidence=0.1,
                    defect_class=DefectClass.REDUNDANCY_BLOAT,
                ),
            ),
        ),
        skipped=(
            SkippedChecker(
                checker_id="unsupported_claim",
                reasons=(
                    SkipReason.MISSING_SEMANTIC_CONFIGURATION,
                    SkipReason.MISSING_TRANSCRIPTS,
                ),
            ),
            SkippedChecker(
                checker_id="privacy_scope_violation",
                reasons=(SkipReason.MISSING_SCOPE_POLICY,),
            ),
        ),
    )


def _aggregate_document(report: AuditReport) -> dict[str, object]:
    return json.loads(render_audit_report_sarif(report))


def test_sarif_root_and_tool_driver() -> None:
    document = _document(_result())

    assert document["$schema"] == SCHEMA_URI
    assert document["version"] == "2.1.0"
    assert len(document["runs"]) == 1
    run = document["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] == "Palintrace"
    assert driver["version"] == __version__ == "0.3.0"
    assert len(driver["rules"]) == 1
    assert run["results"] == []


def test_sarif_rule_identity_comes_from_checker_result() -> None:
    builtin = CheckerResult(
        checker_id="orphaned_provenance",
        checker_version="1.0",
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        findings=(_finding(),),
        stats=CheckerStats(memories_scanned=2, findings_emitted=1),
    )
    alternate = _result(
        _finding(),
        checker_id="alternate-provenance",
        rule_id=builtin.rule_id,
    )

    for result in (builtin, alternate):
        run = _document(result)["runs"][0]
        assert run["tool"]["driver"]["rules"][0]["id"] == result.rule_id
        assert run["results"][0]["ruleId"] == result.rule_id
        assert run["results"][0]["ruleId"] != result.checker_id


@pytest.mark.parametrize(
    ("severity", "level"),
    [("info", "note"), ("warning", "warning"), ("error", "error")],
)
def test_sarif_severity_mapping(severity: str, level: str) -> None:
    run = _document(_result(_finding(), severity=severity))["runs"][0]

    assert run["tool"]["driver"]["rules"][0]["defaultConfiguration"]["level"] == level
    assert run["results"][0]["level"] == level


@pytest.mark.parametrize("count", [0, 1, 2])
def test_each_finding_becomes_one_sarif_result(count: int) -> None:
    findings = tuple(
        _finding(f"finding-{index}", memory_ids=(f"memory-{index}",))
        for index in range(count)
    )

    results = _document(_result(*findings))["runs"][0]["results"]

    assert len(results) == count


def test_sarif_result_preserves_finding_metadata_and_evidence() -> None:
    result = _result(_finding())
    finding = result.findings[0]
    sarif_result = _document(result)["runs"][0]["results"][0]
    properties = sarif_result["properties"]

    assert sarif_result["ruleId"] == result.rule_id
    assert sarif_result["ruleIndex"] == 0
    assert sarif_result["fingerprints"]["palintraceFindingId"] == finding.finding_id
    assert sarif_result["message"]["text"] == "; ".join(
        item.message for item in finding.evidence
    )
    assert properties == {
        "finding_id": finding.finding_id,
        "checker_id": result.checker_id,
        "checker_version": result.checker_version,
        "rule_version": result.rule_version,
        "defect_class": result.defect_class.value,
        "severity": result.severity,
        "memory_ids": list(finding.memory_ids),
        "confidence": finding.confidence,
        "checker_result_schema": result.schema_version,
        "evidence": [item.model_dump(mode="json") for item in finding.evidence],
    }


def test_sarif_uses_only_logical_memory_locations() -> None:
    result = _result(_finding())
    sarif_result = _document(result)["runs"][0]["results"][0]

    assert sarif_result["locations"] == [
        {"logicalLocations": [{"name": memory_id, "kind": "memory"}]}
        for memory_id in result.findings[0].memory_ids
    ]
    serialized = json.dumps(sarif_result)
    for forbidden in ("physicalLocation", "artifactLocation", '"line"', '"column"'):
        assert forbidden not in serialized


def test_sarif_is_deterministic_and_does_not_mutate_canonical_result() -> None:
    result = _result(_finding())
    canonical_before = result.to_json()

    first = render_sarif(result)
    second = render_sarif(result)

    assert first == second
    assert json.loads(first)
    assert result.to_json() == canonical_before
    for forbidden in (
        "executed_at",
        "generated_at",
        "timestamp",
        "duration",
        "hostname",
        '"cwd"',
    ):
        assert forbidden not in first


def test_sarif_file_output_matches_returned_text(tmp_path: Path) -> None:
    output = tmp_path / "result.sarif"

    text = render_sarif(_result(_finding()), output)

    assert output.read_text(encoding="utf-8") == text
    assert output.read_bytes() == text.encode("utf-8")


def test_aggregate_sarif_root_driver_and_rule_order() -> None:
    report = _aggregate_report()
    document = _aggregate_document(report)

    assert document["$schema"] == SCHEMA_URI
    assert document["version"] == "2.1.0"
    assert len(document["runs"]) == 1
    run = document["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] == "Palintrace"
    assert driver["version"] == __version__ == "0.3.0"
    assert [rule["id"] for rule in driver["rules"]] == [
        result.rule_id for result in report.results
    ]
    assert len(driver["rules"]) == len(report.results) == 3


def test_aggregate_sarif_rule_indices_follow_completed_result_order() -> None:
    report = _aggregate_report()
    run = _aggregate_document(report)["runs"][0]
    sarif_results = run["results"]

    assert [item["properties"]["finding_id"] for item in sarif_results] == [
        "finding-redundancy-a",
        "finding-redundancy-b",
        "finding-stale",
    ]
    assert [(item["ruleId"], item["ruleIndex"]) for item in sarif_results] == [
        (report.results[1].rule_id, 1),
        (report.results[1].rule_id, 1),
        (report.results[2].rule_id, 2),
    ]


def test_aggregate_sarif_declares_completed_rules_without_findings() -> None:
    report = _aggregate_report()
    run = _aggregate_document(report)["runs"][0]
    orphaned = report.results[0]

    assert orphaned.findings == ()
    assert run["tool"]["driver"]["rules"][0]["id"] == orphaned.rule_id
    assert all(item["ruleIndex"] != 0 for item in run["results"])


def test_aggregate_sarif_preserves_skips_only_as_exact_run_properties() -> None:
    report = _aggregate_report()
    run = _aggregate_document(report)["runs"][0]

    assert run["properties"] == {
        "audit_report_schema": "0.1",
        "skipped": [item.model_dump(mode="json") for item in report.skipped],
    }
    completed_rule_ids = {
        rule["id"] for rule in run["tool"]["driver"]["rules"]
    }
    skipped_checker_ids = {item.checker_id for item in report.skipped}
    assert completed_rule_ids == {result.rule_id for result in report.results}
    assert all(
        item["properties"]["checker_id"] not in skipped_checker_ids
        for item in run["results"]
    )


def test_aggregate_sarif_keeps_empty_skipped_property() -> None:
    report = AuditReport(
        results=tuple(_builtin_result(checker_id) for checker_id in PUBLIC_CHECKER_IDS)
    )
    run = _aggregate_document(report)["runs"][0]

    assert run["properties"] == {
        "audit_report_schema": "0.1",
        "skipped": [],
    }
    assert len(run["tool"]["driver"]["rules"]) == 5
    assert run["results"] == []


def test_aggregate_sarif_supports_all_skipped_report() -> None:
    report = AuditReport(
        skipped=tuple(
            SkippedChecker(
                checker_id=checker_id,
                reasons=(SkipReason.MISSING_TRANSCRIPTS,),
            )
            for checker_id in PUBLIC_CHECKER_IDS
        )
    )
    document = _aggregate_document(report)
    run = document["runs"][0]

    assert len(document["runs"]) == 1
    assert run["tool"]["driver"]["rules"] == []
    assert run["results"] == []
    assert run["properties"] == {
        "audit_report_schema": "0.1",
        "skipped": [item.model_dump(mode="json") for item in report.skipped],
    }


def test_aggregate_sarif_maps_each_result_severity_independently() -> None:
    report = _aggregate_report()
    run = _aggregate_document(report)["runs"][0]
    rules = run["tool"]["driver"]["rules"]
    sarif_results = run["results"]

    assert [rule["defaultConfiguration"]["level"] for rule in rules] == [
        "error",
        "warning",
        "error",
    ]
    assert [item["level"] for item in sarif_results] == [
        "warning",
        "warning",
        "error",
    ]


def test_aggregate_sarif_preserves_existing_finding_projection() -> None:
    report = _aggregate_report()
    redundancy = report.results[1]
    aggregate_finding = _aggregate_document(report)["runs"][0]["results"][0]
    single_finding = _document(redundancy)["runs"][0]["results"][0]

    assert {**aggregate_finding, "ruleIndex": 0} == single_finding
    assert aggregate_finding["ruleId"] == redundancy.rule_id
    assert aggregate_finding["ruleIndex"] == 1
    assert aggregate_finding["fingerprints"]["palintraceFindingId"] == (
        redundancy.findings[0].finding_id
    )
    assert aggregate_finding["locations"] == [
        {"logicalLocations": [{"name": "memory-a", "kind": "memory"}]}
    ]
    assert "physicalLocation" not in json.dumps(aggregate_finding)


def test_aggregate_sarif_is_deterministic_without_mutating_report() -> None:
    report = _aggregate_report()
    canonical_before = report.to_json()

    first = render_audit_report_sarif(report)
    second = render_audit_report_sarif(report)

    assert first == second
    assert first.endswith("\n")
    assert not render_audit_report_sarif(report, indent=None).endswith("\n")
    assert report.to_json() == canonical_before
    for forbidden in (
        "executed_at",
        "generated_at",
        "timestamp",
        "duration",
        "hostname",
        '"cwd"',
    ):
        assert forbidden not in first


def test_aggregate_sarif_file_output_matches_returned_text(tmp_path: Path) -> None:
    output = tmp_path / "aggregate.sarif"

    text = render_audit_report_sarif(_aggregate_report(), output)

    assert output.read_text(encoding="utf-8") == text
    assert output.read_bytes() == text.encode("utf-8")


def test_single_result_sarif_retains_nonaggregate_shape() -> None:
    result = _result(_finding())
    run = _document(result)["runs"][0]

    assert len(run["tool"]["driver"]["rules"]) == 1
    assert run["results"][0]["ruleIndex"] == 0
    assert "properties" not in run
    assert "audit_report_schema" not in json.dumps(run)
    assert '"skipped"' not in json.dumps(run)
