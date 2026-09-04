from __future__ import annotations

import json
from pathlib import Path

import pytest

from palintrace import __version__
from palintrace.checkers import CheckerResult, CheckerStats, EvidenceItem, Finding
from palintrace.sarif import render_sarif
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
) -> Finding:
    return Finding(
        finding_id=finding_id,
        defect_class=DefectClass.ORPHANED_PROVENANCE,
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
