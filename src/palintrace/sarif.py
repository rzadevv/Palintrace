"""Deterministic SARIF rendering for checker results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from palintrace import __version__
from palintrace.audit import AuditReport
from palintrace.checkers import CheckerResult, Finding

_SCHEMA_URI = (
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/os/schemas/"
    "sarif-schema-2.1.0.json"
)
_SARIF_LEVEL = {"info": "note", "warning": "warning", "error": "error"}


def _render_finding(
    result: CheckerResult,
    finding: Finding,
    level: str,
    *,
    rule_index: int,
) -> dict[str, object]:
    return {
        "ruleId": result.rule_id,
        "ruleIndex": rule_index,
        "level": level,
        "message": {"text": "; ".join(item.message for item in finding.evidence)},
        "fingerprints": {"palintraceFindingId": finding.finding_id},
        "locations": [
            {"logicalLocations": [{"name": memory_id, "kind": "memory"}]}
            for memory_id in finding.memory_ids
        ],
        "properties": {
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
        },
    }


def _render_rule(result: CheckerResult, level: str) -> dict[str, object]:
    return {
        "id": result.rule_id,
        "defaultConfiguration": {"level": level},
        "properties": {
            "rule_version": result.rule_version,
            "checker_id": result.checker_id,
            "checker_version": result.checker_version,
            "defect_class": result.defect_class.value,
            "severity": result.severity,
        },
    }


def _serialize_sarif(
    document: Mapping[str, object],
    output: str | Path | None,
    *,
    indent: int | None,
) -> str:
    text = json.dumps(
        document,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )
    if indent is not None:
        text += "\n"
    if output is not None:
        Path(output).write_text(text, encoding="utf-8")
    return text


def render_sarif(
    result: CheckerResult,
    output: str | Path | None = None,
    *,
    indent: int | None = 2,
) -> str:
    level = _SARIF_LEVEL[result.severity]
    document = {
        "$schema": _SCHEMA_URI,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Palintrace",
                        "version": __version__,
                        "rules": [_render_rule(result, level)],
                    }
                },
                "results": [
                    _render_finding(result, finding, level, rule_index=0)
                    for finding in result.findings
                ],
            }
        ],
    }
    return _serialize_sarif(document, output, indent=indent)


def render_audit_report_sarif(
    report: AuditReport,
    output: str | Path | None = None,
    *,
    indent: int | None = 2,
) -> str:
    """Project one complete aggregate audit report into one SARIF run."""

    rules: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    for rule_index, result in enumerate(report.results):
        level = _SARIF_LEVEL[result.severity]
        rules.append(_render_rule(result, level))
        findings.extend(
            _render_finding(result, finding, level, rule_index=rule_index)
            for finding in result.findings
        )

    document = {
        "$schema": _SCHEMA_URI,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Palintrace",
                        "version": __version__,
                        "rules": rules,
                    }
                },
                "results": findings,
                "properties": {
                    "audit_report_schema": report.schema_version,
                    "skipped": [item.model_dump(mode="json") for item in report.skipped],
                },
            }
        ],
    }
    return _serialize_sarif(document, output, indent=indent)
