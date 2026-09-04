"""Deterministic SARIF rendering for checker results."""

from __future__ import annotations

import json
from pathlib import Path

from palintrace import __version__
from palintrace.checkers import CheckerResult, Finding

_SCHEMA_URI = (
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/os/schemas/"
    "sarif-schema-2.1.0.json"
)
_SARIF_LEVEL = {"info": "note", "warning": "warning", "error": "error"}


def _render_finding(
    result: CheckerResult, finding: Finding, level: str
) -> dict[str, object]:
    return {
        "ruleId": result.rule_id,
        "ruleIndex": 0,
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
                        "rules": [
                            {
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
                        ],
                    }
                },
                "results": [
                    _render_finding(result, finding, level) for finding in result.findings
                ],
            }
        ],
    }
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
