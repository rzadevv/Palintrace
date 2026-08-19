import pytest
from pydantic import ValidationError

from memlint.checkers import (
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
)
from memlint.taxonomy import DefectClass


def _evidence() -> EvidenceItem:
    return EvidenceItem(
        kind="missing_turn",
        message="Referenced transcript turn does not exist.",
        data={"source_ref_index": 0, "transcript_id": "t1", "turn_idx": 9},
    )


def _finding(
    *,
    finding_id: str = "finding-0123456789abcdef01234567",
    memory_id: str = "m1",
    defect_class: DefectClass = DefectClass.ORPHANED_PROVENANCE,
) -> Finding:
    return Finding(
        finding_id=finding_id,
        defect_class=defect_class,
        memory_ids=(memory_id,),
        confidence=1.0,
        evidence=(_evidence(),),
    )


def _result(*findings: Finding) -> CheckerResult:
    return CheckerResult(
        checker_id="orphaned_provenance",
        checker_version="1.0",
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        findings=findings,
        cost=CheckerCost(),
        stats=CheckerStats(
            memories_scanned=len(findings),
            source_refs_scanned=len(findings),
            findings_emitted=len(findings),
        ),
    )


def test_evidence_item_validates_identity_message_and_json_data() -> None:
    with pytest.raises(ValidationError, match="kind and message"):
        EvidenceItem(kind=" ", message="message")
    with pytest.raises(ValidationError, match="kind and message"):
        EvidenceItem(kind="kind", message=" ")
    with pytest.raises(ValidationError):
        EvidenceItem(kind="kind", message="message", data={"bad": object()})
    with pytest.raises(ValidationError, match="strict JSON"):
        EvidenceItem(kind="kind", message="message", data={"bad": float("nan")})


@pytest.mark.parametrize(
    "changes",
    [
        {"finding_id": " "},
        {"memory_ids": ()},
        {"memory_ids": ("m1", "m1")},
        {"confidence": -0.01},
        {"confidence": 1.01},
        {"evidence": ()},
    ],
)
def test_finding_rejects_invalid_fields(changes: dict[str, object]) -> None:
    payload = _finding().model_dump()
    payload.update(changes)

    with pytest.raises(ValidationError):
        Finding.model_validate(payload)


def test_checker_result_sorts_findings_and_serializes_deterministically() -> None:
    second = _finding(finding_id="finding-bbbbbbbbbbbbbbbbbbbbbbbb", memory_id="m2")
    first = _finding(finding_id="finding-aaaaaaaaaaaaaaaaaaaaaaaa", memory_id="m1")

    result = _result(second, first)

    assert tuple(finding.memory_ids for finding in result.findings) == (("m1",), ("m2",))
    assert first.to_json() == first.to_json()
    assert result.to_json() == result.to_json()
    assert "executed_at" not in result.to_json()
    assert "duration" not in result.to_json()


def test_checker_result_rejects_wrong_schema_and_blank_identity() -> None:
    result = _result(_finding())
    for field, value in (
        ("schema_version", "wrong"),
        ("checker_id", " "),
        ("checker_version", " "),
    ):
        payload = result.model_dump()
        payload[field] = value
        with pytest.raises(ValidationError):
            CheckerResult.model_validate(payload)


def test_checker_result_rejects_duplicate_ids_and_mismatched_defect_class() -> None:
    first = _finding(memory_id="m1")
    duplicate = _finding(memory_id="m2")
    wrong_class = _finding(
        finding_id="finding-bbbbbbbbbbbbbbbbbbbbbbbb",
        defect_class=DefectClass.UNSUPPORTED_CLAIM,
    )

    with pytest.raises(ValidationError, match="finding IDs must be unique"):
        _result(first, duplicate)
    with pytest.raises(ValidationError, match="defect_class"):
        _result(wrong_class)


def test_cost_and_stats_reject_negative_or_inconsistent_values() -> None:
    with pytest.raises(ValidationError):
        CheckerCost(model_calls=-1)
    with pytest.raises(ValidationError):
        CheckerCost(input_tokens=-1)
    with pytest.raises(ValidationError):
        CheckerCost(output_tokens=-1)

    payload = _result(_finding()).model_dump()
    payload["stats"]["findings_emitted"] = 0
    with pytest.raises(ValidationError, match="findings_emitted"):
        CheckerResult.model_validate(payload)
