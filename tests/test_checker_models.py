from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from palintrace.checkers import (
    CHECKER_RESULT_SCHEMA_VERSION,
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
    deterministic_finding_id,
)
from palintrace.taxonomy import DefectClass


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
            findings_emitted=len(findings),
            details={"source_refs_scanned": len(findings)},
        ),
    )


def test_evidence_item_validates_identity_message_and_json_data() -> None:
    with pytest.raises(ValidationError, match="kind and message"):
        EvidenceItem(kind=" ", message="message")
    with pytest.raises(ValidationError, match="kind and message"):
        EvidenceItem(kind="kind", message=" ")
    with pytest.raises(ValidationError):
        EvidenceItem(kind="kind", message="message", data={"bad": object()})
    for nonfinite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError, match="strict JSON"):
            EvidenceItem(
                kind="kind",
                message="message",
                data={"outer": {"values": [1, nonfinite]}},
            )
    with pytest.raises(ValidationError):
        EvidenceItem(
            kind="kind",
            message="message",
            data={"outer": {"values": [object()]}},
        )


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

    assert CHECKER_RESULT_SCHEMA_VERSION == "0.3"
    assert result.schema_version == "0.3"
    assert result.rule_id == "memory.provenance.orphaned"
    assert result.rule_version == "1.0.0"
    assert result.severity == "error"
    assert tuple(result.model_dump()) == (
        "schema_version",
        "checker_id",
        "checker_version",
        "rule_id",
        "rule_version",
        "severity",
        "defect_class",
        "findings",
        "cost",
        "stats",
    )
    assert tuple(finding.memory_ids for finding in result.findings) == (("m1",), ("m2",))
    assert first.to_json() == first.to_json()
    assert result.to_json() == result.to_json()
    assert CheckerResult.model_validate_json(result.to_json()).to_json() == result.to_json()
    assert "executed_at" not in result.to_json()
    assert "duration" not in result.to_json()
    assert {"rule_id", "rule_version", "severity"}.isdisjoint(Finding.model_fields)


def test_checker_result_rejects_wrong_schema_and_blank_identity() -> None:
    result = _result(_finding())
    for field, value in (
        ("schema_version", "0.2"),
        ("checker_id", " "),
        ("checker_version", " "),
    ):
        payload = result.model_dump()
        payload[field] = value
        with pytest.raises(ValidationError):
            CheckerResult.model_validate(payload)


@pytest.mark.parametrize(
    ("checker_id", "checker_version", "defect_class", "rule_id", "severity"),
    [
        (
            "orphaned_provenance",
            "1.0",
            DefectClass.ORPHANED_PROVENANCE,
            "memory.provenance.orphaned",
            "error",
        ),
        (
            "redundancy_bloat",
            "1.0",
            DefectClass.REDUNDANCY_BLOAT,
            "memory.duplication.exact",
            "warning",
        ),
        (
            "stale_active",
            "1.0",
            DefectClass.STALE_ACTIVE,
            "memory.state.explicit-stale",
            "error",
        ),
        (
            "privacy_scope_violation",
            "1.0",
            DefectClass.PRIVACY_SCOPE_VIOLATION,
            "memory.scope.prohibited-exact-replica",
            "error",
        ),
        (
            "unsupported_claim",
            "1.0",
            DefectClass.UNSUPPORTED_CLAIM,
            "memory.claim.unsupported",
            "error",
        ),
        (
            "unsupported_claim_identity_grounded",
            "0.1",
            DefectClass.UNSUPPORTED_CLAIM,
            "memory.claim.unsupported",
            "error",
        ),
        (
            "retrieval_shadowing",
            "1.0",
            DefectClass.RETRIEVAL_SHADOWING,
            "memory.retrieval.shadowing",
            "error",
        ),
    ],
)
def test_builtin_checker_results_receive_canonical_rule_metadata(
    checker_id: str,
    checker_version: str,
    defect_class: DefectClass,
    rule_id: str,
    severity: str,
) -> None:
    result = CheckerResult(
        checker_id=checker_id,
        checker_version=checker_version,
        defect_class=defect_class,
        stats=CheckerStats(memories_scanned=0, findings_emitted=0),
    )

    assert result.rule_id == rule_id
    assert result.rule_version == "1.0.0"
    assert result.severity == severity


def test_alternate_checker_implementations_can_share_a_rule() -> None:
    stats = CheckerStats(memories_scanned=0, findings_emitted=0)
    supported = CheckerResult(
        checker_id="unsupported_claim",
        checker_version="1.0",
        defect_class=DefectClass.UNSUPPORTED_CLAIM,
        stats=stats,
    )
    identity_grounded = CheckerResult(
        checker_id="unsupported_claim_identity_grounded",
        checker_version="0.1",
        defect_class=DefectClass.UNSUPPORTED_CLAIM,
        stats=stats,
    )

    assert supported.checker_id != identity_grounded.checker_id
    assert supported.rule_id == identity_grounded.rule_id == "memory.claim.unsupported"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rule_id", "memory.claim.unsupported"),
        ("rule_version", "2.0.0"),
        ("severity", "warning"),
    ],
)
def test_builtin_checker_rejects_noncanonical_rule_metadata(
    field: str, value: str
) -> None:
    payload = _result().model_dump()
    payload[field] = value

    with pytest.raises(ValidationError, match="canonical values"):
        CheckerResult.model_validate(payload)


def test_builtin_checker_id_must_match_defect_class() -> None:
    payload = _result().model_dump()
    payload["defect_class"] = DefectClass.REDUNDANCY_BLOAT

    with pytest.raises(ValidationError, match="checker_id does not match defect_class"):
        CheckerResult.model_validate(payload)


def test_custom_checker_requires_and_accepts_explicit_rule_metadata() -> None:
    payload = _result().model_dump()
    payload["checker_id"] = "custom-checker"
    for field in ("rule_id", "rule_version", "severity"):
        del payload[field]

    with pytest.raises(ValidationError, match="must supply explicit rule metadata"):
        CheckerResult.model_validate(payload)

    result = CheckerResult(
        checker_id="custom-checker",
        checker_version="revision-1",
        rule_id="memory.test.synthetic",
        rule_version="0.1.0",
        severity="info",
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        stats=CheckerStats(memories_scanned=0, findings_emitted=0),
    )

    assert result.rule_id == "memory.test.synthetic"
    assert result.rule_version == "0.1.0"
    assert result.severity == "info"


@pytest.mark.parametrize(
    "rule_id",
    [
        "",
        "Memory.test.synthetic",
        "memory.test_synthetic.value",
        "memory.test",
        "memory.test.synthetic.extra",
        "memory.palintrace.synthetic",
        "memory.test.synthetic ",
    ],
)
def test_checker_result_rejects_invalid_rule_id(rule_id: str) -> None:
    payload = _result().model_dump()
    payload["checker_id"] = "custom-checker"
    payload["rule_id"] = rule_id

    with pytest.raises(ValidationError, match="memory.<area>.<defect>"):
        CheckerResult.model_validate(payload)


@pytest.mark.parametrize("rule_version", ["1", "1.0", "v1.0.0", "latest", "rule-1"])
def test_checker_result_rejects_malformed_rule_version(rule_version: str) -> None:
    payload = _result().model_dump()
    payload["checker_id"] = "custom-checker"
    payload["rule_version"] = rule_version

    with pytest.raises(ValidationError, match="MAJOR.MINOR.PATCH"):
        CheckerResult.model_validate(payload)


def test_checker_result_rejects_invalid_severity() -> None:
    payload = _result().model_dump()
    payload["checker_id"] = "custom-checker"
    payload["severity"] = "critical"

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
    with pytest.raises(ValidationError, match="detail keys"):
        CheckerStats(memories_scanned=1, findings_emitted=0, details={" ": 1})
    with pytest.raises(ValidationError):
        CheckerStats(
            memories_scanned=1,
            findings_emitted=0,
            details={"source_refs_scanned": -1},
        )

    payload = _result(_finding()).model_dump()
    payload["stats"]["findings_emitted"] = 0
    with pytest.raises(ValidationError, match="findings_emitted"):
        CheckerResult.model_validate(payload)


def test_checker_stats_is_frozen_and_serializes_details_deterministically() -> None:
    first = CheckerStats(
        memories_scanned=2,
        findings_emitted=0,
        details={"source_refs_scanned": 2, "declared_memories_scanned": 1},
    )
    second = CheckerStats(
        memories_scanned=2,
        findings_emitted=0,
        details={"declared_memories_scanned": 1, "source_refs_scanned": 2},
    )

    first_result = _result().model_copy(update={"stats": first})
    second_result = _result().model_copy(update={"stats": second})

    assert first_result.to_json() == second_result.to_json()
    before = first_result.to_json()
    with pytest.raises(TypeError):
        first.details["source_refs_scanned"] = 999
    assert first_result.to_json() == before
    with pytest.raises(ValidationError, match="frozen"):
        first.memories_scanned = 3
    with pytest.raises(ValidationError):
        CheckerStats.model_validate(
            {
                "memories_scanned": 2,
                "findings_emitted": 0,
                "details": {},
                "unexpected": 1,
            }
        )


def test_finding_id_canonicalizes_memory_and_evidence_order() -> None:
    first = EvidenceItem(kind="missing_turn", message="First", data={"turn_idx": 9})
    second = EvidenceItem(
        kind="missing_transcript",
        message="Second",
        data={"transcript_id": "t1"},
    )

    forward = deterministic_finding_id(
        checker_id="checker",
        checker_version="1.0",
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        memory_ids=("m1", "m2"),
        evidence=(first, second),
    )
    reversed_memory_ids = deterministic_finding_id(
        checker_id="checker",
        checker_version="1.0",
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        memory_ids=("m2", "m1"),
        evidence=(first, second),
    )
    reversed_evidence = deterministic_finding_id(
        checker_id="checker",
        checker_version="1.0",
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        memory_ids=("m1", "m2"),
        evidence=(second, first),
    )

    assert forward == reversed_memory_ids
    assert forward == reversed_evidence


def test_finding_id_is_unchanged_by_result_metadata_contract() -> None:
    finding_id = deterministic_finding_id(
        checker_id="orphaned_provenance",
        checker_version="1.0",
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        memory_ids=("m1",),
        evidence=(_evidence(),),
    )

    assert finding_id == "finding-40b457405be7323e58161e25"


def test_finding_id_ignores_message_but_tracks_semantic_identity_and_version() -> None:
    original = EvidenceItem(kind="missing_turn", message="Original wording", data={"turn_idx": 9})

    def finding_id(evidence: EvidenceItem, *, checker_version: str = "1.0") -> str:
        return deterministic_finding_id(
            checker_id="checker",
            checker_version=checker_version,
            defect_class=DefectClass.ORPHANED_PROVENANCE,
            memory_ids=("m1",),
            evidence=(evidence,),
        )

    changed_message = EvidenceItem(
        kind="missing_turn",
        message="Revised wording",
        data={"turn_idx": 9},
    )
    changed_kind = EvidenceItem(
        kind="invalid_span",
        message="Original wording",
        data={"turn_idx": 9},
    )
    changed_data = EvidenceItem(
        kind="missing_turn",
        message="Original wording",
        data={"turn_idx": 10},
    )

    assert finding_id(original) == finding_id(changed_message)
    assert finding_id(original) != finding_id(changed_kind)
    assert finding_id(original) != finding_id(changed_data)
    assert finding_id(original) != finding_id(original, checker_version="1.1")


def test_evidence_is_deeply_immutable_after_finding_id_creation() -> None:
    evidence = EvidenceItem(
        kind="example",
        message="Example.",
        data={"outer": {"items": [1, 2]}},
    )
    finding_id = deterministic_finding_id(
        checker_id="checker",
        checker_version="1.0",
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        memory_ids=("m1",),
        evidence=(evidence,),
    )
    finding = Finding(
        finding_id=finding_id,
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        memory_ids=("m1",),
        confidence=1.0,
        evidence=(evidence,),
    )
    before = finding.to_json()
    outer = evidence.data["outer"]
    assert isinstance(outer, Mapping)
    items = outer["items"]

    with pytest.raises(TypeError):
        evidence.data["outer"] = {}
    with pytest.raises(TypeError):
        outer["items"] = []
    with pytest.raises(AttributeError):
        items.append(3)

    assert finding.to_json() == before
    assert finding.finding_id == deterministic_finding_id(
        checker_id="checker",
        checker_version="1.0",
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        memory_ids=finding.memory_ids,
        evidence=finding.evidence,
    )
    assert finding.model_dump(mode="json")["evidence"][0]["data"] == {
        "outer": {"items": [1, 2]}
    }


def test_finding_canonicalizes_visible_memory_and_evidence_order() -> None:
    first = EvidenceItem(kind="missing_turn", message="First", data={"turn_idx": 9})
    second = EvidenceItem(
        kind="missing_transcript",
        message="Second",
        data={"transcript_id": "t1"},
    )
    finding_id = deterministic_finding_id(
        checker_id="checker",
        checker_version="1.0",
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        memory_ids=("m1", "m2"),
        evidence=(first, second),
    )
    forward = Finding(
        finding_id=finding_id,
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        memory_ids=("m1", "m2"),
        confidence=1.0,
        evidence=(first, second),
    )
    reversed_inputs = Finding(
        finding_id=finding_id,
        defect_class=DefectClass.ORPHANED_PROVENANCE,
        memory_ids=("m2", "m1"),
        confidence=1.0,
        evidence=(second, first),
    )

    assert forward.memory_ids == ("m1", "m2")
    assert reversed_inputs.memory_ids == ("m1", "m2")
    assert forward.evidence == reversed_inputs.evidence
    assert forward.to_json() == reversed_inputs.to_json()
