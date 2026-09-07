import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from palintrace.checkers import Checker, CheckerResult, RedundancyBloatChecker
from palintrace.checkers.base import deterministic_finding_id
from palintrace.cli import main
from palintrace.models import (
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    TranscriptSet,
)
from palintrace.mutations import MutationRequest, mutate
from palintrace.serialization import load_store
from palintrace.taxonomy import DefectClass

ScopeValues = tuple[str | None, str | None, str | None]


def _memory(
    memory_id: str,
    content: str = "User prefers Python.",
    *,
    scope: ScopeValues = ("user-a", None, None),
    **changes: object,
) -> NormalizedMemory:
    payload: dict[str, object] = {
        "id": memory_id,
        "content": content,
        "scope": {
            "user_id": scope[0],
            "agent_id": scope[1],
            "session_id": scope[2],
        },
    }
    payload.update(changes)
    return NormalizedMemory.model_validate(payload)


def _store(*memories: NormalizedMemory) -> NormalizedStore:
    return NormalizedStore(adapter="test", memories=memories)


def test_two_exact_same_scope_duplicates_emit_one_group_finding() -> None:
    content = "User prefers Python."
    checker: Checker = RedundancyBloatChecker()
    result = checker.check(_store(_memory("m2", content), _memory("m1", content)))

    assert checker.checker_id == "redundancy_bloat"
    assert checker.checker_version == "2.0"
    assert checker.defect_class is DefectClass.REDUNDANCY_BLOAT
    assert result.rule_id == "memory.duplication.exact"
    assert result.rule_version == "1.0.0"
    assert result.severity == "warning"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.defect_class is DefectClass.REDUNDANCY_BLOAT
    assert finding.memory_ids == ("m1", "m2")
    assert finding.confidence == 1.0
    assert len(finding.evidence) == 1
    assert finding.evidence[0].kind == "exact_duplicate"
    assert finding.evidence[0].model_dump(mode="json")["data"] == {
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_length": len(content),
        "scope": {
            "user_id": "user-a",
            "agent_id": None,
            "session_id": None,
        },
    }
    assert content not in result.to_json()
    assert result.cost.model_calls == 0
    assert result.cost.input_tokens == 0
    assert result.cost.output_tokens == 0
    assert result.stats.memories_scanned == 2
    assert result.stats.findings_emitted == 1
    assert result.stats.details == {
        "eligible_memories": 2,
        "unscoped_memories_skipped": 0,
        "duplicate_groups": 1,
    }


def test_three_duplicates_emit_one_complete_group_with_deterministic_identity() -> None:
    memories = (_memory("c"), _memory("a"), _memory("b"))
    checker = RedundancyBloatChecker()

    first = checker.check(_store(*memories))
    second = checker.check(
        _store(*reversed(memories)),
        transcripts=TranscriptSet(),
    )

    assert len(first.findings) == 1
    finding = first.findings[0]
    assert finding.memory_ids == ("a", "b", "c")
    assert first.stats.details["duplicate_groups"] == 1
    assert finding.finding_id == deterministic_finding_id(
        checker_id="redundancy_bloat",
        checker_version="2.0",
        defect_class=DefectClass.REDUNDANCY_BLOAT,
        memory_ids=("a", "b", "c"),
        evidence=finding.evidence,
    )
    assert finding.finding_id != deterministic_finding_id(
        checker_id="redundancy_bloat",
        checker_version="1.0",
        defect_class=DefectClass.REDUNDANCY_BLOAT,
        memory_ids=("a", "b"),
        evidence=finding.evidence,
    )
    assert first.to_json() == second.to_json()


def test_no_duplicates_emit_no_findings_or_duplicate_groups() -> None:
    result = RedundancyBloatChecker().check(
        _store(_memory("a", "First"), _memory("b", "Second"))
    )

    assert result.findings == ()
    assert result.stats.findings_emitted == 0
    assert result.stats.details["duplicate_groups"] == 0


def test_two_duplicate_groups_emit_once_each_in_memory_id_order() -> None:
    memories = (
        _memory("z2", "Z group"),
        _memory("a2", "A group"),
        _memory("z1", "Z group"),
        _memory("a1", "A group"),
    )

    first = RedundancyBloatChecker().check(_store(*memories))
    second = RedundancyBloatChecker().check(_store(*reversed(memories)))

    assert tuple(finding.memory_ids for finding in first.findings) == (
        ("a1", "a2"),
        ("z1", "z2"),
    )
    assert first.stats.findings_emitted == 2
    assert first.stats.details["duplicate_groups"] == 2
    assert first.to_json() == second.to_json()


def test_completely_unscoped_memories_remain_skipped() -> None:
    result = RedundancyBloatChecker().check(
        _store(
            _memory("a", scope=(None, None, None)),
            _memory("b", scope=(None, None, None)),
        )
    )

    assert result.findings == ()
    assert result.stats.details == {
        "eligible_memories": 0,
        "unscoped_memories_skipped": 2,
        "duplicate_groups": 0,
    }


@pytest.mark.parametrize(
    ("first_content", "second_content", "first_scope", "second_scope"),
    [
        ("Claim one", "Claim two", ("user-a", None, None), ("user-a", None, None)),
        ("Same", "Same", ("user-a", None, None), ("user-b", None, None)),
        ("Same", "Same", ("user-a", "agent-a", None), ("user-a", "agent-b", None)),
        ("Same", "Same", ("user-a", None, "s1"), ("user-a", None, "s2")),
        ("Same", "Same", ("user-a", None, None), (None, None, None)),
        ("Same", "Same", (None, None, None), (None, None, None)),
        (
            "User prefers Python.",
            "user prefers python.",
            ("user-a", None, None),
            ("user-a", None, None),
        ),
        (
            "User prefers Python.",
            "User prefers Python. ",
            ("user-a", None, None),
            ("user-a", None, None),
        ),
        (
            "User prefers Python.",
            "User uses Python.",
            ("user-a", None, None),
            ("user-a", None, None),
        ),
        (
            "Same",
            "Same",
            ("user-a", None, None),
            ("user-a", "agent-x", None),
        ),
    ],
    ids=(
        "different_content",
        "different_user",
        "different_agent",
        "different_session",
        "scoped_and_unscoped",
        "both_unscoped",
        "case_difference",
        "whitespace_difference",
        "semantic_only",
        "unknown_vs_known_dimension",
    ),
)
def test_non_exact_or_non_shared_scope_pairs_are_not_flagged(
    first_content: str,
    second_content: str,
    first_scope: ScopeValues,
    second_scope: ScopeValues,
) -> None:
    result = RedundancyBloatChecker().check(
        _store(
            _memory("m1", first_content, scope=first_scope),
            _memory("m2", second_content, scope=second_scope),
        )
    )

    assert result.findings == ()


def test_equal_partially_known_scope_is_sufficient() -> None:
    result = RedundancyBloatChecker().check(
        _store(
            _memory("m1", scope=("user-a", None, None)),
            _memory("m2", scope=("user-a", None, None)),
        )
    )

    assert tuple(finding.memory_ids for finding in result.findings) == (("m1", "m2"),)


def test_different_metadata_does_not_change_exact_claim_identity() -> None:
    first = _memory(
        "m1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_refs=(SourceRef(transcript_id="t1", turn_idx=0),),
        provenance_status=ProvenanceStatus.DECLARED,
        active=True,
    )
    second = _memory(
        "m2",
        created_at=datetime(2026, 2, 1, tzinfo=UTC),
        source_refs=(SourceRef(transcript_id="t2", turn_idx=4),),
        provenance_status=ProvenanceStatus.DECLARED,
        active=False,
    )

    result = RedundancyBloatChecker().check(_store(first, second))

    assert tuple(finding.memory_ids for finding in result.findings) == (("m1", "m2"),)


def test_checker_matches_exact_duplicate_mutation_gold_without_receiving_manifest() -> None:
    base_store = load_store("examples/mutation-store.json")
    mutation_result = mutate(
        base_store,
        MutationRequest(
            defect_class=DefectClass.REDUNDANCY_BLOAT,
            subtype="exact_duplicate",
            target_memory_id="preference-python",
            seed=42,
        ),
    )

    checker_result = RedundancyBloatChecker().check(mutation_result.mutated_store)
    manifest = mutation_result.manifest

    assert len(checker_result.findings) == 1
    assert checker_result.findings[0].memory_ids == tuple(sorted(manifest.gold_label.memory_ids))


def test_redundancy_audit_cli_needs_no_transcripts_and_ignores_them(tmp_path: Path) -> None:
    mutation_result = mutate(
        load_store("examples/mutation-store.json"),
        MutationRequest(
            defect_class=DefectClass.REDUNDANCY_BLOAT,
            subtype="exact_duplicate",
            target_memory_id="preference-python",
        ),
    )
    store_path = tmp_path / "duplicated.json"
    without_transcripts = tmp_path / "without-transcripts.json"
    with_transcripts = tmp_path / "with-transcripts.json"
    mutation_result.mutated_store.to_json(store_path)

    common = [
        "audit",
        "--store",
        str(store_path),
        "--checker",
        "redundancy_bloat",
    ]
    assert main([*common, "--output", str(without_transcripts)]) == 0
    assert (
        main(
            [
                *common,
                "--transcripts",
                "examples/mutation-transcripts.json",
                "--output",
                str(with_transcripts),
            ]
        )
        == 0
    )

    result = CheckerResult.model_validate_json(without_transcripts.read_text(encoding="utf-8"))
    assert len(result.findings) == 1
    assert without_transcripts.read_bytes() == with_transcripts.read_bytes()
