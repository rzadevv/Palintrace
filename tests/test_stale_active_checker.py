from datetime import UTC, datetime
from pathlib import Path

import pytest

from palintrace.checkers import Checker, CheckerResult, StaleActiveChecker
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


def _memory(
    memory_id: str,
    content: str = "A memory",
    *,
    active: bool | None = None,
    supersedes: tuple[str, ...] = (),
    **changes: object,
) -> NormalizedMemory:
    payload: dict[str, object] = {
        "id": memory_id,
        "content": content,
        "active": active,
        "supersedes": supersedes,
    }
    payload.update(changes)
    return NormalizedMemory.model_validate(payload)


def _store(*memories: NormalizedMemory) -> NormalizedStore:
    return NormalizedStore(adapter="test", memories=memories)


def test_explicit_supersession_of_active_memory_emits_structural_finding() -> None:
    checker: Checker = StaleActiveChecker()
    result = checker.check(
        _store(
            _memory("m1", "User works at A.", active=True),
            _memory("m2", "User works at B.", active=False, supersedes=("m1",)),
        )
    )

    assert checker.checker_id == "stale_active"
    assert checker.checker_version == "1.0"
    assert checker.defect_class is DefectClass.STALE_ACTIVE
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.defect_class is DefectClass.STALE_ACTIVE
    assert finding.memory_ids == ("m1",)
    assert finding.confidence == 1.0
    assert len(finding.evidence) == 1
    assert finding.evidence[0].kind == "active_superseded"
    assert finding.evidence[0].model_dump(mode="json")["data"] == {
        "superseding_memory_id": "m2",
        "old_active": True,
    }
    assert "User works at A." not in result.to_json()
    assert "User works at B." not in result.to_json()
    assert result.cost.model_calls == 0
    assert result.cost.input_tokens == 0
    assert result.cost.output_tokens == 0
    assert result.stats.memories_scanned == 2
    assert result.stats.findings_emitted == 1
    assert result.stats.details == {
        "supersession_links_scanned": 1,
        "resolved_supersession_links": 1,
        "missing_targets_skipped": 0,
        "self_links_skipped": 0,
    }


@pytest.mark.parametrize("active", [False, None], ids=("inactive", "unknown"))
def test_superseded_memory_must_be_explicitly_active(active: bool | None) -> None:
    result = StaleActiveChecker().check(
        _store(
            _memory("m1", active=active),
            _memory("m2", supersedes=("m1",)),
        )
    )

    assert result.findings == ()
    assert result.stats.details["resolved_supersession_links"] == 1


def test_multiple_superseders_are_aggregated_into_one_finding() -> None:
    memories = (
        _memory("m1", active=True),
        _memory("m3", active=None, supersedes=("m1",)),
        _memory("m2", active=False, supersedes=("m1",)),
    )
    checker = StaleActiveChecker()

    first = checker.check(_store(*memories))
    second = checker.check(
        _store(*reversed(memories)),
        transcripts=TranscriptSet(),
    )

    assert len(first.findings) == 1
    assert first.findings[0].memory_ids == ("m1",)
    assert tuple(
        item.data["superseding_memory_id"] for item in first.findings[0].evidence
    ) == ("m2", "m3")
    assert len(first.findings[0].evidence) == 2
    assert first.to_json() == second.to_json()


def test_different_stale_memories_produce_separate_findings() -> None:
    result = StaleActiveChecker().check(
        _store(
            _memory("m4", supersedes=("m3",)),
            _memory("m3", active=True),
            _memory("m2", supersedes=("m1",)),
            _memory("m1", active=True),
        )
    )

    assert tuple(finding.memory_ids for finding in result.findings) == (("m1",), ("m3",))


def test_explicit_chain_reports_each_directly_superseded_active_memory() -> None:
    all_active = StaleActiveChecker().check(
        _store(
            _memory("m1", active=True),
            _memory("m2", active=True, supersedes=("m1",)),
            _memory("m3", active=True, supersedes=("m2",)),
        )
    )
    middle_inactive = StaleActiveChecker().check(
        _store(
            _memory("m1", active=True),
            _memory("m2", active=False, supersedes=("m1",)),
            _memory("m3", active=True, supersedes=("m2",)),
        )
    )

    assert tuple(finding.memory_ids for finding in all_active.findings) == (("m1",), ("m2",))
    assert tuple(finding.memory_ids for finding in middle_inactive.findings) == (("m1",),)


def test_missing_target_and_self_link_do_not_create_findings() -> None:
    self_link = _memory("m1", active=True).model_copy(update={"supersedes": ("m1",)})
    malformed_store = NormalizedStore.model_construct(
        adapter="test",
        memories=(self_link, _memory("m2", supersedes=("missing-id",))),
    )
    result = StaleActiveChecker().check(
        malformed_store
    )

    assert result.findings == ()
    assert result.stats.details == {
        "supersession_links_scanned": 2,
        "resolved_supersession_links": 0,
        "missing_targets_skipped": 1,
        "self_links_skipped": 1,
    }


def test_content_and_timestamps_without_explicit_link_do_not_infer_staleness() -> None:
    result = StaleActiveChecker().check(
        _store(
            _memory(
                "m1",
                "User works at A.",
                active=True,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            _memory(
                "m2",
                "User works at B.",
                active=True,
                created_at=datetime(2026, 2, 1, tzinfo=UTC),
            ),
        )
    )

    assert result.findings == ()
    assert result.stats.details["supersession_links_scanned"] == 0


def test_metadata_and_unknown_scope_do_not_override_explicit_stale_rule() -> None:
    old = _memory(
        "m1",
        active=True,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_refs=(SourceRef(transcript_id="t1", turn_idx=0),),
        provenance_status=ProvenanceStatus.DECLARED,
        embedding=(0.1, 0.2),
    )
    replacement = _memory(
        "m2",
        active=None,
        supersedes=("m1",),
        created_at=datetime(2026, 2, 1, tzinfo=UTC),
        embedding=(0.9, 0.8),
        scope={"user_id": "user-b"},
    )

    result = StaleActiveChecker().check(_store(old, replacement))

    assert tuple(finding.memory_ids for finding in result.findings) == (("m1",),)


def test_checker_matches_stale_mutation_gold_without_receiving_manifest() -> None:
    mutation_result = mutate(
        load_store("examples/mutation-store.json"),
        MutationRequest(
            defect_class=DefectClass.STALE_ACTIVE,
            subtype="explicit_supersession",
            target_memory_id="employment-aster",
            replace_from="Aster Labs",
            replace_to="Beacon Works",
            seed=42,
        ),
    )

    checker_result = StaleActiveChecker().check(mutation_result.mutated_store)
    manifest = mutation_result.manifest

    assert len(checker_result.findings) == 1
    assert checker_result.findings[0].memory_ids == manifest.gold_label.memory_ids


def test_stale_audit_cli_needs_no_transcripts_and_ignores_them(tmp_path: Path) -> None:
    mutation_result = mutate(
        load_store("examples/mutation-store.json"),
        MutationRequest(
            defect_class=DefectClass.STALE_ACTIVE,
            subtype="explicit_supersession",
            target_memory_id="employment-aster",
            replace_from="Aster Labs",
            replace_to="Beacon Works",
        ),
    )
    store_path = tmp_path / "stale.json"
    without_transcripts = tmp_path / "without-transcripts.json"
    with_transcripts = tmp_path / "with-transcripts.json"
    mutation_result.mutated_store.to_json(store_path)

    common = ["audit", "--store", str(store_path), "--checker", "stale_active"]
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
    assert tuple(finding.memory_ids for finding in result.findings) == (("employment-aster",),)
    assert without_transcripts.read_bytes() == with_transcripts.read_bytes()
