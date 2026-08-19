import pytest

from memlint.checkers import Checker, CheckerInputError, OrphanedProvenanceChecker
from memlint.models import (
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    Transcript,
    TranscriptSet,
    TranscriptTurn,
)
from memlint.mutations import MutationRequest, mutate
from memlint.serialization import load_store, load_transcripts
from memlint.taxonomy import DefectClass


def _store(*memories: NormalizedMemory) -> NormalizedStore:
    return NormalizedStore(adapter="test", memories=memories)


def _memory(memory_id: str, *source_refs: SourceRef, content: str = "A memory") -> NormalizedMemory:
    return NormalizedMemory(
        id=memory_id,
        content=content,
        source_refs=source_refs,
        provenance_status=ProvenanceStatus.DECLARED,
        active=True,
    )


def _transcripts() -> TranscriptSet:
    return TranscriptSet(
        transcripts=(
            Transcript(
                id="t1",
                turns=(TranscriptTurn(index=0, role="user", content="I prefer Python."),),
            ),
        ),
    )


def test_checker_implements_protocol_and_declares_frozen_defect_class() -> None:
    checker: Checker = OrphanedProvenanceChecker()

    assert checker.checker_id == "orphaned_provenance"
    assert checker.checker_version == "1.0"
    assert checker.defect_class is DefectClass.ORPHANED_PROVENANCE


def test_missing_transcript_emits_one_structural_finding() -> None:
    checker = OrphanedProvenanceChecker()
    result = checker.check(
        _store(_memory("m1", SourceRef(transcript_id="absent", turn_idx=4))),
        transcripts=_transcripts(),
    )

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.memory_ids == ("m1",)
    assert finding.confidence == 1.0
    assert tuple(item.kind for item in finding.evidence) == ("missing_transcript",)
    assert finding.evidence[0].data == {
        "source_ref_index": 0,
        "transcript_id": "absent",
    }
    assert result.cost.model_calls == 0
    assert result.cost.input_tokens == 0
    assert result.cost.output_tokens == 0
    assert result.stats.memories_scanned == 1
    assert result.stats.source_refs_scanned == 1
    assert result.stats.findings_emitted == 1


def test_missing_turn_emits_correct_evidence() -> None:
    result = OrphanedProvenanceChecker().check(
        _store(_memory("m1", SourceRef(transcript_id="t1", turn_idx=9))),
        transcripts=_transcripts(),
    )

    assert len(result.findings) == 1
    assert result.findings[0].evidence[0].kind == "missing_turn"
    assert result.findings[0].evidence[0].data == {
        "source_ref_index": 0,
        "transcript_id": "t1",
        "turn_idx": 9,
    }


def test_invalid_span_emits_length_without_transcript_content() -> None:
    result = OrphanedProvenanceChecker().check(
        _store(_memory("m1", SourceRef(transcript_id="t1", turn_idx=0, span=(0, 99)))),
        transcripts=_transcripts(),
    )

    assert len(result.findings) == 1
    evidence = result.findings[0].evidence[0]
    assert evidence.kind == "invalid_span"
    assert evidence.data == {
        "source_ref_index": 0,
        "transcript_id": "t1",
        "turn_idx": 0,
        "span": [0, 99],
        "turn_length": 16,
    }
    assert "I prefer Python" not in result.to_json()


@pytest.mark.parametrize(
    "source_ref",
    [
        SourceRef(transcript_id="t1", turn_idx=0, span=(0, 16)),
        SourceRef(transcript_id="t1", turn_idx=0),
        SourceRef(transcript_id="t1"),
    ],
    ids=("valid_span", "full_turn", "whole_transcript"),
)
def test_valid_reference_shapes_emit_no_finding(source_ref: SourceRef) -> None:
    result = OrphanedProvenanceChecker().check(
        _store(_memory("m1", source_ref)),
        transcripts=_transcripts(),
    )

    assert result.findings == ()


def test_unavailable_and_known_absent_provenance_emit_no_findings() -> None:
    store = _store(
        NormalizedMemory(id="unavailable", content="No exposed provenance"),
        NormalizedMemory(
            id="absent",
            content="Explicitly absent provenance",
            provenance_status=ProvenanceStatus.KNOWN_ABSENT,
        ),
    )

    result = OrphanedProvenanceChecker().check(store, transcripts=_transcripts())

    assert result.findings == ()
    assert result.stats.memories_scanned == 2
    assert result.stats.source_refs_scanned == 0


def test_missing_transcript_set_raises_input_error() -> None:
    store = _store(_memory("m1", SourceRef(transcript_id="absent")))

    with pytest.raises(CheckerInputError, match="requires a TranscriptSet"):
        OrphanedProvenanceChecker().check(store)


def test_multiple_broken_refs_produce_one_finding_with_sorted_evidence() -> None:
    store = _store(
        _memory(
            "m1",
            SourceRef(transcript_id="absent", turn_idx=2),
            SourceRef(transcript_id="t1", turn_idx=0, span=(17, 20)),
        )
    )

    result = OrphanedProvenanceChecker().check(store, transcripts=_transcripts())

    assert len(result.findings) == 1
    assert tuple(item.kind for item in result.findings[0].evidence) == (
        "missing_transcript",
        "invalid_span",
    )
    assert tuple(item.data["source_ref_index"] for item in result.findings[0].evidence) == (0, 1)
    assert result.stats.source_refs_scanned == 2


def test_multiple_memories_are_sorted_and_results_are_deterministic() -> None:
    store = _store(
        _memory("m2", SourceRef(transcript_id="t1", turn_idx=9)),
        _memory("m1", SourceRef(transcript_id="absent")),
    )
    checker = OrphanedProvenanceChecker()

    first = checker.check(store, transcripts=_transcripts())
    second = checker.check(store, transcripts=_transcripts())

    assert tuple(finding.memory_ids for finding in first.findings) == (("m1",), ("m2",))
    assert tuple(finding.finding_id for finding in first.findings) == tuple(
        finding.finding_id for finding in second.findings
    )
    assert first.to_json() == second.to_json()


def test_structurally_valid_but_unsupported_claim_is_not_flagged() -> None:
    store = _store(
        _memory(
            "m1",
            SourceRef(transcript_id="t1", turn_idx=0, span=(0, 16)),
            content="User prefers Rust.",
        )
    )

    result = OrphanedProvenanceChecker().check(store, transcripts=_transcripts())

    assert result.findings == ()


@pytest.mark.parametrize("subtype", ["missing_transcript", "missing_turn", "invalid_span"])
def test_checker_matches_part_two_mutation_gold_without_receiving_manifest(subtype: str) -> None:
    base_store = load_store("examples/mutation-store.json")
    transcripts = load_transcripts("examples/mutation-transcripts.json")
    mutation_result = mutate(
        base_store,
        MutationRequest(
            defect_class=DefectClass.ORPHANED_PROVENANCE,
            subtype=subtype,
            target_memory_id="preference-python",
            seed=42,
        ),
        transcripts,
    )

    checker_result = OrphanedProvenanceChecker().check(
        mutation_result.mutated_store,
        transcripts=transcripts,
    )
    manifest = mutation_result.manifest

    assert len(checker_result.findings) == 1
    assert checker_result.findings[0].memory_ids == manifest.gold_label.memory_ids
