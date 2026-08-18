from datetime import datetime

import pytest
from pydantic import ValidationError

from memlint.models import (
    MemoryScope,
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    Transcript,
    TranscriptSet,
    TranscriptTurn,
)


def test_valid_memory_construction() -> None:
    memory = NormalizedMemory(
        id="m1",
        content="User prefers Python.",
        created_at="2026-08-10T14:20:00+02:00",
        source_refs=(SourceRef(transcript_id="conversation-1", turn_idx=4, span=(0, 20)),),
        provenance_status=ProvenanceStatus.DECLARED,
        scope=MemoryScope(user_id="user-123"),
        active=True,
        embedding=(0.1, 0.2),
        raw={"backend_field": "preserved"},
    )

    assert memory.created_at == datetime.fromisoformat("2026-08-10T14:20:00+02:00")
    assert memory.source_refs[0].span == (0, 20)
    assert "raw" not in memory.semantic_dict()


def test_missing_information_remains_explicitly_unknown() -> None:
    memory = NormalizedMemory(id="m1", content="A memory")

    assert memory.created_at is None
    assert memory.updated_at is None
    assert memory.active is None
    assert memory.scope == MemoryScope()
    assert memory.source_refs == ()
    assert memory.provenance_status is ProvenanceStatus.UNAVAILABLE
    assert memory.embedding is None


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        NormalizedMemory(id="m1", content="A memory", created_at="2026-08-10T14:20:00")


def test_independently_available_updated_timestamp_is_preserved() -> None:
    memory = NormalizedMemory(
        id="m1",
        content="A memory",
        updated_at="2026-08-10T14:20:00+02:00",
    )

    assert memory.created_at is None
    assert memory.updated_at == datetime.fromisoformat("2026-08-10T14:20:00+02:00")


def test_all_provenance_states_have_distinct_meanings() -> None:
    declared = NormalizedMemory(
        id="m0",
        content="Declared memory",
        source_refs=(SourceRef(transcript_id="t1"),),
        provenance_status=ProvenanceStatus.DECLARED,
    )
    unavailable = NormalizedMemory(id="m1", content="A memory")
    absent = NormalizedMemory(
        id="m2",
        content="Another memory",
        provenance_status=ProvenanceStatus.KNOWN_ABSENT,
    )

    assert declared.source_refs == (SourceRef(transcript_id="t1"),)
    assert declared.provenance_status is ProvenanceStatus.DECLARED
    assert unavailable.source_refs == absent.source_refs == ()
    assert unavailable.provenance_status is ProvenanceStatus.UNAVAILABLE
    assert absent.provenance_status is ProvenanceStatus.KNOWN_ABSENT


def test_declared_provenance_requires_refs_and_refs_require_declared_status() -> None:
    with pytest.raises(ValidationError, match="requires at least one"):
        NormalizedMemory(
            id="m1",
            content="A memory",
            provenance_status=ProvenanceStatus.DECLARED,
        )
    with pytest.raises(ValidationError, match="require provenance_status"):
        NormalizedMemory(
            id="m2",
            content="A memory",
            source_refs=(SourceRef(transcript_id="t1"),),
        )


def test_malformed_span_and_scope_are_rejected() -> None:
    with pytest.raises(ValidationError, match="span end"):
        SourceRef(transcript_id="t1", turn_idx=0, span=(10, 2))
    with pytest.raises(ValidationError, match="span end"):
        SourceRef(transcript_id="t1", turn_idx=0, span=(5, 5))
    with pytest.raises(ValidationError, match="requires turn_idx"):
        SourceRef(transcript_id="t1", span=(0, 5))
    with pytest.raises(ValidationError, match="must not be blank"):
        MemoryScope(user_id=" ")


def test_store_rejects_duplicate_ids_and_supports_lookup() -> None:
    memory = NormalizedMemory(id="m1", content="A memory")
    store = NormalizedStore(adapter="file", memories=(memory,))

    assert len(store) == 1
    assert store.get("m1") == memory
    assert store.get("missing") is None
    with pytest.raises(ValidationError, match="duplicate"):
        NormalizedStore(adapter="file", memories=(memory, memory))


def test_transcript_models_enforce_order_and_connect_to_source_ref() -> None:
    transcript = Transcript(
        id="conversation-1",
        turns=(TranscriptTurn(index=4, role="user", content="User prefers Python."),),
    )
    transcripts = TranscriptSet(transcripts=(transcript,))
    ref = SourceRef(transcript_id="conversation-1", turn_idx=4, span=(0, 20))

    linked = transcripts.get(ref.transcript_id)
    assert linked is not None
    assert ref.span is not None
    assert linked.turns[0].content[slice(*ref.span)] == "User prefers Python."

    with pytest.raises(ValidationError, match="ordered"):
        Transcript(
            id="bad",
            turns=(
                TranscriptTurn(index=2, role="user", content="later"),
                TranscriptTurn(index=1, role="user", content="earlier"),
            ),
        )
