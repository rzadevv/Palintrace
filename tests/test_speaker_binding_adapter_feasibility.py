from __future__ import annotations

from memlint.adapters.file import normalize_file_record
from memlint.adapters.graphiti import normalize_graphiti_record
from memlint.adapters.letta import normalize_letta_record
from memlint.adapters.mem0 import normalize_mem0_record
from memlint.models import (
    MemoryScope,
    NormalizedMemory,
    ProvenanceStatus,
    SourceRef,
    Transcript,
    TranscriptSet,
    TranscriptTurn,
)
from memlint.semantics import (
    SpeakerIdentityBinding,
    SpeakerIdentityBindings,
    SpeakerIdentityResolutionStatus,
    SpeakerIdentitySourceAssertion,
    SpeakerIdentitySourceAssertions,
    SpeakerIdentityTrust,
    resolve_speaker_identity,
)


def test_file_memory_fields_do_not_automatically_become_speaker_identity() -> None:
    memory = normalize_file_record(
        {
            "id": "file-memory",
            "content": "Alice uses a mechanical keyboard.",
            "user_id": "user-123",
            "source_refs": [{"transcript_id": "conversation-1", "turn_idx": 0}],
            "speaker_label": "Alice",
            "principal_id": "user-123",
        },
        source_format="json",
    )

    assert memory.scope.user_id == "user-123"
    assert memory.raw["source_metadata"] == {
        "speaker_label": "Alice",
        "principal_id": "user-123",
    }
    assert resolve_speaker_identity(
        memory, SpeakerIdentityBindings()
    ).status is SpeakerIdentityResolutionStatus.UNAVAILABLE

    configured = SpeakerIdentitySourceAssertions(
        assertions=(
            SpeakerIdentitySourceAssertion(
                transcript_id="conversation-1",
                turn_idx=0,
                trust=SpeakerIdentityTrust.TRUSTED_CONFIGURED,
                source_system="operator-file",
                source_reference="participants.yaml:2",
                principal_id="user-123",
                speaker_label="Alice",
            ),
        )
    ).to_speaker_identity_bindings()
    assert resolve_speaker_identity(
        memory, configured
    ).status is SpeakerIdentityResolutionStatus.RESOLVED


def test_mem0_scope_and_metadata_lack_documented_turn_label_mapping() -> None:
    memory = normalize_mem0_record(
        {
            "id": "mem0-memory",
            "memory": "Alice uses a mechanical keyboard.",
            "user_id": "user-123",
            "agent_id": "agent-9",
            "run_id": "run-4",
            "metadata": {"display_name": "Alice"},
        }
    )

    assert memory.scope == MemoryScope(
        user_id="user-123",
        agent_id="agent-9",
        session_id="run-4",
    )
    assert memory.source_refs == ()
    assert memory.provenance_status is ProvenanceStatus.UNAVAILABLE
    assert memory.raw["metadata"] == {"display_name": "Alice"}


def test_letta_block_context_and_raw_sender_fields_do_not_create_turn_identity() -> None:
    memory = normalize_letta_record(
        {
            "id": "block-1",
            "memory_type": "core",
            "value": "Alice uses a mechanical keyboard.",
            "sender_id": "identity-1",
            "name": "Alice",
        },
        scope=MemoryScope(user_id="user-123", agent_id="agent-9"),
    )

    assert memory.scope == MemoryScope(user_id="user-123", agent_id="agent-9")
    assert memory.source_refs == ()
    assert memory.provenance_status is ProvenanceStatus.UNAVAILABLE
    assert memory.raw["sender_id"] == "identity-1"
    assert memory.raw["name"] == "Alice"


def test_graphiti_episode_mapping_is_provenance_not_speaker_attribution() -> None:
    memory = normalize_graphiti_record(
        {
            "uuid": "edge-1",
            "fact": "Alice uses a mechanical keyboard.",
            "group_id": "customer-team",
            "episodes": ["episode-1"],
        },
        scope=MemoryScope(user_id="user-123"),
        episode_transcript_map={"episode-1": "conversation-1"},
    )
    bindings = SpeakerIdentityBindings(
        bindings=(
            SpeakerIdentityBinding(
                transcript_id="conversation-1",
                turn_idx=0,
                speaker_label="Alice",
            ),
        )
    )

    assert memory.source_refs == (SourceRef(transcript_id="conversation-1"),)
    assert memory.provenance_status is ProvenanceStatus.DECLARED
    assert memory.raw["group_id"] == "customer-team"
    assert resolve_speaker_identity(
        memory, bindings
    ).status is SpeakerIdentityResolutionStatus.UNAVAILABLE


def test_role_and_arbitrary_transcript_metadata_are_not_identity_bindings() -> None:
    memory = NormalizedMemory(
        id="m1",
        content="Alice uses a mechanical keyboard.",
        source_refs=(SourceRef(transcript_id="conversation-1", turn_idx=0),),
        provenance_status=ProvenanceStatus.DECLARED,
    )
    transcripts = TranscriptSet(
        transcripts=(
            Transcript(
                id="conversation-1",
                turns=(
                    TranscriptTurn(
                        index=0,
                        role="user",
                        content="I use a mechanical keyboard.",
                        metadata={
                            "principal_id": "user-123",
                            "speaker_label": "Alice",
                        },
                    ),
                ),
            ),
        )
    )

    assert transcripts.transcripts[0].turns[0].role == "user"
    assert resolve_speaker_identity(
        memory, SpeakerIdentityBindings()
    ).status is SpeakerIdentityResolutionStatus.UNAVAILABLE
