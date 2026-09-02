from __future__ import annotations

import pytest
from pydantic import ValidationError

from palintrace.models import NormalizedMemory, ProvenanceStatus, SourceRef
from palintrace.semantics import (
    SpeakerIdentityAdmissionError,
    SpeakerIdentityResolutionStatus,
    SpeakerIdentitySourceAssertion,
    SpeakerIdentitySourceAssertions,
    SpeakerIdentityTrust,
    resolve_speaker_identity,
)


def _assertion(
    *,
    transcript_id: str = "conversation-1",
    turn_idx: int | None = 0,
    trust: SpeakerIdentityTrust = SpeakerIdentityTrust.TRUSTED_EXPLICIT,
    source_system: str = "configured-file",
    source_reference: str = "participants.json#alice",
    principal_id: str | None = "user-123",
    speaker_label: str | None = "Alice",
) -> SpeakerIdentitySourceAssertion:
    return SpeakerIdentitySourceAssertion(
        transcript_id=transcript_id,
        turn_idx=turn_idx,
        trust=trust,
        source_system=source_system,
        source_reference=source_reference,
        principal_id=principal_id,
        speaker_label=speaker_label,
    )


def test_explicit_principal_and_display_label_compile_to_frozen_binding() -> None:
    sources = SpeakerIdentitySourceAssertions(assertions=(_assertion(),))
    bindings = sources.to_speaker_identity_bindings()

    assert bindings.bindings[0].transcript_id == "conversation-1"
    assert bindings.bindings[0].turn_idx == 0
    assert bindings.bindings[0].speaker_label == "Alice"


def test_operator_configured_label_does_not_require_principal_id() -> None:
    sources = SpeakerIdentitySourceAssertions(
        assertions=(
            _assertion(
                trust=SpeakerIdentityTrust.TRUSTED_CONFIGURED,
                source_system="operator-config",
                source_reference="speaker-bindings.yaml:4",
                principal_id=None,
            ),
        )
    )

    assert sources.to_speaker_identity_bindings().bindings[0].speaker_label == "Alice"


def test_trusted_explicit_requires_both_principal_and_label() -> None:
    with pytest.raises(ValidationError, match="requires principal_id"):
        _assertion(principal_id=None)
    with pytest.raises(ValidationError, match="turn_idx and speaker_label"):
        _assertion(speaker_label=None)


def test_stable_mem0_user_id_without_display_label_is_unavailable() -> None:
    source = _assertion(
        trust=SpeakerIdentityTrust.UNAVAILABLE,
        source_system="mem0",
        source_reference="memory-42",
        principal_id="user-123",
        speaker_label=None,
    )
    sources = SpeakerIdentitySourceAssertions(assertions=(source,))

    with pytest.raises(SpeakerIdentityAdmissionError, match="unavailable or ambiguous"):
        sources.to_speaker_identity_bindings()


def test_transcript_level_graphiti_episode_mapping_is_unavailable() -> None:
    source = _assertion(
        turn_idx=None,
        trust=SpeakerIdentityTrust.UNAVAILABLE,
        source_system="graphiti",
        source_reference="episode-7",
        principal_id=None,
        speaker_label=None,
    )

    with pytest.raises(SpeakerIdentityAdmissionError, match="conversation-1.*None"):
        SpeakerIdentitySourceAssertions(
            assertions=(source,)
        ).to_speaker_identity_bindings()


def test_ambiguous_principal_never_carries_or_compiles_a_label() -> None:
    with pytest.raises(ValidationError, match="cannot carry speaker_label"):
        _assertion(
            trust=SpeakerIdentityTrust.AMBIGUOUS,
            source_system="letta",
            source_reference="message-ambiguous",
            principal_id=None,
        )
    source = _assertion(
        trust=SpeakerIdentityTrust.AMBIGUOUS,
        source_system="letta",
        source_reference="message-ambiguous",
        principal_id=None,
        speaker_label=None,
    )
    with pytest.raises(SpeakerIdentityAdmissionError):
        SpeakerIdentitySourceAssertions(
            assertions=(source,)
        ).to_speaker_identity_bindings()


def test_multiple_users_and_assistant_turns_remain_turn_specific() -> None:
    sources = SpeakerIdentitySourceAssertions(
        assertions=(
            _assertion(
                turn_idx=0,
                source_system="letta",
                source_reference="message-user-a",
                principal_id="identity-a",
                speaker_label="Alice",
            ),
            _assertion(
                turn_idx=1,
                source_system="letta",
                source_reference="message-agent",
                principal_id="agent-7",
                speaker_label="Concierge",
            ),
            _assertion(
                turn_idx=2,
                source_system="letta",
                source_reference="message-user-b",
                principal_id="identity-b",
                speaker_label="Bob",
            ),
        )
    )
    bindings = sources.to_speaker_identity_bindings()

    assert [binding.speaker_label for binding in bindings.bindings] == [
        "Alice",
        "Concierge",
        "Bob",
    ]


def test_multiple_turns_from_same_principal_compile_deterministically() -> None:
    sources = SpeakerIdentitySourceAssertions(
        assertions=(
            _assertion(turn_idx=3, source_reference="message-3"),
            _assertion(turn_idx=1, source_reference="message-1"),
        )
    )

    assert [
        (binding.transcript_id, binding.turn_idx, binding.speaker_label)
        for binding in sources.to_speaker_identity_bindings().bindings
    ] == [
        ("conversation-1", 1, "Alice"),
        ("conversation-1", 3, "Alice"),
    ]


@pytest.mark.parametrize("conflict", ["label", "principal"])
def test_conflicting_explicit_mappings_fail_closed_without_leaking_values(
    conflict: str,
) -> None:
    second = (
        _assertion(
            source_system="letta",
            source_reference="message-0",
            principal_id="user-999",
            speaker_label="Bob",
        )
        if conflict == "label"
        else _assertion(
            source_system="letta",
            source_reference="message-0",
            principal_id="user-999",
        )
    )
    sources = SpeakerIdentitySourceAssertions(assertions=(_assertion(), second))

    with pytest.raises(SpeakerIdentityAdmissionError) as error:
        sources.to_speaker_identity_bindings()
    assert "Alice" not in str(error.value)
    assert "Bob" not in str(error.value)
    assert "user-123" not in str(error.value)
    assert "user-999" not in str(error.value)


def test_multi_speaker_memory_remains_conflict_after_safe_admission() -> None:
    sources = SpeakerIdentitySourceAssertions(
        assertions=(
            _assertion(turn_idx=0),
            _assertion(
                turn_idx=1,
                source_reference="participants.json#bob",
                principal_id="user-456",
                speaker_label="Bob",
            ),
        )
    )
    memory = NormalizedMemory(
        id="m1",
        content="A mixed-speaker claim.",
        source_refs=(
            SourceRef(transcript_id="conversation-1", turn_idx=0),
            SourceRef(transcript_id="conversation-1", turn_idx=1),
        ),
        provenance_status=ProvenanceStatus.DECLARED,
    )

    assert resolve_speaker_identity(
        memory, sources.to_speaker_identity_bindings()
    ).status is SpeakerIdentityResolutionStatus.CONFLICT


def test_source_assertions_are_immutable_canonical_and_deterministic() -> None:
    first = _assertion(turn_idx=2, source_reference="message-2")
    second = _assertion(turn_idx=0, source_reference="message-0")
    sources = SpeakerIdentitySourceAssertions(assertions=(first, second))

    assert [assertion.turn_idx for assertion in sources.assertions] == [0, 2]
    assert sources.to_json() == SpeakerIdentitySourceAssertions(
        assertions=(second, first)
    ).to_json()
    with pytest.raises(ValidationError, match="frozen"):
        sources.assertions = ()  # type: ignore[misc]


def test_duplicate_source_assertion_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate identity source assertion"):
        SpeakerIdentitySourceAssertions(
            assertions=(
                _assertion(),
                _assertion(speaker_label="Alicia"),
            )
        )


def test_contract_contains_no_claim_role_scope_raw_or_backend_mapper_fields() -> None:
    assert set(SpeakerIdentitySourceAssertion.model_fields) == {
        "transcript_id",
        "turn_idx",
        "trust",
        "source_system",
        "source_reference",
        "principal_id",
        "speaker_label",
    }
    assert {
        "content",
        "role",
        "scope",
        "raw",
        "metadata",
        "user_id",
        "agent_id",
    }.isdisjoint(SpeakerIdentitySourceAssertion.model_fields)
