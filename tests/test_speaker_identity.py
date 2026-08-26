from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from memlint.models import MemoryScope, NormalizedMemory, ProvenanceStatus, SourceRef
from memlint.semantics import (
    SpeakerIdentityBinding,
    SpeakerIdentityBindings,
    SpeakerIdentityError,
    SpeakerIdentityResolution,
    SpeakerIdentityResolutionStatus,
    build_speaker_grounded_premise,
    resolve_speaker_identity,
)


def _binding(
    transcript_id: str,
    turn_idx: int,
    speaker_label: str,
) -> SpeakerIdentityBinding:
    return SpeakerIdentityBinding(
        transcript_id=transcript_id,
        turn_idx=turn_idx,
        speaker_label=speaker_label,
    )


def _bindings(*bindings: SpeakerIdentityBinding) -> SpeakerIdentityBindings:
    return SpeakerIdentityBindings(bindings=bindings)


def _memory(*source_refs: SourceRef, content: str = "Lina prefers tea.") -> NormalizedMemory:
    return NormalizedMemory(
        id="memory-1",
        content=content,
        source_refs=source_refs,
        provenance_status=ProvenanceStatus.DECLARED,
    )


def test_valid_single_binding_is_typed_frozen_and_forbids_extra_fields() -> None:
    binding = _binding("conversation-A", 1, "Lina")
    assert binding.model_dump(mode="json") == {
        "transcript_id": "conversation-A",
        "turn_idx": 1,
        "speaker_label": "Lina",
    }

    with pytest.raises(ValidationError, match="Instance is frozen"):
        binding.speaker_label = "Other"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SpeakerIdentityBinding.model_validate(
            {
                "transcript_id": "conversation-A",
                "turn_idx": 1,
                "speaker_label": "Lina",
                "backend": "mem0",
            }
        )


def test_multiple_bindings_are_canonically_ordered() -> None:
    bindings = _bindings(
        _binding("conversation-B", 8, "Bob"),
        _binding("conversation-A", 4, "Alice"),
        _binding("conversation-A", 1, "Alice"),
    )
    assert [
        (binding.transcript_id, binding.turn_idx) for binding in bindings.bindings
    ] == [
        ("conversation-A", 1),
        ("conversation-A", 4),
        ("conversation-B", 8),
    ]
    assert bindings.get("conversation-A", 4) == _binding("conversation-A", 4, "Alice")
    assert bindings.get("conversation-A", 3) is None


def test_duplicate_binding_keys_are_rejected_even_when_labels_differ() -> None:
    with pytest.raises(ValidationError, match="duplicate speaker identity binding keys"):
        _bindings(
            _binding("conversation-A", 1, "Alice"),
            _binding("conversation-A", 1, "Bob"),
        )


@pytest.mark.parametrize("transcript_id", ["", " ", "\t"])
def test_blank_transcript_id_is_rejected(transcript_id: str) -> None:
    with pytest.raises(ValidationError, match="transcript_id must not be blank"):
        _binding(transcript_id, 0, "Alice")


def test_negative_and_non_integer_turn_indices_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _binding("conversation-A", -1, "Alice")
    for value in (True, 1.0, "1"):
        with pytest.raises(ValidationError):
            SpeakerIdentityBinding(
                transcript_id="conversation-A",
                turn_idx=value,  # type: ignore[arg-type]
                speaker_label="Alice",
            )


@pytest.mark.parametrize("speaker_label", ["", " ", "\t"])
def test_blank_speaker_label_is_rejected(speaker_label: str) -> None:
    with pytest.raises(ValidationError, match="speaker_label must not be blank"):
        _binding("conversation-A", 0, speaker_label)


def test_newline_and_carriage_return_speaker_labels_are_rejected() -> None:
    with pytest.raises(ValidationError, match="single line"):
        _binding("conversation-A", 0, "Alice\nBob")
    with pytest.raises(ValidationError, match="single line"):
        _binding("conversation-A", 0, "Alice\rBob")


@pytest.mark.parametrize("speaker_label", [" Alice", "Alice ", "\tAlice", "Alice\t"])
def test_leading_or_trailing_whitespace_is_rejected(speaker_label: str) -> None:
    with pytest.raises(ValidationError, match="leading or trailing whitespace"):
        _binding("conversation-A", 0, speaker_label)


def test_overlong_speaker_label_is_rejected() -> None:
    assert _binding("conversation-A", 0, "A" * 128).speaker_label == "A" * 128
    with pytest.raises(ValidationError, match="at most 128 Unicode characters"):
        _binding("conversation-A", 0, "A" * 129)


def test_unicode_human_names_are_preserved_exactly() -> None:
    label = "李雷 · Zoë Hernández"
    assert _binding("conversation-A", 0, label).speaker_label == label


def test_binding_json_is_deterministic_canonical_and_unicode_preserving() -> None:
    first = _bindings(
        _binding("β", 2, "李雷"),
        _binding("alpha", 3, "Zoë"),
    )
    second = _bindings(*reversed(first.bindings))

    assert first.to_json() == second.to_json()
    assert first.to_json() == first.to_json()
    assert "李雷" in first.to_json()
    assert first.to_json().endswith("\n")
    assert SpeakerIdentityBindings.model_validate_json(first.to_json()) == first
    assert list(json.loads(first.to_json())) == ["bindings", "schema_version"]


def test_resolution_status_enum_has_exactly_three_states() -> None:
    assert tuple(SpeakerIdentityResolutionStatus) == (
        SpeakerIdentityResolutionStatus.RESOLVED,
        SpeakerIdentityResolutionStatus.UNAVAILABLE,
        SpeakerIdentityResolutionStatus.CONFLICT,
    )


def test_one_bound_reference_resolves() -> None:
    resolution = resolve_speaker_identity(
        _memory(SourceRef(transcript_id="conversation-A", turn_idx=1)),
        _bindings(_binding("conversation-A", 1, "Alice")),
    )
    assert resolution == SpeakerIdentityResolution(
        status=SpeakerIdentityResolutionStatus.RESOLVED,
        speaker_label="Alice",
        source_turns=(("conversation-A", 1),),
    )


def test_multiple_references_with_the_same_speaker_resolve() -> None:
    resolution = resolve_speaker_identity(
        _memory(
            SourceRef(transcript_id="conversation-A", turn_idx=4),
            SourceRef(transcript_id="conversation-A", turn_idx=1),
        ),
        _bindings(
            _binding("conversation-A", 4, "Alice"),
            _binding("conversation-A", 1, "Alice"),
        ),
    )
    assert resolution.status is SpeakerIdentityResolutionStatus.RESOLVED
    assert resolution.speaker_label == "Alice"
    assert resolution.source_turns == (("conversation-A", 1), ("conversation-A", 4))


def test_duplicate_references_to_one_turn_are_canonicalized_and_spans_do_not_matter() -> None:
    resolution = resolve_speaker_identity(
        _memory(
            SourceRef(transcript_id="conversation-A", turn_idx=1),
            SourceRef(transcript_id="conversation-A", turn_idx=1, span=(0, 1)),
            SourceRef(transcript_id="conversation-A", turn_idx=1),
        ),
        _bindings(_binding("conversation-A", 1, "Alice")),
    )
    assert resolution.status is SpeakerIdentityResolutionStatus.RESOLVED
    assert resolution.source_turns == (("conversation-A", 1),)


def test_missing_binding_is_unavailable_and_does_not_return_partial_identity() -> None:
    resolution = resolve_speaker_identity(
        _memory(
            SourceRef(transcript_id="conversation-A", turn_idx=1),
            SourceRef(transcript_id="conversation-A", turn_idx=4),
        ),
        _bindings(_binding("conversation-A", 1, "Alice")),
    )
    assert resolution.status is SpeakerIdentityResolutionStatus.UNAVAILABLE
    assert resolution.speaker_label is None
    assert resolution.source_turns == (("conversation-A", 1), ("conversation-A", 4))


def test_source_reference_without_turn_index_is_unavailable() -> None:
    resolution = resolve_speaker_identity(
        _memory(SourceRef(transcript_id="conversation-A")),
        _bindings(_binding("conversation-A", 0, "Alice")),
    )
    assert resolution.status is SpeakerIdentityResolutionStatus.UNAVAILABLE
    assert resolution.speaker_label is None
    assert resolution.source_turns == ()


def test_multiple_distinct_speakers_conflict() -> None:
    resolution = resolve_speaker_identity(
        _memory(
            SourceRef(transcript_id="conversation-A", turn_idx=1),
            SourceRef(transcript_id="conversation-B", turn_idx=8),
        ),
        _bindings(
            _binding("conversation-A", 1, "Alice"),
            _binding("conversation-B", 8, "Bob"),
        ),
    )
    assert resolution.status is SpeakerIdentityResolutionStatus.CONFLICT
    assert resolution.speaker_label is None
    assert resolution.source_turns == (("conversation-A", 1), ("conversation-B", 8))


def test_resolution_never_uses_claim_scope_or_raw_fields() -> None:
    source_ref = SourceRef(transcript_id="conversation-A", turn_idx=1)
    bindings = _bindings(_binding("conversation-A", 1, "Alice"))
    claimed_alice = NormalizedMemory(
        id="alice-claim",
        content="Alice prefers tea.",
        source_refs=(source_ref,),
        provenance_status=ProvenanceStatus.DECLARED,
        scope=MemoryScope(user_id="Bob", agent_id="Alice", session_id="Lina"),
        raw={"speaker_name": "Mallory"},
    )
    claimed_bob = claimed_alice.model_copy(
        update={
            "id": "bob-claim",
            "content": "Bob prefers coffee.",
            "scope": MemoryScope(user_id="Different"),
            "raw": {"speaker_name": "Different"},
        }
    )
    assert resolve_speaker_identity(claimed_alice, bindings) == resolve_speaker_identity(
        claimed_bob,
        bindings,
    )


def test_grounded_premise_format_is_exact() -> None:
    resolution = SpeakerIdentityResolution(
        status=SpeakerIdentityResolutionStatus.RESOLVED,
        speaker_label="Lina",
        source_turns=(("conversation-A", 1),),
    )
    assert build_speaker_grounded_premise("I prefer tea.", resolution) == (
        "The speaker is Lina.\nI prefer tea."
    )


@pytest.mark.parametrize(
    "resolution",
    [
        SpeakerIdentityResolution(status=SpeakerIdentityResolutionStatus.UNAVAILABLE),
        SpeakerIdentityResolution(
            status=SpeakerIdentityResolutionStatus.CONFLICT,
            source_turns=(("conversation-A", 1), ("conversation-B", 8)),
        ),
    ],
)
def test_grounded_premise_rejects_unavailable_and_conflicting_identity(
    resolution: SpeakerIdentityResolution,
) -> None:
    with pytest.raises(SpeakerIdentityError, match="grounded premise requires resolved identity"):
        build_speaker_grounded_premise("I prefer tea.", resolution)
