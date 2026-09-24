from __future__ import annotations

import pytest
from pydantic import ValidationError

from palintrace.semantics import (
    ComposedEvidence,
    EvidenceSegment,
    SemanticCompositionError,
    compose_evidence,
)


def _segment(
    *,
    source_ref_index: int = 0,
    transcript_id: str = "t1",
    turn_idx: int = 0,
    role: str = "user",
    span: tuple[int, int] | None = None,
    text: str = "Exact evidence.",
) -> EvidenceSegment:
    return EvidenceSegment(
        source_ref_index=source_ref_index,
        transcript_id=transcript_id,
        turn_idx=turn_idx,
        role=role,
        span=span,
        text=text,
    )


def test_empty_segments_are_unassessable_and_rejected() -> None:
    with pytest.raises(SemanticCompositionError, match="at least one"):
        compose_evidence(())


def test_segments_are_canonically_ordered_by_structural_coordinates() -> None:
    segments = (
        _segment(transcript_id="t2", turn_idx=0, text="t2 turn 0"),
        _segment(transcript_id="t1", turn_idx=1, text="t1 turn 1"),
        _segment(transcript_id="t1", turn_idx=0, text="t1 turn 0"),
    )
    composed = compose_evidence(segments)
    assert composed.text == "t1 turn 0\nt1 turn 1\nt2 turn 0"


def test_composition_is_invariant_to_input_tuple_order() -> None:
    first = _segment(transcript_id="t1", turn_idx=0, text="first")
    second = _segment(transcript_id="t1", turn_idx=1, text="second")
    assert compose_evidence((first, second)) == compose_evidence((second, first))


def test_exact_duplicate_coordinates_are_deduplicated_across_declarations() -> None:
    first_declaration = _segment(source_ref_index=0, text="One fact.")
    second_declaration = _segment(source_ref_index=1, text="One fact.")
    composed = compose_evidence((second_declaration, first_declaration))
    assert composed.text == "One fact."
    assert composed.segment_count == 2
    assert composed.unique_segment_count == 1


def test_same_text_at_different_coordinates_remains_distinct() -> None:
    composed = compose_evidence(
        (
            _segment(turn_idx=1, source_ref_index=1, text="Repeated fact."),
            _segment(turn_idx=0, source_ref_index=0, text="Repeated fact."),
        )
    )
    assert composed.text == "Repeated fact.\nRepeated fact."
    assert composed.segment_count == 2
    assert composed.unique_segment_count == 2


def test_rendering_preserves_exact_text_without_normalization() -> None:
    composed = compose_evidence(
        (
            _segment(turn_idx=1, text="  Mixed CASE and punctuation?!  "),
            _segment(turn_idx=0, text="Café"),
        )
    )
    assert composed.text == "Café\n  Mixed CASE and punctuation?!  "


def test_span_segment_uses_only_the_already_resolved_text() -> None:
    composed = compose_evidence((_segment(span=(8, 15), text="resolved"),))
    assert composed.text == "resolved"


def test_composition_does_not_filter_any_declared_role() -> None:
    roles = ("user", "assistant", "system", "tool", "custom")
    segments = tuple(
        _segment(turn_idx=index, source_ref_index=index, role=role, text=str(index))
        for index, role in enumerate(roles)
    )
    composed = compose_evidence(segments)
    assert composed.text == "\n".join(str(index) for index in range(len(roles)))
    assert composed.segment_count == len(roles)
    assert composed.unique_segment_count == len(roles)


def test_repeated_composition_is_deterministic() -> None:
    segments = (
        _segment(turn_idx=1, text="second"),
        _segment(turn_idx=0, text="first"),
    )
    assert compose_evidence(segments) == compose_evidence(segments)


def test_composed_evidence_is_frozen_strict_and_forbids_extra_fields() -> None:
    composed = ComposedEvidence(text="evidence", segment_count=2, unique_segment_count=1)
    with pytest.raises(ValidationError):
        composed.text = "changed"
    with pytest.raises(ValidationError):
        ComposedEvidence.model_validate(
            {
                "text": "evidence",
                "segment_count": 1,
                "unique_segment_count": 1,
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError):
        ComposedEvidence(text="evidence", segment_count=1, unique_segment_count=2)
