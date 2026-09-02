from __future__ import annotations

import pytest
from pydantic import ValidationError

from palintrace.semantics import (
    PRIMARY_EVIDENCE_COMPOSITION_STYLE,
    ComposedEvidence,
    EvidenceCompositionStyle,
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


def test_composition_styles_have_exact_frozen_values() -> None:
    assert tuple(EvidenceCompositionStyle) == (
        EvidenceCompositionStyle.PLAIN,
        EvidenceCompositionStyle.ROLE_LABELED,
    )
    assert [style.value for style in EvidenceCompositionStyle] == [
        "plain",
        "role_labeled",
    ]


def test_primary_composition_style_and_default_are_plain() -> None:
    assert PRIMARY_EVIDENCE_COMPOSITION_STYLE is EvidenceCompositionStyle.PLAIN
    composed = compose_evidence((_segment(text="Exact."),))
    assert composed.style is EvidenceCompositionStyle.PLAIN
    assert composed.text == "Exact."


def test_empty_segments_are_unassessable_and_rejected() -> None:
    with pytest.raises(SemanticCompositionError, match="at least one"):
        compose_evidence((), style=EvidenceCompositionStyle.PLAIN)


def test_segments_are_canonically_ordered_by_structural_coordinates() -> None:
    segments = (
        _segment(transcript_id="t2", turn_idx=0, text="t2 turn 0"),
        _segment(transcript_id="t1", turn_idx=1, text="t1 turn 1"),
        _segment(transcript_id="t1", turn_idx=0, text="t1 turn 0"),
    )
    composed = compose_evidence(segments, style=EvidenceCompositionStyle.PLAIN)
    assert composed.text == "t1 turn 0\nt1 turn 1\nt2 turn 0"


def test_composition_is_invariant_to_input_tuple_order() -> None:
    first = _segment(transcript_id="t1", turn_idx=0, text="first")
    second = _segment(transcript_id="t1", turn_idx=1, text="second")
    forward = compose_evidence(
        (first, second),
        style=EvidenceCompositionStyle.ROLE_LABELED,
    )
    reverse = compose_evidence(
        (second, first),
        style=EvidenceCompositionStyle.ROLE_LABELED,
    )
    assert forward == reverse


def test_exact_duplicate_coordinates_are_deduplicated_across_declarations() -> None:
    first_declaration = _segment(source_ref_index=0, text="One fact.")
    second_declaration = _segment(source_ref_index=1, text="One fact.")
    composed = compose_evidence(
        (second_declaration, first_declaration),
        style=EvidenceCompositionStyle.PLAIN,
    )
    assert composed.text == "One fact."
    assert composed.segment_count == 2
    assert composed.unique_segment_count == 1


def test_same_text_at_different_coordinates_remains_distinct() -> None:
    composed = compose_evidence(
        (
            _segment(turn_idx=1, source_ref_index=1, text="Repeated fact."),
            _segment(turn_idx=0, source_ref_index=0, text="Repeated fact."),
        ),
        style=EvidenceCompositionStyle.PLAIN,
    )
    assert composed.text == "Repeated fact.\nRepeated fact."
    assert composed.segment_count == 2
    assert composed.unique_segment_count == 2


def test_plain_rendering_preserves_exact_text_without_normalization() -> None:
    composed = compose_evidence(
        (
            _segment(turn_idx=1, text="  Mixed CASE and punctuation?!  "),
            _segment(turn_idx=0, text="Café"),
        ),
        style=EvidenceCompositionStyle.PLAIN,
    )
    assert composed.text == "Café\n  Mixed CASE and punctuation?!  "


def test_role_labeled_rendering_preserves_exact_roles_and_text() -> None:
    composed = compose_evidence(
        (
            _segment(turn_idx=1, role="assistant", text="Understood."),
            _segment(turn_idx=0, role="USER", text=" Exact text. "),
        ),
        style=EvidenceCompositionStyle.ROLE_LABELED,
    )
    assert composed.text == "USER:  Exact text. \nassistant: Understood."


def test_span_segment_uses_only_the_already_resolved_text() -> None:
    composed = compose_evidence(
        (
            _segment(
                span=(8, 15),
                text="resolved",
            ),
        ),
        style=EvidenceCompositionStyle.PLAIN,
    )
    assert composed.text == "resolved"


def test_composition_does_not_filter_any_declared_role() -> None:
    roles = ("user", "assistant", "system", "tool", "custom")
    segments = tuple(
        _segment(turn_idx=index, source_ref_index=index, role=role, text=str(index))
        for index, role in enumerate(roles)
    )
    composed = compose_evidence(
        segments,
        style=EvidenceCompositionStyle.ROLE_LABELED,
    )
    assert composed.text == "\n".join(
        f"{role}: {index}" for index, role in enumerate(roles)
    )
    assert composed.segment_count == len(roles)
    assert composed.unique_segment_count == len(roles)


def test_repeated_composition_is_deterministic() -> None:
    segments = (
        _segment(turn_idx=1, text="second"),
        _segment(turn_idx=0, text="first"),
    )
    first = compose_evidence(segments, style=EvidenceCompositionStyle.PLAIN)
    second = compose_evidence(segments, style=EvidenceCompositionStyle.PLAIN)
    assert first == second


def test_composed_evidence_is_frozen_strict_and_forbids_extra_fields() -> None:
    composed = ComposedEvidence(
        style=EvidenceCompositionStyle.PLAIN,
        text="evidence",
        segment_count=2,
        unique_segment_count=1,
    )
    with pytest.raises(ValidationError):
        composed.text = "changed"
    with pytest.raises(ValidationError):
        ComposedEvidence.model_validate(
            {
                "style": "plain",
                "text": "evidence",
                "segment_count": 1,
                "unique_segment_count": 1,
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError):
        ComposedEvidence(
            style=EvidenceCompositionStyle.PLAIN,
            text="evidence",
            segment_count=1,
            unique_segment_count=2,
        )
