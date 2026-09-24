"""Deterministic composition of resolved evidence into one semantic premise."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from palintrace.semantics.models import EvidenceSegment

_StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]


class ComposedEvidence(BaseModel):
    """One deterministic premise plus transparent segment counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    segment_count: _StrictPositiveInt
    unique_segment_count: _StrictPositiveInt

    @model_validator(mode="after")
    def unique_count_cannot_exceed_input_count(self) -> ComposedEvidence:
        if self.unique_segment_count > self.segment_count:
            raise ValueError("unique_segment_count cannot exceed segment_count")
        return self


class SemanticCompositionError(ValueError):
    """Resolved evidence cannot be composed under the requested policy."""


def _span_key(span: tuple[int, int] | None) -> tuple[int, int]:
    return (-1, -1) if span is None else span


def _canonical_key(segment: EvidenceSegment) -> tuple[object, ...]:
    return (
        segment.transcript_id,
        segment.turn_idx,
        *_span_key(segment.span),
        segment.source_ref_index,
        segment.role,
        segment.text,
    )


def _duplicate_key(segment: EvidenceSegment) -> tuple[object, ...]:
    return (
        segment.transcript_id,
        segment.turn_idx,
        segment.span,
        segment.text,
        segment.role,
    )


def canonical_unique_evidence_segments(
    segments: tuple[EvidenceSegment, ...],
) -> tuple[EvidenceSegment, ...]:
    """Return the exact canonical unique segments used by composition."""

    if not segments:
        raise SemanticCompositionError("at least one resolved evidence segment is required")

    unique_segments: list[EvidenceSegment] = []
    seen: set[tuple[object, ...]] = set()
    for segment in sorted(segments, key=_canonical_key):
        duplicate_key = _duplicate_key(segment)
        if duplicate_key in seen:
            continue
        seen.add(duplicate_key)
        unique_segments.append(segment)
    return tuple(unique_segments)


def compose_evidence(
    segments: tuple[EvidenceSegment, ...],
) -> ComposedEvidence:
    """Canonicalize, exactly deduplicate, and join resolved evidence texts by newline."""

    unique_segments = canonical_unique_evidence_segments(segments)
    return ComposedEvidence(
        text="\n".join(segment.text for segment in unique_segments),
        segment_count=len(segments),
        unique_segment_count=len(unique_segments),
    )
