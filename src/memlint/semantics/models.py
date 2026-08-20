"""Provider-independent semantic judgments and resolved evidence models."""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
    field_validator,
    model_validator,
)


class SemanticRelation(StrEnum):
    """Generic directional relation between a premise and hypothesis."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class SemanticUsage(BaseModel):
    """Deterministic judge usage counters without pricing or timing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_calls: NonNegativeInt = 0
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0


class SemanticJudgment(BaseModel):
    """One provider-independent semantic-relation decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation: SemanticRelation
    score: FiniteFloat = Field(ge=0.0, le=1.0)
    usage: SemanticUsage = Field(default_factory=SemanticUsage)


class EvidenceIssueKind(StrEnum):
    """Structural failure modes shared with provenance resolution."""

    MISSING_TRANSCRIPT = "missing_transcript"
    MISSING_TURN = "missing_turn"
    INVALID_SPAN = "invalid_span"


class EvidenceSegment(BaseModel):
    """Exact transcript text resolved from one declared source coordinate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_ref_index: NonNegativeInt
    transcript_id: str
    turn_idx: NonNegativeInt
    role: str
    span: tuple[NonNegativeInt, NonNegativeInt] | None = None
    text: str

    @field_validator("transcript_id", "role")
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transcript_id and role must not be blank")
        return value

    @field_validator("span")
    @classmethod
    def span_end_must_follow_start(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is not None and value[1] <= value[0]:
            raise ValueError("span end must be greater than span start")
        return value


class EvidenceResolutionIssue(BaseModel):
    """Minimal structural coordinates for one unresolvable source reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: EvidenceIssueKind
    source_ref_index: NonNegativeInt
    transcript_id: str
    turn_idx: NonNegativeInt | None = None
    span: tuple[NonNegativeInt, NonNegativeInt] | None = None
    turn_length: NonNegativeInt | None = None

    @field_validator("transcript_id")
    @classmethod
    def transcript_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transcript_id must not be blank")
        return value

    @field_validator("span")
    @classmethod
    def span_end_must_follow_start(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is not None and value[1] <= value[0]:
            raise ValueError("span end must be greater than span start")
        return value

    @model_validator(mode="after")
    def fields_must_match_issue_kind(self) -> EvidenceResolutionIssue:
        if self.kind is EvidenceIssueKind.MISSING_TRANSCRIPT:
            if self.turn_idx is not None or self.span is not None or self.turn_length is not None:
                raise ValueError("missing_transcript accepts only transcript coordinates")
        elif self.kind is EvidenceIssueKind.MISSING_TURN:
            if self.turn_idx is None or self.span is not None or self.turn_length is not None:
                raise ValueError("missing_turn requires only a turn index")
        elif self.turn_idx is None or self.span is None or self.turn_length is None:
            raise ValueError("invalid_span requires turn, span, and turn_length")
        elif self.span[1] <= self.turn_length:
            raise ValueError("invalid_span end must exceed turn_length")
        return self


def _span_key(span: tuple[int, int] | None) -> tuple[int, int]:
    return (-1, -1) if span is None else span


def _segment_key(segment: EvidenceSegment) -> tuple[object, ...]:
    return (
        segment.transcript_id,
        segment.turn_idx,
        *_span_key(segment.span),
        segment.source_ref_index,
        segment.role,
        segment.text,
    )


def _issue_key(issue: EvidenceResolutionIssue) -> tuple[object, ...]:
    return (
        issue.transcript_id,
        -1 if issue.turn_idx is None else issue.turn_idx,
        *_span_key(issue.span),
        issue.source_ref_index,
        issue.kind.value,
        -1 if issue.turn_length is None else issue.turn_length,
    )


class EvidenceResolution(BaseModel):
    """Deterministically ordered transcript segments and structural issues."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segments: tuple[EvidenceSegment, ...] = ()
    issues: tuple[EvidenceResolutionIssue, ...] = ()

    @field_validator("segments")
    @classmethod
    def segments_are_canonical(
        cls, value: tuple[EvidenceSegment, ...]
    ) -> tuple[EvidenceSegment, ...]:
        return tuple(sorted(value, key=_segment_key))

    @field_validator("issues")
    @classmethod
    def issues_are_canonical(
        cls, value: tuple[EvidenceResolutionIssue, ...]
    ) -> tuple[EvidenceResolutionIssue, ...]:
        return tuple(sorted(value, key=_issue_key))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically without execution metadata."""

        text = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )
        if indent is not None:
            text += "\n"
        return text
