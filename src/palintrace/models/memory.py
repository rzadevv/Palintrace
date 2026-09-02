"""Backend-independent normalized memory records."""

from __future__ import annotations

from enum import StrEnum
from typing import cast

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    NonNegativeInt,
    StrictBool,
    field_validator,
    model_validator,
)


class ProvenanceStatus(StrEnum):
    """What the source or adapter declares about transcript provenance."""

    DECLARED = "declared"
    KNOWN_ABSENT = "known_absent"
    UNAVAILABLE = "unavailable"


class SourceRef(BaseModel):
    """A link to transcript evidence, with optional turn and character offsets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_id: str
    turn_idx: NonNegativeInt | None = None
    span: tuple[NonNegativeInt, NonNegativeInt] | None = None

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
    def span_requires_turn(self) -> SourceRef:
        if self.span is not None and self.turn_idx is None:
            raise ValueError("span requires turn_idx")
        return self


class MemoryScope(BaseModel):
    """Portable identity/session dimensions; unavailable dimensions remain ``None``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None

    @field_validator("user_id", "agent_id", "session_id")
    @classmethod
    def scope_values_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("scope identifiers must not be blank")
        return value


class NormalizedMemory(BaseModel):
    """Stable memory representation used across Palintrace.

    ``raw`` is intentionally frozen with the rest of the model and excluded by
    :meth:`semantic_dict`. It is for adapter diagnostics and reproduction only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    content: str
    created_at: AwareDatetime | None = None
    updated_at: AwareDatetime | None = None
    source_refs: tuple[SourceRef, ...] = ()
    provenance_status: ProvenanceStatus = ProvenanceStatus.UNAVAILABLE
    scope: MemoryScope = Field(default_factory=MemoryScope)
    active: StrictBool | None = None
    supersedes: tuple[str, ...] = ()
    embedding: tuple[FiniteFloat, ...] | None = None
    raw: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id must not be blank")
        return value

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value

    @field_validator("supersedes")
    @classmethod
    def supersedes_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("supersedes IDs must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("supersedes IDs must be unique")
        return value

    @field_validator("embedding")
    @classmethod
    def embedding_must_not_be_empty(
        cls, value: tuple[float, ...] | None
    ) -> tuple[float, ...] | None:
        if value == ():
            raise ValueError("embedding must be null or a non-empty vector")
        return value

    @model_validator(mode="after")
    def validate_cross_field_invariants(self) -> NormalizedMemory:
        if self.id in self.supersedes:
            raise ValueError("a memory cannot supersede itself")
        if self.source_refs and self.provenance_status is not ProvenanceStatus.DECLARED:
            raise ValueError("source_refs require provenance_status='declared'")
        if not self.source_refs and self.provenance_status is ProvenanceStatus.DECLARED:
            raise ValueError("declared provenance requires at least one source_ref")
        return self

    def semantic_dict(self) -> dict[str, JsonValue]:
        """Return the portable record fields, deliberately excluding ``raw``."""

        return cast(dict[str, JsonValue], self.model_dump(mode="json", exclude={"raw"}))
