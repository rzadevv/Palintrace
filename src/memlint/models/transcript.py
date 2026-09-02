"""Minimal transcript representation for provenance-aware auditing."""

from __future__ import annotations

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeInt,
    field_validator,
)


class TranscriptTurn(BaseModel):
    """One ordered conversational turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: NonNegativeInt
    role: str
    content: str
    timestamp: AwareDatetime | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("role")
    @classmethod
    def role_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("role must not be blank")
        return value


class Transcript(BaseModel):
    """A transcript with stable identity and uniquely indexed turns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    turns: tuple[TranscriptTurn, ...]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transcript id must not be blank")
        return value

    @field_validator("turns")
    @classmethod
    def turn_indices_must_be_unique_and_ordered(
        cls, value: tuple[TranscriptTurn, ...]
    ) -> tuple[TranscriptTurn, ...]:
        indices = [turn.index for turn in value]
        if len(indices) != len(set(indices)):
            raise ValueError("transcript turn indices must be unique")
        if indices != sorted(indices):
            raise ValueError("transcript turns must be ordered by index")
        return value


class TranscriptSet(BaseModel):
    """A deterministically serializable collection of transcripts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.1"
    transcripts: tuple[Transcript, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def supported_schema_version(cls, value: str) -> str:
        if value != "0.1":
            raise ValueError(f"unsupported transcript schema_version: {value!r}")
        return value

    @field_validator("transcripts")
    @classmethod
    def transcript_ids_must_be_unique(cls, value: tuple[Transcript, ...]) -> tuple[Transcript, ...]:
        ids = [transcript.id for transcript in value]
        if len(ids) != len(set(ids)):
            raise ValueError("transcript IDs must be unique")
        return value

    def get(self, transcript_id: str) -> Transcript | None:
        return next(
            (transcript for transcript in self.transcripts if transcript.id == transcript_id), None
        )
