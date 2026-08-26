"""Explicit speaker-identity bindings and deterministic resolution."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from memlint.models import NormalizedMemory

_SCHEMA_VERSION = "0.1"
_MAX_SPEAKER_LABEL_LENGTH = 128
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")

StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
SourceTurn: TypeAlias = tuple[StrictStr, StrictNonNegativeInt]


def _validate_speaker_label(value: str) -> str:
    if not value.strip():
        raise ValueError("speaker_label must not be blank")
    if value != value.strip():
        raise ValueError("speaker_label must not have leading or trailing whitespace")
    if any(character in _LINE_BREAKS for character in value):
        raise ValueError("speaker_label must be a single line")
    if len(value) > _MAX_SPEAKER_LABEL_LENGTH:
        raise ValueError(
            f"speaker_label must contain at most {_MAX_SPEAKER_LABEL_LENGTH} Unicode characters"
        )
    return value


class SpeakerIdentityBinding(BaseModel):
    """Trusted audit input naming the speaker of one transcript turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_id: StrictStr
    turn_idx: StrictNonNegativeInt
    speaker_label: StrictStr

    @field_validator("transcript_id")
    @classmethod
    def transcript_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transcript_id must not be blank")
        return value

    @field_validator("speaker_label")
    @classmethod
    def speaker_label_must_be_safe(cls, value: str) -> str:
        return _validate_speaker_label(value)


class SpeakerIdentityBindings(BaseModel):
    """Versioned, canonical collection of explicit turn-to-speaker bindings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictStr = _SCHEMA_VERSION
    bindings: tuple[SpeakerIdentityBinding, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def supported_schema_version(cls, value: str) -> str:
        if value != _SCHEMA_VERSION:
            raise ValueError(f"unsupported speaker identity schema_version: {value!r}")
        return value

    @field_validator("bindings")
    @classmethod
    def binding_keys_must_be_unique_and_canonical(
        cls,
        value: tuple[SpeakerIdentityBinding, ...],
    ) -> tuple[SpeakerIdentityBinding, ...]:
        keys = [(binding.transcript_id, binding.turn_idx) for binding in value]
        if len(keys) != len(set(keys)):
            duplicates = sorted({key for key in keys if keys.count(key) > 1})
            raise ValueError(f"duplicate speaker identity binding keys: {duplicates!r}")
        return tuple(sorted(value, key=lambda item: (item.transcript_id, item.turn_idx)))

    def get(self, transcript_id: str, turn_idx: int) -> SpeakerIdentityBinding | None:
        """Return the explicit binding for an exact normalized source coordinate."""

        return next(
            (
                binding
                for binding in self.bindings
                if binding.transcript_id == transcript_id and binding.turn_idx == turn_idx
            ),
            None,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically without execution or backend metadata."""

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


class SpeakerIdentityResolutionStatus(StrEnum):
    """Exhaustive outcomes for explicit source-turn identity resolution."""

    RESOLVED = "resolved"
    UNAVAILABLE = "unavailable"
    CONFLICT = "conflict"


class SpeakerIdentityResolution(BaseModel):
    """One resolved label or a non-guessing identity capability failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SpeakerIdentityResolutionStatus
    speaker_label: StrictStr | None = None
    source_turns: tuple[SourceTurn, ...] = ()

    @field_validator("speaker_label")
    @classmethod
    def optional_speaker_label_must_be_safe(cls, value: str | None) -> str | None:
        return None if value is None else _validate_speaker_label(value)

    @field_validator("source_turns")
    @classmethod
    def source_turns_must_be_canonical(
        cls,
        value: tuple[SourceTurn, ...],
    ) -> tuple[SourceTurn, ...]:
        for transcript_id, _turn_idx in value:
            if not transcript_id.strip():
                raise ValueError("source turn transcript_id must not be blank")
        return tuple(sorted(set(value), key=lambda item: (item[0], item[1])))

    @model_validator(mode="after")
    def label_must_match_status(self) -> SpeakerIdentityResolution:
        if self.status is SpeakerIdentityResolutionStatus.RESOLVED:
            if self.speaker_label is None:
                raise ValueError("resolved speaker identity requires speaker_label")
            if not self.source_turns:
                raise ValueError("resolved speaker identity requires at least one source turn")
        elif self.speaker_label is not None:
            raise ValueError("unavailable or conflicting speaker identity cannot carry a label")
        if self.status is SpeakerIdentityResolutionStatus.CONFLICT and len(self.source_turns) < 2:
            raise ValueError("conflicting speaker identity requires at least two source turns")
        return self


class SpeakerIdentityError(ValueError):
    """Explicit identity input cannot support speaker-grounded composition."""


def resolve_speaker_identity(
    memory: NormalizedMemory,
    bindings: SpeakerIdentityBindings,
) -> SpeakerIdentityResolution:
    """Resolve a memory's source turns using only explicit identity bindings."""

    has_reference_without_turn = any(
        source_ref.turn_idx is None for source_ref in memory.source_refs
    )
    source_turns = tuple(
        sorted(
            {
                (source_ref.transcript_id, source_ref.turn_idx)
                for source_ref in memory.source_refs
                if source_ref.turn_idx is not None
            }
        )
    )

    if has_reference_without_turn or not source_turns:
        return SpeakerIdentityResolution(
            status=SpeakerIdentityResolutionStatus.UNAVAILABLE,
            source_turns=source_turns,
        )

    resolved_bindings = tuple(
        bindings.get(transcript_id, turn_idx) for transcript_id, turn_idx in source_turns
    )
    if any(binding is None for binding in resolved_bindings):
        return SpeakerIdentityResolution(
            status=SpeakerIdentityResolutionStatus.UNAVAILABLE,
            source_turns=source_turns,
        )

    speaker_labels = {
        binding.speaker_label for binding in resolved_bindings if binding is not None
    }
    if len(speaker_labels) == 1:
        return SpeakerIdentityResolution(
            status=SpeakerIdentityResolutionStatus.RESOLVED,
            speaker_label=next(iter(speaker_labels)),
            source_turns=source_turns,
        )
    return SpeakerIdentityResolution(
        status=SpeakerIdentityResolutionStatus.CONFLICT,
        source_turns=source_turns,
    )


def build_speaker_grounded_premise(
    evidence_text: str,
    resolution: SpeakerIdentityResolution,
) -> str:
    """Add the exact frozen Part 6F speaker prefix to resolvable evidence."""

    speaker_label = resolution.speaker_label
    if (
        resolution.status is not SpeakerIdentityResolutionStatus.RESOLVED
        or speaker_label is None
    ):
        raise SpeakerIdentityError(
            f"speaker identity is {resolution.status.value}; "
            "grounded premise requires resolved identity"
        )
    return f"The speaker is {speaker_label}.\n{evidence_text}"
