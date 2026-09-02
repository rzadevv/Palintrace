"""Fail-closed admission of explicit speaker-identity source assertions."""

from __future__ import annotations

import json
from collections import defaultdict
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from palintrace.semantics.identity import (
    SpeakerIdentityBinding,
    SpeakerIdentityBindings,
)

_SCHEMA_VERSION = "0.1"
_MAX_IDENTITY_SOURCE_STRING_LENGTH = 256
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")

StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class SpeakerIdentityTrust(StrEnum):
    """Exhaustive trust classes for a possible turn-level identity source."""

    TRUSTED_EXPLICIT = "trusted_explicit"
    TRUSTED_CONFIGURED = "trusted_configured"
    UNAVAILABLE = "unavailable"
    AMBIGUOUS = "ambiguous"


class SpeakerIdentityAdmissionError(ValueError):
    """Source assertions cannot safely produce the frozen binding contract."""


def _bounded_single_line(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    if value != value.strip():
        raise ValueError(f"{field_name} must not have leading or trailing whitespace")
    if any(character in _LINE_BREAKS for character in value):
        raise ValueError(f"{field_name} must be a single line")
    if len(value) > _MAX_IDENTITY_SOURCE_STRING_LENGTH:
        raise ValueError(
            f"{field_name} must contain at most "
            f"{_MAX_IDENTITY_SOURCE_STRING_LENGTH} Unicode characters"
        )
    return value


class SpeakerIdentitySourceAssertion(BaseModel):
    """One caller/integration assertion without transcript or claim content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_id: StrictStr
    turn_idx: StrictNonNegativeInt | None = None
    trust: SpeakerIdentityTrust
    source_system: StrictStr
    source_reference: StrictStr
    principal_id: StrictStr | None = None
    speaker_label: StrictStr | None = None

    @field_validator(
        "transcript_id",
        "source_system",
        "source_reference",
        "principal_id",
    )
    @classmethod
    def source_strings_are_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_single_line(value, field_name="identity source string")

    @model_validator(mode="after")
    def fields_match_trust_class(self) -> SpeakerIdentitySourceAssertion:
        trusted = self.trust in {
            SpeakerIdentityTrust.TRUSTED_EXPLICIT,
            SpeakerIdentityTrust.TRUSTED_CONFIGURED,
        }
        if trusted:
            if self.turn_idx is None or self.speaker_label is None:
                raise ValueError(
                    "trusted speaker identity requires turn_idx and speaker_label"
                )
            if (
                self.trust is SpeakerIdentityTrust.TRUSTED_EXPLICIT
                and self.principal_id is None
            ):
                raise ValueError(
                    "trusted_explicit speaker identity requires principal_id"
                )
            SpeakerIdentityBinding(
                transcript_id=self.transcript_id,
                turn_idx=self.turn_idx,
                speaker_label=self.speaker_label,
            )
        elif self.speaker_label is not None:
            raise ValueError(
                "unavailable or ambiguous identity cannot carry speaker_label"
            )
        return self


def _assertion_key(assertion: SpeakerIdentitySourceAssertion) -> tuple[object, ...]:
    return (
        assertion.transcript_id,
        -1 if assertion.turn_idx is None else assertion.turn_idx,
        assertion.source_system,
        assertion.source_reference,
        assertion.trust.value,
        "" if assertion.principal_id is None else assertion.principal_id,
        "" if assertion.speaker_label is None else assertion.speaker_label,
    )


class SpeakerIdentitySourceAssertions(BaseModel):
    """Canonical source assertions that compile only when every turn is trustworthy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictStr = _SCHEMA_VERSION
    assertions: tuple[SpeakerIdentitySourceAssertion, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def supported_schema_version(cls, value: str) -> str:
        if value != _SCHEMA_VERSION:
            raise ValueError(f"unsupported identity source schema_version: {value!r}")
        return value

    @field_validator("assertions")
    @classmethod
    def assertions_are_unique_and_canonical(
        cls,
        value: tuple[SpeakerIdentitySourceAssertion, ...],
    ) -> tuple[SpeakerIdentitySourceAssertion, ...]:
        source_keys = [
            (
                assertion.transcript_id,
                assertion.turn_idx,
                assertion.source_system,
                assertion.source_reference,
            )
            for assertion in value
        ]
        if len(source_keys) != len(set(source_keys)):
            duplicates = sorted(
                {key for key in source_keys if source_keys.count(key) > 1},
                key=lambda item: (
                    item[0],
                    -1 if item[1] is None else item[1],
                    item[2],
                    item[3],
                ),
            )
            raise ValueError(f"duplicate identity source assertion keys: {duplicates!r}")
        return tuple(sorted(value, key=_assertion_key))

    def to_speaker_identity_bindings(self) -> SpeakerIdentityBindings:
        """Compile trusted unanimous assertions; fail closed on every other class."""

        inadmissible = tuple(
            assertion
            for assertion in self.assertions
            if assertion.trust
            not in {
                SpeakerIdentityTrust.TRUSTED_EXPLICIT,
                SpeakerIdentityTrust.TRUSTED_CONFIGURED,
            }
        )
        if inadmissible:
            coordinates = sorted(
                {
                    (assertion.transcript_id, assertion.turn_idx)
                    for assertion in inadmissible
                },
                key=lambda item: (
                    item[0],
                    -1 if item[1] is None else item[1],
                ),
            )
            raise SpeakerIdentityAdmissionError(
                "identity source assertions are unavailable or ambiguous at "
                f"coordinates {coordinates!r}"
            )

        by_coordinate: dict[
            tuple[str, int], list[SpeakerIdentitySourceAssertion]
        ] = defaultdict(list)
        for assertion in self.assertions:
            turn_idx = assertion.turn_idx
            if turn_idx is None:  # pragma: no cover - trusted model validation guarantees this
                raise SpeakerIdentityAdmissionError(
                    "trusted identity source assertion lacks turn_idx"
                )
            by_coordinate[(assertion.transcript_id, turn_idx)].append(assertion)

        bindings: list[SpeakerIdentityBinding] = []
        for coordinate in sorted(by_coordinate):
            assertions = by_coordinate[coordinate]
            labels = {
                assertion.speaker_label
                for assertion in assertions
                if assertion.speaker_label is not None
            }
            principals = {
                assertion.principal_id
                for assertion in assertions
                if assertion.principal_id is not None
            }
            if len(labels) != 1 or len(principals) > 1:
                raise SpeakerIdentityAdmissionError(
                    "trusted identity source assertions conflict at "
                    f"coordinate {coordinate!r}"
                )
            bindings.append(
                SpeakerIdentityBinding(
                    transcript_id=coordinate[0],
                    turn_idx=coordinate[1],
                    speaker_label=next(iter(labels)),
                )
            )
        return SpeakerIdentityBindings(bindings=tuple(bindings))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically without timestamps or backend data."""

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
