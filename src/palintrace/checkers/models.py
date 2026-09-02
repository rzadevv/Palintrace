"""Validated deterministic checker findings and result envelopes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    NonNegativeInt,
    field_serializer,
    field_validator,
    model_validator,
)

from palintrace.taxonomy import DefectClass

CHECKER_RESULT_SCHEMA_VERSION = "0.3"

Severity = Literal["info", "warning", "error"]

_BUILTIN_RULE_METADATA: Mapping[
    str, tuple[DefectClass, str, str, Severity]
] = MappingProxyType(
    {
        "orphaned_provenance": (
            DefectClass.ORPHANED_PROVENANCE,
            "memory.provenance.orphaned",
            "1.0.0",
            "error",
        ),
        "redundancy_bloat": (
            DefectClass.REDUNDANCY_BLOAT,
            "memory.duplication.exact",
            "1.0.0",
            "warning",
        ),
        "stale_active": (
            DefectClass.STALE_ACTIVE,
            "memory.state.explicit-stale",
            "1.0.0",
            "error",
        ),
        "privacy_scope_violation": (
            DefectClass.PRIVACY_SCOPE_VIOLATION,
            "memory.scope.prohibited-exact-replica",
            "1.0.0",
            "error",
        ),
        "unsupported_claim": (
            DefectClass.UNSUPPORTED_CLAIM,
            "memory.claim.unsupported",
            "1.0.0",
            "error",
        ),
        "unsupported_claim_identity_grounded": (
            DefectClass.UNSUPPORTED_CLAIM,
            "memory.claim.unsupported",
            "1.0.0",
            "error",
        ),
        "retrieval_shadowing": (
            DefectClass.RETRIEVAL_SHADOWING,
            "memory.retrieval.shadowing",
            "1.0.0",
            "error",
        ),
    }
)

_RULE_ID_PATTERN = re.compile(
    r"memory\.[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+(?:-[a-z0-9]+)*"
)
_RULE_VERSION_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
)


def _freeze_json(value: JsonValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return cast(JsonValue, value)


class EvidenceItem(BaseModel):
    """Minimal machine-readable and human-readable support for a finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    message: str
    data: Mapping[str, JsonValue] = Field(default_factory=dict, validate_default=True)

    @field_validator("kind", "message")
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence kind and message must not be blank")
        return value

    @field_validator("data")
    @classmethod
    def data_must_be_strict_json(
        cls, value: Mapping[str, JsonValue]
    ) -> Mapping[str, JsonValue]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("evidence data must contain strict JSON values") from error
        return cast(Mapping[str, JsonValue], _freeze_json(dict(value)))

    @field_serializer("data")
    def serialize_data(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        thawed = _thaw_json(value)
        if not isinstance(thawed, dict):  # pragma: no cover - field validation guarantees this
            raise TypeError("evidence data must be a JSON object")
        return thawed


def _evidence_identity(item: EvidenceItem) -> dict[str, JsonValue]:
    serialized = item.model_dump(mode="json")
    return {"data": cast(JsonValue, serialized["data"]), "kind": item.kind}


def _evidence_sort_key(item: EvidenceItem) -> tuple[str, str]:
    identity = json.dumps(
        _evidence_identity(item),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return identity, item.message


class Finding(BaseModel):
    """One typed defect finding over one or more normalized memories."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_id: str
    defect_class: DefectClass
    memory_ids: tuple[str, ...]
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    evidence: tuple[EvidenceItem, ...]

    @field_validator("finding_id")
    @classmethod
    def finding_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("finding_id must not be blank")
        return value

    @field_validator("memory_ids")
    @classmethod
    def memory_ids_must_be_nonempty_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("memory_ids must not be empty")
        if any(not memory_id.strip() for memory_id in value):
            raise ValueError("memory IDs must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("memory IDs must be unique")
        return tuple(sorted(value))

    @field_validator("evidence")
    @classmethod
    def evidence_must_not_be_empty(
        cls, value: tuple[EvidenceItem, ...]
    ) -> tuple[EvidenceItem, ...]:
        if not value:
            raise ValueError("evidence must not be empty")
        return tuple(sorted(value, key=_evidence_sort_key))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the finding with stable key ordering."""

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


class CheckerCost(BaseModel):
    """Deterministic model-use counters without pricing or runtime assumptions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_calls: NonNegativeInt = 0
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0


class CheckerStats(BaseModel):
    """Deterministic structural work and output counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memories_scanned: NonNegativeInt
    findings_emitted: NonNegativeInt
    details: Mapping[str, NonNegativeInt] = Field(default_factory=dict, validate_default=True)

    @field_validator("details")
    @classmethod
    def detail_keys_must_not_be_blank(
        cls, value: Mapping[str, NonNegativeInt]
    ) -> Mapping[str, NonNegativeInt]:
        if any(not key.strip() for key in value):
            raise ValueError("checker stat detail keys must not be blank")
        return MappingProxyType(dict(value))

    @field_serializer("details")
    def serialize_details(self, value: Mapping[str, NonNegativeInt]) -> dict[str, int]:
        return dict(value)


class CheckerResult(BaseModel):
    """Versioned deterministic output from one checker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CHECKER_RESULT_SCHEMA_VERSION
    checker_id: str
    checker_version: str
    rule_id: str = Field(default="", validate_default=True)
    rule_version: str = Field(default="", validate_default=True)
    severity: Severity = Field(default="info", validate_default=True)
    defect_class: DefectClass
    findings: tuple[Finding, ...] = ()
    cost: CheckerCost = Field(default_factory=CheckerCost)
    stats: CheckerStats

    @model_validator(mode="before")
    @classmethod
    def populate_builtin_rule_metadata(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        checker_id = data.get("checker_id")
        if not isinstance(checker_id, str):
            return data
        metadata = _BUILTIN_RULE_METADATA.get(checker_id)
        if metadata is None:
            if any(field not in data for field in ("rule_id", "rule_version", "severity")):
                raise ValueError("custom checkers must supply explicit rule metadata")
            return data
        _, rule_id, rule_version, severity = metadata
        populated = dict(data)
        populated.setdefault("rule_id", rule_id)
        populated.setdefault("rule_version", rule_version)
        populated.setdefault("severity", severity)
        return populated

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_match(cls, value: str) -> str:
        if value != CHECKER_RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checker result schema_version: {value!r}")
        return value

    @field_validator("checker_id", "checker_version")
    @classmethod
    def checker_identity_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("checker_id and checker_version must not be blank")
        return value

    @field_validator("rule_id")
    @classmethod
    def rule_id_must_be_canonical(cls, value: str) -> str:
        if _RULE_ID_PATTERN.fullmatch(value) is None or "palintrace" in value:
            raise ValueError("rule_id must use the memory.<area>.<defect> format")
        return value

    @field_validator("rule_version")
    @classmethod
    def rule_version_must_be_numeric_semver(cls, value: str) -> str:
        if _RULE_VERSION_PATTERN.fullmatch(value) is None:
            raise ValueError("rule_version must use numeric MAJOR.MINOR.PATCH form")
        return value

    @field_validator("findings")
    @classmethod
    def findings_are_sorted(cls, value: tuple[Finding, ...]) -> tuple[Finding, ...]:
        return tuple(sorted(value, key=lambda finding: (finding.memory_ids, finding.finding_id)))

    @model_validator(mode="after")
    def findings_match_result(self) -> CheckerResult:
        metadata = _BUILTIN_RULE_METADATA.get(self.checker_id)
        if metadata is not None:
            expected_defect, rule_id, rule_version, severity = metadata
            if self.defect_class is not expected_defect:
                raise ValueError("built-in checker_id does not match defect_class")
            if (self.rule_id, self.rule_version, self.severity) != (
                rule_id,
                rule_version,
                severity,
            ):
                raise ValueError("built-in checker rule metadata must match canonical values")
        finding_ids = [finding.finding_id for finding in self.findings]
        if len(set(finding_ids)) != len(finding_ids):
            raise ValueError("finding IDs must be unique")
        if any(finding.defect_class is not self.defect_class for finding in self.findings):
            raise ValueError("every finding defect_class must match the result")
        if self.stats.findings_emitted != len(self.findings):
            raise ValueError("stats findings_emitted must equal the number of findings")
        return self

    def to_json(self, output: str | Path | None = None, *, indent: int | None = 2) -> str:
        """Serialize deterministically and optionally write a UTF-8 result file."""

        text = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )
        if indent is not None:
            text += "\n"
        if output is not None:
            Path(output).write_text(text, encoding="utf-8")
        return text
