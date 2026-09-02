"""Policy-directed checker for prohibited exact replicas across principal scopes."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from palintrace.checkers.base import CheckerInputError, deterministic_finding_id
from palintrace.checkers.models import (
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
)
from palintrace.models import NormalizedMemory, NormalizedStore, TranscriptSet
from palintrace.taxonomy import DefectClass

SCOPE_POLICY_SCHEMA_VERSION = "0.1"


class ScopeDimension(StrEnum):
    """Normalized principal dimensions supported by scope-isolation policies."""

    USER_ID = "user_id"
    AGENT_ID = "agent_id"


class PrincipalBoundaryRule(BaseModel):
    """Declare one authoritative principal and its prohibited destinations."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dimension: ScopeDimension
    authoritative_source_principal: str
    prohibited_destination_principals: tuple[str, ...]

    @field_validator("authoritative_source_principal")
    @classmethod
    def authoritative_principal_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("authoritative_source_principal must not be blank")
        return value

    @field_validator("prohibited_destination_principals")
    @classmethod
    def destinations_must_be_valid(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one prohibited destination principal is required")
        if any(not principal.strip() for principal in value):
            raise ValueError("prohibited destination principals must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("prohibited destination principals must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def authoritative_principal_must_not_be_prohibited(self) -> PrincipalBoundaryRule:
        if self.authoritative_source_principal in self.prohibited_destination_principals:
            raise ValueError("authoritative source principal cannot be prohibited")
        return self


class ScopeIsolationPolicy(BaseModel):
    """Canonical versioned authoritative principal-boundary policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = SCOPE_POLICY_SCHEMA_VERSION
    rules: tuple[PrincipalBoundaryRule, ...]

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_match(cls, value: str) -> str:
        if value != SCOPE_POLICY_SCHEMA_VERSION:
            raise ValueError(f"unsupported scope policy schema_version: {value!r}")
        return value

    @field_validator("rules")
    @classmethod
    def rules_must_be_nonempty_and_canonical(
        cls, value: tuple[PrincipalBoundaryRule, ...]
    ) -> tuple[PrincipalBoundaryRule, ...]:
        if not value:
            raise ValueError("scope policy requires at least one rule")
        grouped: dict[tuple[ScopeDimension, str], set[str]] = {}
        for rule in value:
            key = (rule.dimension, rule.authoritative_source_principal)
            grouped.setdefault(key, set()).update(rule.prohibited_destination_principals)
        return tuple(
            PrincipalBoundaryRule(
                dimension=dimension,
                authoritative_source_principal=authoritative_principal,
                prohibited_destination_principals=tuple(sorted(destinations)),
            )
            for (dimension, authoritative_principal), destinations in sorted(
                grouped.items(), key=lambda item: (item[0][0].value, item[0][1])
            )
        )

    def to_json(self, output: str | Path | None = None, *, indent: int | None = 2) -> str:
        """Serialize the canonical policy deterministically."""

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


def load_scope_policy(path: str | Path) -> ScopeIsolationPolicy:
    """Load and validate a JSON scope-isolation policy."""

    policy_path = Path(path)
    try:
        text = policy_path.read_text(encoding="utf-8")
    except OSError as error:
        raise CheckerInputError(f"could not read scope policy {policy_path}: {error}") from error
    try:
        return ScopeIsolationPolicy.model_validate_json(text)
    except ValidationError as error:
        raise CheckerInputError(f"invalid scope policy {policy_path}: {error}") from error


def _portable_replica_identity(memory: NormalizedMemory, dimension: ScopeDimension) -> str:
    payload = memory.semantic_dict()
    payload.pop("id")
    scope = cast(dict[str, JsonValue], payload["scope"])
    scope.pop(dimension.value)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class PrivacyScopeViolationChecker:
    """Find prohibited exact replicas relative to authoritative principals."""

    checker_id = "privacy_scope_violation"
    checker_version = "1.0"
    defect_class = DefectClass.PRIVACY_SCOPE_VIOLATION

    def __init__(self, policy: ScopeIsolationPolicy) -> None:
        self.policy = policy

    def check(
        self,
        store: NormalizedStore,
        *,
        transcripts: TranscriptSet | None = None,
    ) -> CheckerResult:
        """Match authoritative and prohibited records by exact portable identity."""

        evidence_by_destination: dict[str, list[EvidenceItem]] = {}
        authoritative_candidates = 0
        destination_candidates = 0
        exact_replica_matches = 0

        for rule in self.policy.rules:
            authoritative_by_identity: dict[str, list[NormalizedMemory]] = {}
            destinations: list[NormalizedMemory] = []
            prohibited_destinations = set(rule.prohibited_destination_principals)
            for memory in store.memories:
                principal = getattr(memory.scope, rule.dimension.value)
                if principal == rule.authoritative_source_principal:
                    authoritative_candidates += 1
                    identity = _portable_replica_identity(memory, rule.dimension)
                    authoritative_by_identity.setdefault(identity, []).append(memory)
                elif principal in prohibited_destinations:
                    destination_candidates += 1
                    destinations.append(memory)

            for destination in destinations:
                identity = _portable_replica_identity(destination, rule.dimension)
                matching_sources = authoritative_by_identity.get(identity, ())
                replica_sha256 = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                destination_principal = getattr(destination.scope, rule.dimension.value)
                for authoritative_memory in matching_sources:
                    exact_replica_matches += 1
                    evidence_by_destination.setdefault(destination.id, []).append(
                        EvidenceItem(
                            kind="prohibited_exact_replica",
                            message=(
                                "An exact portable replica of an authoritative-source memory "
                                "exists in a prohibited principal scope."
                            ),
                            data={
                                "authoritative_source_memory_id": authoritative_memory.id,
                                "scope_dimension": rule.dimension.value,
                                "authoritative_source_principal": (
                                    rule.authoritative_source_principal
                                ),
                                "destination_principal": destination_principal,
                                "replica_sha256": replica_sha256,
                            },
                        )
                    )

        findings: list[Finding] = []
        for destination_id in sorted(evidence_by_destination):
            evidence = tuple(evidence_by_destination[destination_id])
            memory_ids = (destination_id,)
            findings.append(
                Finding(
                    finding_id=deterministic_finding_id(
                        checker_id=self.checker_id,
                        checker_version=self.checker_version,
                        defect_class=self.defect_class,
                        memory_ids=memory_ids,
                        evidence=evidence,
                    ),
                    defect_class=self.defect_class,
                    memory_ids=memory_ids,
                    confidence=1.0,
                    evidence=evidence,
                )
            )

        return CheckerResult(
            checker_id=self.checker_id,
            checker_version=self.checker_version,
            defect_class=self.defect_class,
            findings=tuple(findings),
            cost=CheckerCost(),
            stats=CheckerStats(
                memories_scanned=len(store.memories),
                findings_emitted=len(findings),
                details={
                    "policy_rules_scanned": len(self.policy.rules),
                    "authoritative_candidates": authoritative_candidates,
                    "destination_candidates": destination_candidates,
                    "exact_replica_matches": exact_replica_matches,
                },
            ),
        )
