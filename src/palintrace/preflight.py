"""Deterministic assessability inspection for the public static checkers."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from palintrace.adapters.capabilities import AdapterCapabilities, adapter_capabilities
from palintrace.audit import SkipReason
from palintrace.checker_requirements import _CHECKER_REQUIREMENTS, PUBLIC_CHECKER_IDS
from palintrace.checkers.privacy_scope_violation import ScopeIsolationPolicy
from palintrace.models import NormalizedStore, ProvenanceStatus, TranscriptSet

PREFLIGHT_REPORT_SCHEMA_VERSION = "0.1"

_BUILTIN_ADAPTERS = frozenset(("file", "mem0", "graphiti", "letta"))
_CHECKER_ORDER = MappingProxyType(
    {checker_id: index for index, checker_id in enumerate(PUBLIC_CHECKER_IDS)}
)
_SKIP_REASON_ORDER = MappingProxyType(
    {reason: index for index, reason in enumerate(SkipReason)}
)


class PreflightStatus(StrEnum):
    """Whether a checker can meaningfully assess the supplied normalized store."""

    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class PreflightLimitation(StrEnum):
    """An observable limitation on checker assessability."""

    NON_DECLARED_PROVENANCE = "non_declared_provenance"
    UNSCOPED_MEMORIES = "unscoped_memories"
    ACTIVE_STATE_UNAVAILABLE = "active_state_unavailable"
    SUPERSESSION_UNSUPPORTED = "supersession_unsupported"
    POLICY_SCOPE_UNAVAILABLE = "policy_scope_unavailable"
    POLICY_SCOPE_UNSUPPORTED = "policy_scope_unsupported"


_LIMITATION_ORDER = MappingProxyType(
    {limitation: index for index, limitation in enumerate(PreflightLimitation)}
)


class PreflightChecker(BaseModel):
    """Assessability state for one public static checker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checker_id: str
    status: PreflightStatus
    total_memories: int
    assessable_memories: int
    blocked_reasons: tuple[SkipReason, ...] = ()
    limitations: tuple[PreflightLimitation, ...] = ()

    @field_validator("checker_id")
    @classmethod
    def checker_id_must_be_public(cls, value: str) -> str:
        if value not in PUBLIC_CHECKER_IDS:
            raise ValueError(f"unknown public checker_id: {value!r}")
        return value

    @field_validator("total_memories", "assessable_memories", mode="before")
    @classmethod
    def counts_must_be_integers(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("preflight memory counts must be integers")
        return value

    @field_validator("total_memories", "assessable_memories")
    @classmethod
    def counts_must_be_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("preflight memory counts must be nonnegative")
        return value

    @field_validator("blocked_reasons")
    @classmethod
    def blocked_reasons_must_be_unique_and_canonical(
        cls, value: tuple[SkipReason, ...]
    ) -> tuple[SkipReason, ...]:
        if len(set(value)) != len(value):
            raise ValueError("blocked reasons must be unique")
        return tuple(sorted(value, key=_SKIP_REASON_ORDER.__getitem__))

    @field_validator("limitations")
    @classmethod
    def limitations_must_be_unique_and_canonical(
        cls, value: tuple[PreflightLimitation, ...]
    ) -> tuple[PreflightLimitation, ...]:
        if len(set(value)) != len(value):
            raise ValueError("preflight limitations must be unique")
        return tuple(sorted(value, key=_LIMITATION_ORDER.__getitem__))

    @model_validator(mode="after")
    def status_invariants_must_hold(self) -> PreflightChecker:
        if self.assessable_memories > self.total_memories:
            raise ValueError("assessable_memories must not exceed total_memories")
        if self.status is PreflightStatus.READY:
            if self.blocked_reasons or self.limitations:
                raise ValueError("ready checker cannot have blockers or limitations")
        elif self.status is PreflightStatus.DEGRADED:
            if self.blocked_reasons or not self.limitations:
                raise ValueError("degraded checker requires limitations and no blockers")
        else:
            if not self.blocked_reasons:
                raise ValueError("blocked checker requires at least one blocked reason")
            if self.limitations:
                raise ValueError("blocked checker cannot have limitations")
            if self.assessable_memories != 0:
                raise ValueError("blocked checker must have zero assessable memories")
        return self


class PreflightReport(BaseModel):
    """Complete deterministic assessability report for the public static checkers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = PREFLIGHT_REPORT_SCHEMA_VERSION
    adapter: str
    adapter_capabilities: AdapterCapabilities | None
    checks: tuple[PreflightChecker, ...]

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_match(cls, value: str) -> str:
        if value != PREFLIGHT_REPORT_SCHEMA_VERSION:
            raise ValueError(f"unsupported preflight report schema_version: {value!r}")
        return value

    @field_validator("adapter")
    @classmethod
    def adapter_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("adapter must not be blank")
        return value

    @field_validator("checks")
    @classmethod
    def checks_are_canonical(
        cls, value: tuple[PreflightChecker, ...]
    ) -> tuple[PreflightChecker, ...]:
        return tuple(sorted(value, key=lambda item: _CHECKER_ORDER[item.checker_id]))

    @model_validator(mode="after")
    def checker_set_must_be_complete(self) -> PreflightReport:
        checker_ids = [item.checker_id for item in self.checks]
        if len(checker_ids) != len(set(checker_ids)):
            raise ValueError("preflight checker IDs must be unique")
        missing = set(PUBLIC_CHECKER_IDS) - set(checker_ids)
        if missing:
            raise ValueError(f"preflight report is missing public checker IDs: {sorted(missing)}")
        return self

    def to_json(
        self,
        output: str | Path | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        """Serialize deterministically and optionally write a UTF-8 report file."""

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


def _checker_blockers(
    checker_id: str,
    *,
    transcripts: TranscriptSet | None,
    scope_policy: ScopeIsolationPolicy | None,
    semantic_configured: bool,
) -> tuple[SkipReason, ...]:
    requirement = _CHECKER_REQUIREMENTS[checker_id]
    reasons: list[SkipReason] = []
    if requirement.requires_transcripts and transcripts is None:
        reasons.append(SkipReason.MISSING_TRANSCRIPTS)
    if requirement.requires_scope_policy and scope_policy is None:
        reasons.append(SkipReason.MISSING_SCOPE_POLICY)
    if requirement.requires_semantic_judge and not semantic_configured:
        reasons.append(SkipReason.MISSING_SEMANTIC_CONFIGURATION)
    return tuple(reasons)


def _coverage_check(
    checker_id: str,
    *,
    total_memories: int,
    assessable_memories: int,
    limitation: PreflightLimitation,
) -> PreflightChecker:
    limitations = (limitation,) if assessable_memories < total_memories else ()
    return PreflightChecker(
        checker_id=checker_id,
        status=PreflightStatus.DEGRADED if limitations else PreflightStatus.READY,
        total_memories=total_memories,
        assessable_memories=assessable_memories,
        limitations=limitations,
    )


def _preflight_stale(
    store: NormalizedStore,
    capabilities: AdapterCapabilities | None,
) -> PreflightChecker:
    assessable = sum(memory.active is not None for memory in store.memories)
    limitations: list[PreflightLimitation] = []
    if assessable < len(store.memories):
        limitations.append(PreflightLimitation.ACTIVE_STATE_UNAVAILABLE)
    if capabilities is not None and capabilities.supersession == "unsupported":
        limitations.append(PreflightLimitation.SUPERSESSION_UNSUPPORTED)
    return PreflightChecker(
        checker_id="stale_active",
        status=PreflightStatus.DEGRADED if limitations else PreflightStatus.READY,
        total_memories=len(store.memories),
        assessable_memories=assessable,
        limitations=tuple(limitations),
    )


def _preflight_privacy(
    store: NormalizedStore,
    policy: ScopeIsolationPolicy,
    capabilities: AdapterCapabilities | None,
) -> PreflightChecker:
    dimensions = {rule.dimension.value for rule in policy.rules}
    assessable = sum(
        all(getattr(memory.scope, dimension) is not None for dimension in dimensions)
        for memory in store.memories
    )
    limitations: list[PreflightLimitation] = []
    if assessable < len(store.memories):
        limitations.append(PreflightLimitation.POLICY_SCOPE_UNAVAILABLE)
    capability_facets = {"user_id": "user_scope", "agent_id": "agent_scope"}
    if capabilities is not None and any(
        getattr(capabilities, capability_facets[dimension]) == "unsupported"
        for dimension in dimensions
    ):
        limitations.append(PreflightLimitation.POLICY_SCOPE_UNSUPPORTED)
    return PreflightChecker(
        checker_id="privacy_scope_violation",
        status=PreflightStatus.DEGRADED if limitations else PreflightStatus.READY,
        total_memories=len(store.memories),
        assessable_memories=assessable,
        limitations=tuple(limitations),
    )


def run_preflight(
    store: NormalizedStore,
    *,
    transcripts: TranscriptSet | None = None,
    scope_policy: ScopeIsolationPolicy | None = None,
    semantic_configured: bool = False,
) -> PreflightReport:
    """Inspect checker assessability without executing checkers or external systems."""

    capabilities = (
        adapter_capabilities(store.adapter) if store.adapter in _BUILTIN_ADAPTERS else None
    )
    total_memories = len(store.memories)
    declared_memories = sum(
        memory.provenance_status is ProvenanceStatus.DECLARED for memory in store.memories
    )
    scoped_memories = sum(
        any(
            value is not None
            for value in (
                memory.scope.user_id,
                memory.scope.agent_id,
                memory.scope.session_id,
            )
        )
        for memory in store.memories
    )
    checks: list[PreflightChecker] = []

    for checker_id in PUBLIC_CHECKER_IDS:
        blockers = _checker_blockers(
            checker_id,
            transcripts=transcripts,
            scope_policy=scope_policy,
            semantic_configured=semantic_configured,
        )
        if blockers:
            checks.append(
                PreflightChecker(
                    checker_id=checker_id,
                    status=PreflightStatus.BLOCKED,
                    total_memories=total_memories,
                    assessable_memories=0,
                    blocked_reasons=blockers,
                )
            )
        elif checker_id in {"orphaned_provenance", "unsupported_claim"}:
            checks.append(
                _coverage_check(
                    checker_id,
                    total_memories=total_memories,
                    assessable_memories=declared_memories,
                    limitation=PreflightLimitation.NON_DECLARED_PROVENANCE,
                )
            )
        elif checker_id == "redundancy_bloat":
            checks.append(
                _coverage_check(
                    checker_id,
                    total_memories=total_memories,
                    assessable_memories=scoped_memories,
                    limitation=PreflightLimitation.UNSCOPED_MEMORIES,
                )
            )
        elif checker_id == "stale_active":
            checks.append(_preflight_stale(store, capabilities))
        else:
            assert checker_id == "privacy_scope_violation"
            assert scope_policy is not None
            checks.append(_preflight_privacy(store, scope_policy, capabilities))

    return PreflightReport(
        adapter=store.adapter,
        adapter_capabilities=capabilities,
        checks=tuple(checks),
    )
