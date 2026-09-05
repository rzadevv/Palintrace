"""Deterministic aggregate execution for the built-in public checkers."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from palintrace.checker_requirements import _CHECKER_REQUIREMENTS, PUBLIC_CHECKER_IDS
from palintrace.checkers.models import CheckerResult
from palintrace.checkers.orphaned_provenance import OrphanedProvenanceChecker
from palintrace.checkers.privacy_scope_violation import (
    PrivacyScopeViolationChecker,
    ScopeIsolationPolicy,
)
from palintrace.checkers.redundancy_bloat import RedundancyBloatChecker
from palintrace.checkers.stale_active import StaleActiveChecker
from palintrace.checkers.unsupported_claim import UnsupportedClaimChecker
from palintrace.models import NormalizedStore, TranscriptSet
from palintrace.semantics.base import SemanticJudge

AUDIT_REPORT_SCHEMA_VERSION = "0.1"

_CHECKER_ORDER = MappingProxyType(
    {checker_id: index for index, checker_id in enumerate(PUBLIC_CHECKER_IDS)}
)


class SkipReason(StrEnum):
    """A missing external input that prevents one checker from executing."""

    MISSING_TRANSCRIPTS = "missing_transcripts"
    MISSING_SCOPE_POLICY = "missing_scope_policy"
    MISSING_SEMANTIC_CONFIGURATION = "missing_semantic_configuration"


_SKIP_REASON_ORDER = MappingProxyType(
    {reason: index for index, reason in enumerate(SkipReason)}
)


class SkippedChecker(BaseModel):
    """One built-in checker omitted because required input was absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checker_id: str
    reasons: tuple[SkipReason, ...]

    @field_validator("checker_id")
    @classmethod
    def checker_id_must_be_public(cls, value: str) -> str:
        if value not in PUBLIC_CHECKER_IDS:
            raise ValueError(f"unknown public checker_id: {value!r}")
        return value

    @field_validator("reasons")
    @classmethod
    def reasons_must_be_nonempty_unique_and_canonical(
        cls, value: tuple[SkipReason, ...]
    ) -> tuple[SkipReason, ...]:
        if not value:
            raise ValueError("skipped checker reasons must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("skipped checker reasons must be unique")
        return tuple(sorted(value, key=_SKIP_REASON_ORDER.__getitem__))


class AuditReport(BaseModel):
    """A complete aggregate run across the built-in public checkers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = AUDIT_REPORT_SCHEMA_VERSION
    results: tuple[CheckerResult, ...] = ()
    skipped: tuple[SkippedChecker, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_match(cls, value: str) -> str:
        if value != AUDIT_REPORT_SCHEMA_VERSION:
            raise ValueError(f"unsupported audit report schema_version: {value!r}")
        return value

    @field_validator("results")
    @classmethod
    def results_are_canonical(
        cls, value: tuple[CheckerResult, ...]
    ) -> tuple[CheckerResult, ...]:
        fallback = len(PUBLIC_CHECKER_IDS)
        return tuple(
            sorted(
                value,
                key=lambda result: _CHECKER_ORDER.get(result.checker_id, fallback),
            )
        )

    @field_validator("skipped")
    @classmethod
    def skipped_are_canonical(
        cls, value: tuple[SkippedChecker, ...]
    ) -> tuple[SkippedChecker, ...]:
        return tuple(sorted(value, key=lambda item: _CHECKER_ORDER[item.checker_id]))

    @model_validator(mode="after")
    def checker_partition_must_be_complete(self) -> AuditReport:
        expected = set(PUBLIC_CHECKER_IDS)
        result_ids = [result.checker_id for result in self.results]
        skipped_ids = [item.checker_id for item in self.skipped]

        unknown = (set(result_ids) | set(skipped_ids)) - expected
        if unknown:
            raise ValueError(f"unknown public checker IDs: {sorted(unknown)}")
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("completed checker IDs must be unique")
        if len(skipped_ids) != len(set(skipped_ids)):
            raise ValueError("skipped checker IDs must be unique")
        overlap = set(result_ids) & set(skipped_ids)
        if overlap:
            raise ValueError(f"checkers cannot be both completed and skipped: {sorted(overlap)}")
        missing = expected - set(result_ids) - set(skipped_ids)
        if missing:
            raise ValueError(f"audit report is missing public checker IDs: {sorted(missing)}")
        return self

    def to_json(self, output: str | Path | None = None, *, indent: int | None = 2) -> str:
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


def run_aggregate_audit(
    store: NormalizedStore,
    *,
    transcripts: TranscriptSet | None = None,
    scope_policy: ScopeIsolationPolicy | None = None,
    semantic_judge: SemanticJudge | None = None,
) -> AuditReport:
    """Run every eligible built-in public checker in canonical order."""

    results: list[CheckerResult] = []
    skipped: list[SkippedChecker] = []

    for checker_id in PUBLIC_CHECKER_IDS:
        requirement = _CHECKER_REQUIREMENTS[checker_id]
        reasons: list[SkipReason] = []
        if requirement.requires_transcripts and transcripts is None:
            reasons.append(SkipReason.MISSING_TRANSCRIPTS)
        if requirement.requires_scope_policy and scope_policy is None:
            reasons.append(SkipReason.MISSING_SCOPE_POLICY)
        if requirement.requires_semantic_judge and semantic_judge is None:
            reasons.append(SkipReason.MISSING_SEMANTIC_CONFIGURATION)
        if reasons:
            skipped.append(SkippedChecker(checker_id=checker_id, reasons=tuple(reasons)))
            continue

        if checker_id == "orphaned_provenance":
            result = OrphanedProvenanceChecker().check(store, transcripts=transcripts)
        elif checker_id == "redundancy_bloat":
            result = RedundancyBloatChecker().check(store)
        elif checker_id == "stale_active":
            result = StaleActiveChecker().check(store)
        elif checker_id == "privacy_scope_violation":
            assert scope_policy is not None
            result = PrivacyScopeViolationChecker(scope_policy).check(store)
        else:
            assert checker_id == "unsupported_claim"
            assert semantic_judge is not None
            result = UnsupportedClaimChecker(semantic_judge).check(
                store,
                transcripts=transcripts,
            )
        results.append(result)

    return AuditReport(results=tuple(results), skipped=tuple(skipped))
