"""Private historical privacy-scope checker for benchmark v0.1 reproducibility."""

from __future__ import annotations

import hashlib
import json
from typing import cast

from pydantic import JsonValue

from palintrace.checkers import ScopeDimension, ScopeIsolationPolicy
from palintrace.checkers.base import deterministic_finding_id
from palintrace.checkers.models import (
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
)
from palintrace.models import NormalizedMemory, NormalizedStore, TranscriptSet
from palintrace.taxonomy import DefectClass


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


class BenchmarkPrivacyScopeViolationCheckerV1:
    """Preserve whole-record exact-replica matching for benchmark v0.1."""

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
