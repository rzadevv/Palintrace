"""Deterministic structural checker for same-scope duplicate claims."""

from __future__ import annotations

import hashlib

from palintrace.checkers.base import deterministic_finding_id, normalized_content
from palintrace.checkers.models import (
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
)
from palintrace.models import NormalizedMemory, NormalizedStore, TranscriptSet
from palintrace.taxonomy import DefectClass

ScopeKey = tuple[str | None, str | None, str | None]
GroupKey = tuple[str, ScopeKey]


class RedundancyBloatChecker:
    """Find exact and normalized duplicate groups in the same observable scope."""

    checker_id = "redundancy_bloat"
    checker_version = "3.0"
    defect_class = DefectClass.REDUNDANCY_BLOAT

    def check(
        self,
        store: NormalizedStore,
        *,
        transcripts: TranscriptSet | None = None,
    ) -> CheckerResult:
        """Group normalized claims by observable scope and emit one finding per group."""

        groups: dict[GroupKey, list[NormalizedMemory]] = {}
        unscoped_memories_skipped = 0
        for memory in store.memories:
            scope_key = (
                memory.scope.user_id,
                memory.scope.agent_id,
                memory.scope.session_id,
            )
            if all(value is None for value in scope_key):
                unscoped_memories_skipped += 1
                continue
            groups.setdefault((normalized_content(memory.content), scope_key), []).append(memory)

        grouped_duplicates: list[tuple[tuple[str, ...], str, bool, ScopeKey]] = []
        exact_groups = 0
        for (normalized, scope_key), memories in groups.items():
            if len(memories) < 2:
                continue
            memory_ids = tuple(sorted(memory.id for memory in memories))
            is_exact = len({memory.content for memory in memories}) == 1
            if is_exact:
                exact_groups += 1
            grouped_duplicates.append((memory_ids, normalized, is_exact, scope_key))

        findings: list[Finding] = []
        for memory_ids, normalized, is_exact, scope_key in sorted(
            grouped_duplicates, key=lambda group: group[0]
        ):
            evidence = (
                EvidenceItem(
                    kind="exact_duplicate" if is_exact else "normalized_duplicate",
                    message=(
                        "Memories contain identical content in the same observable scope."
                        if is_exact
                        else "Memories contain equivalent content once normalized "
                        "in the same observable scope."
                    ),
                    data={
                        "match_kind": "exact" if is_exact else "normalized",
                        "normalized_content_sha256": hashlib.sha256(
                            normalized.encode("utf-8")
                        ).hexdigest(),
                        "normalized_content_length": len(normalized),
                        "scope": {
                            "user_id": scope_key[0],
                            "agent_id": scope_key[1],
                            "session_id": scope_key[2],
                        },
                    },
                ),
            )
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
                    "eligible_memories": len(store.memories) - unscoped_memories_skipped,
                    "unscoped_memories_skipped": unscoped_memories_skipped,
                    "duplicate_groups": len(grouped_duplicates),
                    "exact_duplicate_groups": exact_groups,
                    "normalized_duplicate_groups": len(grouped_duplicates) - exact_groups,
                },
            ),
        )
