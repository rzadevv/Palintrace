"""Deterministic structural checker for exact same-scope duplicates."""

from __future__ import annotations

import hashlib
from itertools import combinations

from memlint.checkers.base import deterministic_finding_id
from memlint.checkers.models import (
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
)
from memlint.models import NormalizedMemory, NormalizedStore, TranscriptSet
from memlint.taxonomy import DefectClass

ScopeKey = tuple[str | None, str | None, str | None]
GroupKey = tuple[str, ScopeKey]


class RedundancyBloatChecker:
    """Find exact-content duplicate pairs in the same observable scope."""

    checker_id = "redundancy_bloat"
    checker_version = "1.0"
    defect_class = DefectClass.REDUNDANCY_BLOAT

    def check(
        self,
        store: NormalizedStore,
        *,
        transcripts: TranscriptSet | None = None,
    ) -> CheckerResult:
        """Group exact claims by observable normalized scope and emit duplicate pairs."""

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
            groups.setdefault((memory.content, scope_key), []).append(memory)

        findings: list[Finding] = []
        duplicate_groups = 0
        for (content, scope_key), memories in groups.items():
            if len(memories) < 2:
                continue
            duplicate_groups += 1
            evidence = (
                EvidenceItem(
                    kind="exact_duplicate",
                    message="Memories contain identical content in the same observable scope.",
                    data={
                        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "content_length": len(content),
                        "scope": {
                            "user_id": scope_key[0],
                            "agent_id": scope_key[1],
                            "session_id": scope_key[2],
                        },
                    },
                ),
            )
            ordered_memories = sorted(memories, key=lambda memory: memory.id)
            for first, second in combinations(ordered_memories, 2):
                memory_ids = (first.id, second.id)
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
                    "duplicate_groups": duplicate_groups,
                },
            ),
        )
