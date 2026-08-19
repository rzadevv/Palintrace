"""Deterministic structural checker for active explicitly superseded memories."""

from __future__ import annotations

from memlint.checkers.base import deterministic_finding_id
from memlint.checkers.models import (
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
)
from memlint.models import NormalizedStore, TranscriptSet
from memlint.taxonomy import DefectClass


class StaleActiveChecker:
    """Find active memories targeted by direct explicit supersession links."""

    checker_id = "stale_active"
    checker_version = "1.0"
    defect_class = DefectClass.STALE_ACTIVE

    def check(
        self,
        store: NormalizedStore,
        *,
        transcripts: TranscriptSet | None = None,
    ) -> CheckerResult:
        """Resolve direct supersession links and report old memories still marked active."""

        memory_by_id = {memory.id: memory for memory in store.memories}
        evidence_by_old_id: dict[str, list[EvidenceItem]] = {}
        supersession_links_scanned = 0
        resolved_supersession_links = 0
        missing_targets_skipped = 0
        self_links_skipped = 0

        for superseder in store.memories:
            for old_id in superseder.supersedes:
                supersession_links_scanned += 1
                if old_id == superseder.id:
                    self_links_skipped += 1
                    continue
                old_memory = memory_by_id.get(old_id)
                if old_memory is None:
                    missing_targets_skipped += 1
                    continue
                resolved_supersession_links += 1
                if old_memory.active is not True:
                    continue
                evidence_by_old_id.setdefault(old_id, []).append(
                    EvidenceItem(
                        kind="active_superseded",
                        message=(
                            "Memory remains active despite an explicit supersession relationship."
                        ),
                        data={
                            "superseding_memory_id": superseder.id,
                            "old_active": True,
                        },
                    )
                )

        findings: list[Finding] = []
        for old_id in sorted(evidence_by_old_id):
            evidence = tuple(evidence_by_old_id[old_id])
            memory_ids = (old_id,)
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
                    "supersession_links_scanned": supersession_links_scanned,
                    "resolved_supersession_links": resolved_supersession_links,
                    "missing_targets_skipped": missing_targets_skipped,
                    "self_links_skipped": self_links_skipped,
                },
            ),
        )
