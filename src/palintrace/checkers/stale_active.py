"""Deterministic structural checker for active explicitly superseded memories."""

from __future__ import annotations

from palintrace.checkers.base import deterministic_finding_id
from palintrace.checkers.models import (
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
)
from palintrace.models import NormalizedStore, TranscriptSet
from palintrace.taxonomy import DefectClass


def _supersession_cycles(links: list[tuple[str, str]]) -> list[tuple[str, ...]]:
    """Return each cyclic component of the resolved supersession graph, iterative Tarjan."""

    adjacency: dict[str, list[str]] = {}
    for superseder, old_id in links:
        adjacency.setdefault(superseder, []).append(old_id)
        adjacency.setdefault(old_id, [])
    for targets in adjacency.values():
        targets.sort()

    index_by_node: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    component_stack: list[str] = []
    next_index = 0
    cycles: list[tuple[str, ...]] = []

    for root in sorted(adjacency):
        if root in index_by_node:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, offset = work[-1]
            if offset == 0:
                index_by_node[node] = next_index
                lowlink[node] = next_index
                next_index += 1
                component_stack.append(node)
                on_stack.add(node)
            targets = adjacency[node]
            if offset < len(targets):
                work[-1] = (node, offset + 1)
                target = targets[offset]
                if target not in index_by_node:
                    work.append((target, 0))
                elif target in on_stack:
                    lowlink[node] = min(lowlink[node], index_by_node[target])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index_by_node[node]:
                component: list[str] = []
                while True:
                    member = component_stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                # self-links never reach the graph, so only multi-node components are cycles
                if len(component) > 1:
                    cycles.append(tuple(sorted(component)))
    return sorted(cycles)


class StaleActiveChecker:
    """Find active memories targeted by explicit supersession links or caught in a cycle."""

    checker_id = "stale_active"
    checker_version = "2.0"
    defect_class = DefectClass.STALE_ACTIVE

    def check(
        self,
        store: NormalizedStore,
        *,
        transcripts: TranscriptSet | None = None,
    ) -> CheckerResult:
        """Resolve supersession links, report cycles, then report remaining active targets."""

        memory_by_id = {memory.id: memory for memory in store.memories}
        evidence_by_old_id: dict[str, list[EvidenceItem]] = {}
        resolved_links: list[tuple[str, str]] = []
        dangling_targets: list[tuple[str, str]] = []
        supersession_links_scanned = 0
        self_links_skipped = 0

        for superseder in store.memories:
            for old_id in superseder.supersedes:
                supersession_links_scanned += 1
                if old_id == superseder.id:
                    self_links_skipped += 1
                    continue
                old_memory = memory_by_id.get(old_id)
                if old_memory is None:
                    dangling_targets.append((superseder.id, old_id))
                    continue
                resolved_links.append((superseder.id, old_id))
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

        cycles = _supersession_cycles(resolved_links)
        findings: list[Finding] = []
        cycle_members: set[str] = set()
        for members in cycles:
            cycle_members.update(members)
            active_members = tuple(
                member for member in members if memory_by_id[member].active is True
            )
            if not active_members:
                continue
            evidence = (
                EvidenceItem(
                    kind="supersession_cycle",
                    message=(
                        "Memories supersede each other in a cycle, so no record is the "
                        "unambiguous replacement."
                    ),
                    data={
                        "members": list(members),
                        "active_members": list(active_members),
                    },
                ),
            )
            findings.append(self._finding(members, evidence))

        # a cycle member's staleness is described by its cycle, not by a direct link
        for old_id in sorted(evidence_by_old_id):
            if old_id in cycle_members:
                continue
            findings.append(self._finding((old_id,), tuple(evidence_by_old_id[old_id])))

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
                    "resolved_supersession_links": len(resolved_links),
                    "missing_targets_skipped": len(dangling_targets),
                    "dangling_supersession_targets": tuple(
                        {"superseder_id": superseder_id, "missing_target_id": missing_id}
                        for superseder_id, missing_id in sorted(dangling_targets)
                    ),
                    "self_links_skipped": self_links_skipped,
                    "supersession_cycles": len(cycles),
                },
            ),
        )

    def _finding(
        self, memory_ids: tuple[str, ...], evidence: tuple[EvidenceItem, ...]
    ) -> Finding:
        return Finding(
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
