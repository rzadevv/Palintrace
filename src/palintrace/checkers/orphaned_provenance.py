"""Deterministic structural checker for broken declared transcript references."""

from __future__ import annotations

from palintrace.checkers.base import CheckerInputError, deterministic_finding_id
from palintrace.checkers.models import (
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
)
from palintrace.models import NormalizedStore, ProvenanceStatus, TranscriptSet
from palintrace.taxonomy import DefectClass


class OrphanedProvenanceChecker:
    """Find missing transcripts, missing turns, and out-of-range spans."""

    checker_id = "orphaned_provenance"
    checker_version = "1.0"
    defect_class = DefectClass.ORPHANED_PROVENANCE

    def check(
        self,
        store: NormalizedStore,
        *,
        transcripts: TranscriptSet | None = None,
    ) -> CheckerResult:
        """Resolve declared source references against a supplied transcript set."""

        if transcripts is None:
            raise CheckerInputError("orphaned_provenance checker requires a TranscriptSet")

        findings: list[Finding] = []
        source_refs_scanned = 0
        for memory in store.memories:
            if memory.provenance_status is not ProvenanceStatus.DECLARED:
                continue
            keyed_evidence: list[tuple[int, str, EvidenceItem]] = []
            for source_ref_index, source_ref in enumerate(memory.source_refs):
                source_refs_scanned += 1
                transcript = transcripts.get(source_ref.transcript_id)
                if transcript is None:
                    item = EvidenceItem(
                        kind="missing_transcript",
                        message="Referenced transcript does not exist.",
                        data={
                            "source_ref_index": source_ref_index,
                            "transcript_id": source_ref.transcript_id,
                        },
                    )
                    keyed_evidence.append((source_ref_index, item.kind, item))
                    continue
                if source_ref.turn_idx is None:
                    continue
                turn = next(
                    (item for item in transcript.turns if item.index == source_ref.turn_idx),
                    None,
                )
                if turn is None:
                    item = EvidenceItem(
                        kind="missing_turn",
                        message="Referenced transcript turn does not exist.",
                        data={
                            "source_ref_index": source_ref_index,
                            "transcript_id": source_ref.transcript_id,
                            "turn_idx": source_ref.turn_idx,
                        },
                    )
                    keyed_evidence.append((source_ref_index, item.kind, item))
                    continue
                if source_ref.span is None or source_ref.span[1] <= len(turn.content):
                    continue
                item = EvidenceItem(
                    kind="invalid_span",
                    message="Referenced character span exceeds the transcript turn.",
                    data={
                        "source_ref_index": source_ref_index,
                        "transcript_id": source_ref.transcript_id,
                        "turn_idx": source_ref.turn_idx,
                        "span": list(source_ref.span),
                        "turn_length": len(turn.content),
                    },
                )
                keyed_evidence.append((source_ref_index, item.kind, item))

            evidence = tuple(
                item for _, _, item in sorted(keyed_evidence, key=lambda value: value[:2])
            )
            if not evidence:
                continue
            memory_ids = (memory.id,)
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
                details={"source_refs_scanned": source_refs_scanned},
            ),
        )
