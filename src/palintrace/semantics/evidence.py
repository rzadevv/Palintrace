"""Deterministic resolution of declared transcript evidence."""

from __future__ import annotations

from palintrace.models import NormalizedMemory, ProvenanceStatus, TranscriptSet
from palintrace.semantics.models import (
    EvidenceIssueKind,
    EvidenceResolution,
    EvidenceResolutionIssue,
    EvidenceSegment,
)


def resolve_declared_evidence(
    memory: NormalizedMemory,
    transcripts: TranscriptSet,
) -> EvidenceResolution:
    """Resolve declared source references without making a semantic claim."""

    if memory.provenance_status is not ProvenanceStatus.DECLARED:
        return EvidenceResolution()

    segments: list[EvidenceSegment] = []
    issues: list[EvidenceResolutionIssue] = []
    for source_ref_index, source_ref in enumerate(memory.source_refs):
        transcript = transcripts.get(source_ref.transcript_id)
        if transcript is None:
            issues.append(
                EvidenceResolutionIssue(
                    kind=EvidenceIssueKind.MISSING_TRANSCRIPT,
                    source_ref_index=source_ref_index,
                    transcript_id=source_ref.transcript_id,
                )
            )
            continue

        if source_ref.turn_idx is None:
            for turn in sorted(transcript.turns, key=lambda item: item.index):
                segments.append(
                    EvidenceSegment(
                        source_ref_index=source_ref_index,
                        transcript_id=transcript.id,
                        turn_idx=turn.index,
                        role=turn.role,
                        span=None,
                        text=turn.content,
                    )
                )
            continue

        resolved_turn = next(
            (item for item in transcript.turns if item.index == source_ref.turn_idx),
            None,
        )
        if resolved_turn is None:
            issues.append(
                EvidenceResolutionIssue(
                    kind=EvidenceIssueKind.MISSING_TURN,
                    source_ref_index=source_ref_index,
                    transcript_id=transcript.id,
                    turn_idx=source_ref.turn_idx,
                )
            )
            continue

        if source_ref.span is None:
            text = resolved_turn.content
        elif source_ref.span[1] > len(resolved_turn.content):
            issues.append(
                EvidenceResolutionIssue(
                    kind=EvidenceIssueKind.INVALID_SPAN,
                    source_ref_index=source_ref_index,
                    transcript_id=transcript.id,
                    turn_idx=resolved_turn.index,
                    span=source_ref.span,
                    turn_length=len(resolved_turn.content),
                )
            )
            continue
        else:
            text = resolved_turn.content[source_ref.span[0] : source_ref.span[1]]

        segments.append(
            EvidenceSegment(
                source_ref_index=source_ref_index,
                transcript_id=transcript.id,
                turn_idx=resolved_turn.index,
                role=resolved_turn.role,
                span=source_ref.span,
                text=text,
            )
        )

    return EvidenceResolution(segments=tuple(segments), issues=tuple(issues))
