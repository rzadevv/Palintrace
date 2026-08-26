"""Development candidate for speaker-grounded unsupported-claim checking."""

from __future__ import annotations

import hashlib

from pydantic import JsonValue

from memlint.checkers.base import CheckerError, CheckerInputError, deterministic_finding_id
from memlint.checkers.models import (
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
)
from memlint.models import NormalizedStore, ProvenanceStatus, TranscriptSet
from memlint.semantics import (
    EvidenceCompositionStyle,
    SemanticInputTooLongError,
    SemanticJudge,
    SemanticJudgment,
    SemanticRelation,
    compose_evidence,
    resolve_declared_evidence,
    semantic_judge_identity,
)
from memlint.semantics.composition import canonical_unique_evidence_segments
from memlint.semantics.identity import (
    SpeakerIdentityBindings,
    SpeakerIdentityResolutionStatus,
    build_speaker_grounded_premise,
    resolve_speaker_identity,
)
from memlint.semantics.models import EvidenceSegment
from memlint.taxonomy import DefectClass


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_coordinates(segments: tuple[EvidenceSegment, ...]) -> list[JsonValue]:
    coordinates: list[JsonValue] = []
    for segment in segments:
        coordinate: dict[str, JsonValue] = {
            "transcript_id": segment.transcript_id,
            "turn_idx": segment.turn_idx,
            "span": list(segment.span) if segment.span is not None else None,
            "role": segment.role,
        }
        coordinates.append(coordinate)
    return coordinates


class IdentityGroundedUnsupportedClaimChecker:
    """Development checker requiring explicit source-turn speaker identity."""

    checker_id = "unsupported_claim_identity_grounded"
    checker_version = "0.1"
    defect_class = DefectClass.UNSUPPORTED_CLAIM

    def __init__(
        self,
        judge: SemanticJudge,
        speaker_bindings: SpeakerIdentityBindings,
    ) -> None:
        if not isinstance(speaker_bindings, SpeakerIdentityBindings):
            raise TypeError("speaker_bindings must be a SpeakerIdentityBindings")
        self._judge = judge
        self._judge_id, self._judge_version = semantic_judge_identity(judge)
        self._speaker_bindings = speaker_bindings

    def check(
        self,
        store: NormalizedStore,
        *,
        transcripts: TranscriptSet | None = None,
    ) -> CheckerResult:
        """Assess only evidence with one explicitly resolved speaker label."""

        if transcripts is None:
            raise CheckerInputError(
                "unsupported_claim_identity_grounded checker requires a TranscriptSet"
            )

        findings: list[Finding] = []
        details = {
            "declared_memories": 0,
            "assessed_memories": 0,
            "skipped_non_declared_provenance": 0,
            "skipped_resolution_issues": 0,
            "skipped_no_evidence": 0,
            "skipped_identity_unavailable": 0,
            "skipped_identity_conflict": 0,
            "skipped_input_too_long": 0,
            "entailment_judgments": 0,
            "neutral_judgments": 0,
            "contradiction_judgments": 0,
        }
        model_calls = 0
        input_tokens = 0
        output_tokens = 0

        for memory in sorted(store.memories, key=lambda item: item.id):
            if memory.provenance_status is not ProvenanceStatus.DECLARED:
                details["skipped_non_declared_provenance"] += 1
                continue
            details["declared_memories"] += 1

            evidence_resolution = resolve_declared_evidence(memory, transcripts)
            if evidence_resolution.issues:
                details["skipped_resolution_issues"] += 1
                continue
            if not evidence_resolution.segments:
                details["skipped_no_evidence"] += 1
                continue

            composed = compose_evidence(
                evidence_resolution.segments,
                style=EvidenceCompositionStyle.PLAIN,
            )
            if not composed.text.strip():
                details["skipped_no_evidence"] += 1
                continue

            identity_resolution = resolve_speaker_identity(memory, self._speaker_bindings)
            if identity_resolution.status is SpeakerIdentityResolutionStatus.UNAVAILABLE:
                details["skipped_identity_unavailable"] += 1
                continue
            if identity_resolution.status is SpeakerIdentityResolutionStatus.CONFLICT:
                details["skipped_identity_conflict"] += 1
                continue

            grounded_premise = build_speaker_grounded_premise(
                composed.text,
                identity_resolution,
            )
            try:
                judgment = self._judge.judge(
                    premise=grounded_premise,
                    hypothesis=memory.content,
                )
                if not isinstance(judgment, SemanticJudgment):
                    raise TypeError("semantic judge returned an invalid judgment")
            except SemanticInputTooLongError:
                details["skipped_input_too_long"] += 1
                continue
            except Exception:
                raise CheckerError(
                    "unsupported_claim_identity_grounded semantic judgment failed "
                    f"for memory {memory.id!r}"
                ) from None

            details["assessed_memories"] += 1
            model_calls += judgment.usage.model_calls
            input_tokens += judgment.usage.input_tokens
            output_tokens += judgment.usage.output_tokens
            if judgment.relation is SemanticRelation.ENTAILMENT:
                details["entailment_judgments"] += 1
                continue
            if judgment.relation is SemanticRelation.NEUTRAL:
                details["neutral_judgments"] += 1
            else:
                details["contradiction_judgments"] += 1

            unique_segments = canonical_unique_evidence_segments(
                evidence_resolution.segments
            )
            evidence = (
                EvidenceItem(
                    kind="speaker_grounded_declared_evidence_not_entailing",
                    message=(
                        "Speaker-grounded declared transcript evidence does not entail the "
                        "stored memory claim under the configured semantic judge."
                    ),
                    data={
                        "semantic_relation": judgment.relation.value,
                        "judge_id": self._judge_id,
                        "judge_version": self._judge_version,
                        "composition_style": EvidenceCompositionStyle.PLAIN.value,
                        "identity_grounding": "explicit_turn_binding_v0.1",
                        "identity_source_turn_count": len(identity_resolution.source_turns),
                        "unique_segment_count": composed.unique_segment_count,
                        "premise_sha256": _text_sha256(grounded_premise),
                        "hypothesis_sha256": _text_sha256(memory.content),
                        "source_coordinates": _source_coordinates(unique_segments),
                    },
                ),
            )
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
                    confidence=judgment.score,
                    evidence=evidence,
                )
            )

        return CheckerResult(
            checker_id=self.checker_id,
            checker_version=self.checker_version,
            defect_class=self.defect_class,
            findings=tuple(findings),
            cost=CheckerCost(
                model_calls=model_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            stats=CheckerStats(
                memories_scanned=len(store.memories),
                findings_emitted=len(findings),
                details=details,
            ),
        )
