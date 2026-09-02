"""Semantic checker for stored claims unsupported by declared evidence."""

from __future__ import annotations

import hashlib

from pydantic import JsonValue

from palintrace.checkers.base import CheckerError, CheckerInputError, deterministic_finding_id
from palintrace.checkers.models import (
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
)
from palintrace.models import NormalizedStore, ProvenanceStatus, TranscriptSet
from palintrace.semantics import (
    PRIMARY_EVIDENCE_COMPOSITION_STYLE,
    EvidenceCompositionStyle,
    SemanticInputTooLongError,
    SemanticJudge,
    SemanticJudgment,
    SemanticRelation,
    compose_evidence,
    resolve_declared_evidence,
    semantic_judge_identity,
)
from palintrace.semantics.composition import canonical_unique_evidence_segments
from palintrace.semantics.models import EvidenceSegment
from palintrace.taxonomy import DefectClass


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


class UnsupportedClaimChecker:
    """Assess declared evidence with an injected directional semantic judge."""

    checker_id = "unsupported_claim"
    checker_version = "1.0"
    defect_class = DefectClass.UNSUPPORTED_CLAIM

    def __init__(
        self,
        judge: SemanticJudge,
        composition_style: EvidenceCompositionStyle = PRIMARY_EVIDENCE_COMPOSITION_STYLE,
    ) -> None:
        if not isinstance(composition_style, EvidenceCompositionStyle):
            raise ValueError("unsupported evidence composition style")
        self._judge = judge
        self._judge_id, self._judge_version = semantic_judge_identity(judge)
        self.composition_style = composition_style

    def check(
        self,
        store: NormalizedStore,
        *,
        transcripts: TranscriptSet | None = None,
    ) -> CheckerResult:
        """Assess complete declared evidence and abstain when it is unavailable."""

        if transcripts is None:
            raise CheckerInputError("unsupported_claim checker requires a TranscriptSet")

        findings: list[Finding] = []
        details = {
            "declared_memories": 0,
            "assessed_memories": 0,
            "skipped_non_declared_provenance": 0,
            "skipped_resolution_issues": 0,
            "skipped_no_evidence": 0,
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

            resolution = resolve_declared_evidence(memory, transcripts)
            if resolution.issues:
                details["skipped_resolution_issues"] += 1
                continue
            if not resolution.segments:
                details["skipped_no_evidence"] += 1
                continue

            composed = compose_evidence(
                resolution.segments,
                style=self.composition_style,
            )
            if not composed.text.strip():
                details["skipped_no_evidence"] += 1
                continue

            try:
                judgment = self._judge.judge(
                    premise=composed.text,
                    hypothesis=memory.content,
                )
                if not isinstance(judgment, SemanticJudgment):
                    raise TypeError("semantic judge returned an invalid judgment")
            except SemanticInputTooLongError:
                details["skipped_input_too_long"] += 1
                continue
            except Exception:
                raise CheckerError(
                    f"unsupported_claim semantic judgment failed for memory {memory.id!r}"
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

            unique_segments = canonical_unique_evidence_segments(resolution.segments)
            evidence = (
                EvidenceItem(
                    kind="declared_evidence_not_entailing",
                    message=(
                        "Declared transcript evidence does not entail the stored memory claim "
                        "under the configured semantic judge."
                    ),
                    data={
                        "semantic_relation": judgment.relation.value,
                        "judge_id": self._judge_id,
                        "judge_version": self._judge_version,
                        "composition_style": self.composition_style.value,
                        "unique_segment_count": composed.unique_segment_count,
                        "premise_sha256": _text_sha256(composed.text),
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
