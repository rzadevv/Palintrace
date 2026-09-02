"""Controlled unsupported-claim mutation."""

from __future__ import annotations

import random
from typing import cast

from pydantic import JsonValue

from palintrace.models import NormalizedMemory, NormalizedStore, SourceRef, TranscriptSet
from palintrace.mutations.base import (
    MutationApplication,
    MutationPreconditionError,
    replace_memory,
    replace_once,
    require_no_embedding,
    select_memory,
    validated_memory_copy,
)
from palintrace.mutations.models import MutationRequest, MutationTarget, MutationTargetRole

SUBTYPE = "factual_substitution"


def apply(
    store: NormalizedStore,
    request: MutationRequest,
    transcripts: TranscriptSet | None,
    rng: random.Random,
    mutation_id: str,
) -> MutationApplication:
    """Substitute one declared fact while retaining its original source references."""

    del mutation_id
    subtype = request.subtype or SUBTYPE
    if subtype != SUBTYPE:
        raise MutationPreconditionError(f"unsupported unsupported-claim subtype: {subtype}")
    target = select_memory(
        store,
        request.target_memory_id,
        rng,
        predicate=lambda memory: bool(memory.source_refs),
        requirement="a memory with declared provenance",
    )
    require_no_embedding(target)
    replacement = replace_once(target.content, request.replace_from, request.replace_to)
    if transcripts is None:
        raise MutationPreconditionError("unsupported-claim mutation requires transcripts")
    evidence = _resolved_evidence(target, transcripts)
    if request.replace_from is None or not any(
        request.replace_from in item for item in evidence
    ):
        raise MutationPreconditionError(
            "no resolvable declared source evidence contains replace_from"
        )
    if request.replace_to is not None and any(request.replace_to in item for item in evidence):
        raise MutationPreconditionError(
            "resolved source evidence already contains replace_to; unsupported gold is ambiguous"
        )
    mutated = validated_memory_copy(target, content=replacement)
    return MutationApplication(
        store=replace_memory(store, target.id, mutated),
        subtype=subtype,
        target_memory_ids=(target.id,),
        targets=(
            MutationTarget(
                memory_id=target.id,
                role=MutationTargetRole.PRIMARY,
            ),
        ),
        modified_memory_ids=(target.id,),
        parameters={
            "original_content": target.content,
            "mutated_content": replacement,
            "replace_from": cast(JsonValue, request.replace_from),
            "replace_to": cast(JsonValue, request.replace_to),
            "source_refs": cast(
                JsonValue,
                [source_ref.model_dump(mode="json") for source_ref in target.source_refs],
            ),
        },
    )


def _resolved_evidence(
    memory: NormalizedMemory, transcripts: TranscriptSet
) -> tuple[str, ...]:
    return tuple(
        evidence
        for source_ref in memory.source_refs
        for evidence in _evidence(source_ref, transcripts)
    )


def _evidence(source_ref: SourceRef, transcripts: TranscriptSet) -> tuple[str, ...]:
    transcript = transcripts.get(source_ref.transcript_id)
    if transcript is None:
        return ()
    if source_ref.turn_idx is None:
        return tuple(turn.content for turn in transcript.turns)
    turn = next((item for item in transcript.turns if item.index == source_ref.turn_idx), None)
    if turn is None:
        return ()
    if source_ref.span is None:
        return (turn.content,)
    start, end = source_ref.span
    if end > len(turn.content):
        return ()
    return (turn.content[start:end],)
