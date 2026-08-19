"""Structurally valid orphaned-provenance mutations."""

from __future__ import annotations

import random

from memlint.models import NormalizedMemory, NormalizedStore, SourceRef, TranscriptSet
from memlint.mutations.base import (
    MutationApplication,
    MutationPreconditionError,
    deterministic_id,
    replace_memory,
    select_memory,
    validated_memory_copy,
)
from memlint.mutations.models import MutationRequest, MutationTarget, MutationTargetRole

SUBTYPES = {"missing_transcript", "missing_turn", "invalid_span"}


def apply(
    store: NormalizedStore,
    request: MutationRequest,
    transcripts: TranscriptSet | None,
    rng: random.Random,
    mutation_id: str,
) -> MutationApplication:
    """Break one resolvable source reference without violating its own schema."""

    subtype = request.subtype or "missing_transcript"
    if subtype not in SUBTYPES:
        raise MutationPreconditionError(f"unsupported orphaned-provenance subtype: {subtype}")
    if transcripts is None:
        raise MutationPreconditionError("orphaned-provenance mutation requires transcripts")
    target = select_memory(
        store,
        request.target_memory_id,
        rng,
        predicate=lambda memory: _has_resolvable_ref(memory, transcripts),
        requirement="a memory with a resolvable source reference",
    )
    ref_index = next(
        index
        for index, source_ref in enumerate(target.source_refs)
        if _ref_is_resolvable(source_ref, transcripts)
    )
    original_ref = target.source_refs[ref_index]
    mutated_ref = _mutate_ref(original_ref, subtype, transcripts, mutation_id)
    refs = list(target.source_refs)
    refs[ref_index] = mutated_ref
    mutated = validated_memory_copy(target, source_refs=tuple(refs))
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
            "source_ref_index": ref_index,
            "original_source_ref": original_ref.model_dump(mode="json"),
            "mutated_source_ref": mutated_ref.model_dump(mode="json"),
        },
    )


def _mutate_ref(
    source_ref: SourceRef,
    subtype: str,
    transcripts: TranscriptSet,
    mutation_id: str,
) -> SourceRef:
    transcript = transcripts.get(source_ref.transcript_id)
    if transcript is None:
        raise MutationPreconditionError("selected source reference is not resolvable")
    if subtype == "missing_transcript":
        collision_index = 0
        missing_id = deterministic_id(
            "transcript", {"mutation_id": mutation_id, "collision_index": collision_index}
        )
        while transcripts.get(missing_id) is not None:
            collision_index += 1
            missing_id = deterministic_id(
                "transcript",
                {"mutation_id": mutation_id, "collision_index": collision_index},
            )
        return source_ref.model_copy(update={"transcript_id": missing_id})
    if subtype == "missing_turn":
        missing_index = max((turn.index for turn in transcript.turns), default=0) + 1
        return source_ref.model_copy(update={"turn_idx": missing_index})
    if source_ref.turn_idx is None:
        raise MutationPreconditionError("invalid_span requires a source reference with a turn")
    turn = next(turn for turn in transcript.turns if turn.index == source_ref.turn_idx)
    start = len(turn.content) + 1
    return source_ref.model_copy(update={"span": (start, start + 1)})


def _has_resolvable_ref(memory: NormalizedMemory, transcripts: TranscriptSet) -> bool:
    return any(_ref_is_resolvable(source_ref, transcripts) for source_ref in memory.source_refs)


def _ref_is_resolvable(source_ref: SourceRef, transcripts: TranscriptSet) -> bool:
    transcript = transcripts.get(source_ref.transcript_id)
    if transcript is None:
        return False
    if source_ref.turn_idx is None:
        return source_ref.span is None
    turn = next((item for item in transcript.turns if item.index == source_ref.turn_idx), None)
    if turn is None:
        return False
    return source_ref.span is None or source_ref.span[1] <= len(turn.content)
