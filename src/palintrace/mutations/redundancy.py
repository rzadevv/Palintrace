"""Deterministic redundancy/bloat mutations."""

from __future__ import annotations

import random

from palintrace.models import NormalizedStore, TranscriptSet
from palintrace.mutations.base import (
    MutationApplication,
    MutationPreconditionError,
    append_memories,
    derived_memory_id,
    select_memory,
    validated_memory_copy,
)
from palintrace.mutations.models import MutationRequest, MutationTarget, MutationTargetRole

SUBTYPE = "exact_duplicate"


def apply(
    store: NormalizedStore,
    request: MutationRequest,
    transcripts: TranscriptSet | None,
    rng: random.Random,
    mutation_id: str,
) -> MutationApplication:
    """Duplicate a record in the same scope under a new deterministic ID."""

    del transcripts
    subtype = request.subtype or SUBTYPE
    if subtype != SUBTYPE:
        raise MutationPreconditionError(f"unsupported redundancy subtype: {subtype}")
    target = select_memory(store, request.target_memory_id, rng)
    duplicate_id = derived_memory_id(mutation_id, "duplicate")
    duplicate = validated_memory_copy(target, id=duplicate_id)
    return MutationApplication(
        store=append_memories(store, (duplicate,)),
        subtype=subtype,
        target_memory_ids=(target.id, duplicate_id),
        targets=(
            MutationTarget(
                memory_id=target.id,
                role=MutationTargetRole.PRIMARY,
            ),
            MutationTarget(
                memory_id=duplicate_id,
                role=MutationTargetRole.DUPLICATE,
            ),
        ),
        created_memory_ids=(duplicate_id,),
        parameters={"original_memory_id": target.id, "duplicate_memory_id": duplicate_id},
    )
