"""Controlled internal-contradiction mutation."""

from __future__ import annotations

import random

from memlint.models import NormalizedStore, ProvenanceStatus, TranscriptSet
from memlint.mutations.base import (
    MutationApplication,
    MutationPreconditionError,
    append_memories,
    derived_memory_id,
    replace_once,
    require_no_embedding,
    select_memory,
    validated_memory_copy,
)
from memlint.mutations.models import (
    ConflictRelation,
    MutationRequest,
    MutationTarget,
    MutationTargetRole,
)

SUBTYPE = "controlled_conflict"


def apply(
    store: NormalizedStore,
    request: MutationRequest,
    transcripts: TranscriptSet | None,
    rng: random.Random,
    mutation_id: str,
) -> MutationApplication:
    """Insert a current same-scope record with an incompatible controlled value."""

    del transcripts
    subtype = request.subtype or SUBTYPE
    if subtype != SUBTYPE:
        raise MutationPreconditionError(f"unsupported contradiction subtype: {subtype}")
    if request.conflict_relation is not ConflictRelation.EXCLUSIVE_VALUE:
        raise MutationPreconditionError(
            "controlled contradiction requires conflict_relation='exclusive_value'"
        )
    target = select_memory(
        store,
        request.target_memory_id,
        rng,
        predicate=lambda memory: memory.active is True,
        requirement="an explicitly active memory",
    )
    require_no_embedding(target)
    conflicting_content = replace_once(target.content, request.replace_from, request.replace_to)
    conflicting_id = derived_memory_id(mutation_id, "conflicting")
    conflicting = validated_memory_copy(
        target,
        id=conflicting_id,
        content=conflicting_content,
        created_at=None,
        updated_at=None,
        source_refs=(),
        provenance_status=ProvenanceStatus.UNAVAILABLE,
        active=True,
        supersedes=(),
        embedding=None,
    )
    return MutationApplication(
        store=append_memories(store, (conflicting,)),
        subtype=subtype,
        target_memory_ids=(target.id, conflicting_id),
        targets=(
            MutationTarget(
                memory_id=target.id,
                role=MutationTargetRole.PRIMARY,
            ),
            MutationTarget(
                memory_id=conflicting_id,
                role=MutationTargetRole.CONFLICTING,
            ),
        ),
        created_memory_ids=(conflicting_id,),
        parameters={
            "original_content": target.content,
            "conflicting_content": conflicting_content,
            "replace_from": request.replace_from,
            "replace_to": request.replace_to,
            "semantic_relation": request.conflict_relation.value,
        },
    )
