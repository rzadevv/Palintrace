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
    select_memory,
    validated_memory_copy,
)
from memlint.mutations.models import MutationRequest, MutationTarget, MutationTargetRole

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
    target = select_memory(
        store,
        request.target_memory_id,
        rng,
        predicate=lambda memory: memory.active is True,
        requirement="an explicitly active memory",
    )
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
                receives_gold_label=True,
            ),
            MutationTarget(
                memory_id=conflicting_id,
                role=MutationTargetRole.CONFLICTING,
                receives_gold_label=True,
            ),
        ),
        created_memory_ids=(conflicting_id,),
        parameters={
            "original_content": target.content,
            "conflicting_content": conflicting_content,
            "replace_from": request.replace_from,
            "replace_to": request.replace_to,
        },
    )
