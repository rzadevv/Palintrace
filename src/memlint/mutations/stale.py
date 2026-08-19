"""Controlled stale-active mutation."""

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
from memlint.mutations.models import MutationRequest, MutationTarget, MutationTargetRole

SUBTYPE = "explicit_supersession"


def apply(
    store: NormalizedStore,
    request: MutationRequest,
    transcripts: TranscriptSet | None,
    rng: random.Random,
    mutation_id: str,
) -> MutationApplication:
    """Insert an explicit replacement while deliberately leaving the old record active."""

    del transcripts
    subtype = request.subtype or SUBTYPE
    if subtype != SUBTYPE:
        raise MutationPreconditionError(f"unsupported stale-active subtype: {subtype}")
    target = select_memory(
        store,
        request.target_memory_id,
        rng,
        predicate=lambda memory: memory.active is True,
        requirement="an explicitly active memory",
    )
    require_no_embedding(target)
    replacement_content = replace_once(target.content, request.replace_from, request.replace_to)
    replacement_id = derived_memory_id(mutation_id, "superseding")
    superseding = validated_memory_copy(
        target,
        id=replacement_id,
        content=replacement_content,
        created_at=None,
        updated_at=None,
        source_refs=(),
        provenance_status=ProvenanceStatus.UNAVAILABLE,
        active=True,
        supersedes=(target.id,),
        embedding=None,
    )
    return MutationApplication(
        store=append_memories(store, (superseding,)),
        subtype=subtype,
        target_memory_ids=(target.id,),
        targets=(
            MutationTarget(
                memory_id=target.id,
                role=MutationTargetRole.PRIMARY,
            ),
            MutationTarget(
                memory_id=replacement_id,
                role=MutationTargetRole.SUPERSEDING,
            ),
        ),
        created_memory_ids=(replacement_id,),
        parameters={
            "obsolete_content": target.content,
            "replacement_content": replacement_content,
            "superseding_memory_id": replacement_id,
            "replace_from": request.replace_from,
            "replace_to": request.replace_to,
        },
    )
