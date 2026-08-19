"""Objective cross-principal scope mutations."""

from __future__ import annotations

import random

from memlint.models import NormalizedStore, TranscriptSet
from memlint.mutations.base import (
    MutationApplication,
    MutationPreconditionError,
    append_memories,
    derived_memory_id,
    select_memory,
    validated_memory_copy,
)
from memlint.mutations.models import MutationRequest, MutationTarget, MutationTargetRole

SUBTYPES = {"cross_user_copy", "cross_agent_copy"}


def apply(
    store: NormalizedStore,
    request: MutationRequest,
    transcripts: TranscriptSet | None,
    rng: random.Random,
    mutation_id: str,
) -> MutationApplication:
    """Copy a memory into a different explicit user or agent scope."""

    del transcripts
    subtype = request.subtype or "cross_user_copy"
    if subtype not in SUBTYPES:
        raise MutationPreconditionError(f"unsupported privacy/scope subtype: {subtype}")
    if subtype == "cross_user_copy":
        destination = request.destination_user_id
        if destination is None:
            raise MutationPreconditionError("cross_user_copy requires destination_user_id")
        target = select_memory(
            store,
            request.target_memory_id,
            rng,
            predicate=lambda memory: memory.scope.user_id is not None
            and memory.scope.user_id != destination,
            requirement="a memory in a different known user scope",
        )
        destination_scope = target.scope.model_copy(update={"user_id": destination})
        source_scope = target.scope.user_id
        scope_field = "user_id"
    else:
        destination = request.destination_agent_id
        if destination is None:
            raise MutationPreconditionError("cross_agent_copy requires destination_agent_id")
        target = select_memory(
            store,
            request.target_memory_id,
            rng,
            predicate=lambda memory: memory.scope.agent_id is not None
            and memory.scope.agent_id != destination,
            requirement="a memory in a different known agent scope",
        )
        destination_scope = target.scope.model_copy(update={"agent_id": destination})
        source_scope = target.scope.agent_id
        scope_field = "agent_id"
    copied_id = derived_memory_id(mutation_id, subtype)
    copied = validated_memory_copy(target, id=copied_id, scope=destination_scope)
    return MutationApplication(
        store=append_memories(store, (copied,)),
        subtype=subtype,
        target_memory_ids=(copied_id,),
        targets=(
            MutationTarget(
                memory_id=copied_id,
                role=MutationTargetRole.PRIMARY,
                receives_gold_label=True,
            ),
            MutationTarget(
                memory_id=target.id,
                role=MutationTargetRole.SOURCE,
                receives_gold_label=False,
            ),
        ),
        created_memory_ids=(copied_id,),
        parameters={
            "source_memory_id": target.id,
            "scope_field": scope_field,
            "source_scope": source_scope,
            "incorrect_destination_scope": destination,
        },
    )
