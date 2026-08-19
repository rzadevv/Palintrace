"""Retrieval-shadowing challenge generation without a retriever."""

from __future__ import annotations

import random

from memlint.models import NormalizedMemory, NormalizedStore, TranscriptSet
from memlint.mutations.base import (
    MutationApplication,
    MutationPreconditionError,
    append_memories,
    derived_memory_id,
    select_memory,
)
from memlint.mutations.models import (
    DistractorFamily,
    MutationRequest,
    MutationTarget,
    MutationTargetRole,
    RetrievalProbe,
)

SUBTYPE = "distractor_crowding"
DISTRACTOR_TEMPLATES = (
    "User has configured syntax highlighting in Visual Studio Code.",
    "User uses Emacs keybindings in a terminal application.",
    "User installed an extension for JetBrains IDEs.",
    "User compared editor themes for Sublime Text.",
    "User keeps software configuration notes for several editors.",
)


def apply(
    store: NormalizedStore,
    request: MutationRequest,
    transcripts: TranscriptSet | None,
    rng: random.Random,
    mutation_id: str,
) -> MutationApplication:
    """Add fixed same-scope distractors and emit a probe for a later retriever."""

    del transcripts
    subtype = request.subtype or SUBTYPE
    if subtype != SUBTYPE:
        raise MutationPreconditionError(f"unsupported retrieval-shadowing subtype: {subtype}")
    if request.query is None:
        raise MutationPreconditionError("distractor_crowding requires an explicit query")
    if request.distractor_family is not DistractorFamily.EDITOR:
        raise MutationPreconditionError(
            "distractor_crowding requires distractor_family='editor'"
        )
    target = select_memory(store, request.target_memory_id, rng)
    distractors = tuple(
        NormalizedMemory(
            id=derived_memory_id(mutation_id, "distractor", index),
            content=DISTRACTOR_TEMPLATES[index],
            scope=target.scope,
            active=True,
        )
        for index in range(request.distractor_count)
    )
    distractor_ids = tuple(memory.id for memory in distractors)
    probe = RetrievalProbe(
        query=request.query,
        expected_memory_ids=(target.id,),
        distractor_memory_ids=distractor_ids,
    )
    return MutationApplication(
        store=append_memories(store, distractors),
        subtype=subtype,
        target_memory_ids=(target.id,),
        targets=(
            MutationTarget(
                memory_id=target.id,
                role=MutationTargetRole.PRIMARY,
            ),
        ),
        created_memory_ids=distractor_ids,
        parameters={
            "challenge_type": subtype,
            "distractor_count": request.distractor_count,
            "distractor_family": request.distractor_family.value,
        },
        requires_runtime_validation=True,
        retrieval_probe=probe,
    )
