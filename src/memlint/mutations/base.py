"""Shared deterministic mutation primitives."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue

from memlint.models import NormalizedMemory, NormalizedStore, TranscriptSet
from memlint.mutations.models import MutationTarget, RetrievalProbe


class MutationError(ValueError):
    """Base error for a mutation request that cannot be completed."""


class MutationPreconditionError(MutationError):
    """The normalized inputs cannot support the requested controlled mutation."""


@dataclass(frozen=True)
class MutationApplication:
    """Internal result used by the engine to build the public manifest."""

    store: NormalizedStore
    subtype: str
    target_memory_ids: tuple[str, ...]
    targets: tuple[MutationTarget, ...]
    created_memory_ids: tuple[str, ...] = ()
    modified_memory_ids: tuple[str, ...] = ()
    removed_memory_ids: tuple[str, ...] = ()
    parameters: dict[str, JsonValue] | None = None
    requires_runtime_validation: bool = False
    retrieval_probe: RetrievalProbe | None = None


def canonical_json(value: object) -> str:
    """Return compact canonical JSON for hashing deterministic inputs."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def semantic_store_digest(store: NormalizedStore) -> str:
    """Fingerprint portable store semantics, excluding raw and export time."""

    payload = {
        "adapter": store.adapter,
        "memories": sorted(
            (memory.semantic_dict() for memory in store.memories), key=lambda item: str(item["id"])
        ),
        "schema_version": store.schema_version,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def transcript_set_digest(transcripts: TranscriptSet | None) -> str | None:
    """Fingerprint transcript inputs in canonical transcript and turn order."""

    if transcripts is None:
        return None
    payload = transcripts.model_dump(mode="json")
    payload["transcripts"] = sorted(
        payload["transcripts"], key=lambda transcript: str(transcript["id"])
    )
    for transcript in payload["transcripts"]:
        transcript["turns"] = sorted(transcript["turns"], key=lambda turn: int(turn["index"]))
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def deterministic_id(namespace: str, payload: object) -> str:
    """Create a stable identifier without random UUIDs or process-local hashes."""

    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:24]
    return f"{namespace}-{digest}"


def derived_memory_id(mutation_id: str, role: str, index: int = 0) -> str:
    """Create an opaque stable memory ID without revealing mutation semantics."""

    return deterministic_id("mem", {"mutation_id": mutation_id, "role": role, "index": index})


def require_no_embedding(memory: NormalizedMemory) -> None:
    """Reject content mutation when its stored embedding cannot be regenerated."""

    if memory.embedding is not None:
        raise MutationPreconditionError(
            "content cannot be mutated safely without regenerating the embedding"
        )


def select_memory(
    store: NormalizedStore,
    target_memory_id: str | None,
    rng: random.Random,
    *,
    predicate: Callable[[NormalizedMemory], bool] | None = None,
    requirement: str = "an eligible memory",
) -> NormalizedMemory:
    """Select an explicit target or reproducibly choose from sorted candidates."""

    if target_memory_id is not None:
        memory = store.get(target_memory_id)
        if memory is None:
            raise MutationPreconditionError(f"target memory does not exist: {target_memory_id}")
        if predicate is not None and not predicate(memory):
            raise MutationPreconditionError(f"target memory does not satisfy {requirement}")
        return memory
    candidates = sorted(
        (memory for memory in store.memories if predicate is None or predicate(memory)),
        key=lambda memory: memory.id,
    )
    if not candidates:
        raise MutationPreconditionError(f"store does not contain {requirement}")
    return rng.choice(candidates)


def replace_once(content: str, replace_from: str | None, replace_to: str | None) -> str:
    """Apply one explicit, unambiguous textual substitution."""

    if replace_from is None or replace_to is None:
        raise MutationPreconditionError(
            "controlled substitution requires replace_from and replace_to"
        )
    if replace_from == replace_to:
        raise MutationPreconditionError("replacement values must differ")
    occurrences = content.count(replace_from)
    if occurrences != 1:
        raise MutationPreconditionError(
            f"replace_from must occur exactly once in target content; found {occurrences}"
        )
    return content.replace(replace_from, replace_to, 1)


def validated_memory_copy(memory: NormalizedMemory, **changes: Any) -> NormalizedMemory:
    """Copy a record and revalidate all normalized invariants."""

    payload = memory.model_dump()
    payload.update(changes)
    return NormalizedMemory.model_validate(payload)


def rebuilt_store(store: NormalizedStore, memories: Sequence[NormalizedMemory]) -> NormalizedStore:
    """Build and validate a new store while preserving the normalized store envelope."""

    return NormalizedStore(
        schema_version=store.schema_version,
        adapter=store.adapter,
        exported_at=store.exported_at,
        memories=tuple(memories),
    )


def replace_memory(
    store: NormalizedStore, memory_id: str, replacement: NormalizedMemory
) -> NormalizedStore:
    """Return a new store with one record replaced in its original position."""

    memories = tuple(replacement if memory.id == memory_id else memory for memory in store.memories)
    return rebuilt_store(store, memories)


def append_memories(
    store: NormalizedStore, additions: Sequence[NormalizedMemory]
) -> NormalizedStore:
    """Return a new store with validated, ordered additions."""

    return rebuilt_store(store, (*store.memories, *additions))
