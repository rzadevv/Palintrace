"""Central deterministic mutation dispatcher."""

from __future__ import annotations

import random
from collections.abc import Callable

from palintrace.models import NormalizedStore, TranscriptSet
from palintrace.mutations import (
    contradiction,
    injection,
    provenance,
    redundancy,
    scope,
    shadowing,
    stale,
    unsupported,
)
from palintrace.mutations.base import (
    MutationApplication,
    deterministic_id,
    semantic_store_digest,
    transcript_set_digest,
)
from palintrace.mutations.models import (
    GOLD_LABEL_UNIT_BY_DEFECT,
    GoldLabel,
    MutationManifest,
    MutationRequest,
    MutationResult,
)
from palintrace.taxonomy import TAXONOMY_VERSION, DefectClass

MutationHandler = Callable[
    [NormalizedStore, MutationRequest, TranscriptSet | None, random.Random, str],
    MutationApplication,
]

HANDLERS: dict[DefectClass, MutationHandler] = {
    DefectClass.UNSUPPORTED_CLAIM: unsupported.apply,
    DefectClass.INTERNAL_CONTRADICTION: contradiction.apply,
    DefectClass.STALE_ACTIVE: stale.apply,
    DefectClass.ORPHANED_PROVENANCE: provenance.apply,
    DefectClass.RETRIEVAL_SHADOWING: shadowing.apply,
    DefectClass.INJECTED_INSTRUCTION: injection.apply,
    DefectClass.PRIVACY_SCOPE_VIOLATION: scope.apply,
    DefectClass.REDUNDANCY_BLOAT: redundancy.apply,
}


def mutate(
    store: NormalizedStore,
    request: MutationRequest,
    transcripts: TranscriptSet | None = None,
) -> MutationResult:
    """Apply one controlled mutation and return the store and separate gold manifest."""

    base_digest = semantic_store_digest(store)
    transcript_digest = transcript_set_digest(transcripts)
    request_payload = request.model_dump(mode="json")
    mutation_id = deterministic_id(
        "mutation",
        {
            "base_store_digest": base_digest,
            "request": request_payload,
            "taxonomy_version": TAXONOMY_VERSION,
            "transcript_digest": transcript_digest,
        },
    )
    rng = random.Random(request.seed)
    application = HANDLERS[request.defect_class](
        store, request, transcripts, rng, mutation_id
    )
    mutated_digest = semantic_store_digest(application.store)
    manifest = MutationManifest(
        mutation_id=mutation_id,
        defect_class=request.defect_class,
        subtype=application.subtype,
        seed=request.seed,
        base_store_digest=base_digest,
        mutated_store_digest=mutated_digest,
        transcript_digest=transcript_digest,
        target_memory_ids=application.target_memory_ids,
        targets=application.targets,
        gold_label=GoldLabel(
            unit=GOLD_LABEL_UNIT_BY_DEFECT[request.defect_class],
            memory_ids=application.target_memory_ids,
            observed_positive=not application.requires_runtime_validation,
        ),
        created_memory_ids=application.created_memory_ids,
        modified_memory_ids=application.modified_memory_ids,
        removed_memory_ids=application.removed_memory_ids,
        parameters=application.parameters or {},
        requires_runtime_validation=application.requires_runtime_validation,
        base_store_status=request.base_store_status,
        retrieval_probe=application.retrieval_probe,
    )
    return MutationResult(mutated_store=application.store, manifest=manifest)
