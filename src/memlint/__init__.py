"""Public, backend-independent MemLint foundation API."""

from memlint.models import (
    MemoryScope,
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    Transcript,
    TranscriptSet,
    TranscriptTurn,
)
from memlint.mutations import (
    BaseStoreStatus,
    MutationError,
    MutationManifest,
    MutationPreconditionError,
    MutationRequest,
    MutationResult,
    RetrievalProbe,
    mutate,
)
from memlint.taxonomy import TAXONOMY_VERSION, DefectClass

__all__ = [
    "MemoryScope",
    "BaseStoreStatus",
    "DefectClass",
    "MutationError",
    "MutationManifest",
    "MutationPreconditionError",
    "MutationRequest",
    "MutationResult",
    "NormalizedMemory",
    "NormalizedStore",
    "ProvenanceStatus",
    "SourceRef",
    "Transcript",
    "TranscriptSet",
    "TranscriptTurn",
    "RetrievalProbe",
    "TAXONOMY_VERSION",
    "mutate",
]

__version__ = "0.1.0"
