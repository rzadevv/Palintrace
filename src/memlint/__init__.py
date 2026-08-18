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

__all__ = [
    "MemoryScope",
    "NormalizedMemory",
    "NormalizedStore",
    "ProvenanceStatus",
    "SourceRef",
    "Transcript",
    "TranscriptSet",
    "TranscriptTurn",
]

__version__ = "0.1.0"
