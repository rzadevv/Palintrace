"""Memory backend adapters and their public exceptions."""

from palintrace.adapters.base import (
    AdapterAuthenticationError,
    AdapterDataError,
    AdapterDependencyError,
    AdapterError,
    MemoryAdapter,
)
from palintrace.adapters.file import FileAdapter
from palintrace.adapters.graphiti import GraphitiAdapter, normalize_graphiti_record
from palintrace.adapters.letta import LettaAdapter, normalize_letta_record
from palintrace.adapters.mem0 import Mem0Adapter, normalize_mem0_record

__all__ = [
    "AdapterAuthenticationError",
    "AdapterDataError",
    "AdapterDependencyError",
    "AdapterError",
    "FileAdapter",
    "GraphitiAdapter",
    "LettaAdapter",
    "Mem0Adapter",
    "MemoryAdapter",
    "normalize_graphiti_record",
    "normalize_letta_record",
    "normalize_mem0_record",
]
