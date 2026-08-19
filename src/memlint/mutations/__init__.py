"""Deterministic defect mutation API."""

from memlint.mutations.base import MutationError, MutationPreconditionError
from memlint.mutations.engine import mutate
from memlint.mutations.models import (
    BaseStoreStatus,
    MutationManifest,
    MutationRequest,
    MutationResult,
    MutationTarget,
    MutationTargetRole,
    RetrievalProbe,
)

__all__ = [
    "BaseStoreStatus",
    "MutationError",
    "MutationManifest",
    "MutationPreconditionError",
    "MutationRequest",
    "MutationResult",
    "MutationTarget",
    "MutationTargetRole",
    "RetrievalProbe",
    "mutate",
]
