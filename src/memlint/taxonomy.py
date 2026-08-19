"""Frozen research labels for memory-store defects."""

from enum import StrEnum

TAXONOMY_VERSION = "1.0"


class DefectClass(StrEnum):
    """The eight defect classes frozen for taxonomy version 1.0."""

    UNSUPPORTED_CLAIM = "unsupported_claim"
    INTERNAL_CONTRADICTION = "internal_contradiction"
    STALE_ACTIVE = "stale_active"
    ORPHANED_PROVENANCE = "orphaned_provenance"
    RETRIEVAL_SHADOWING = "retrieval_shadowing"
    INJECTED_INSTRUCTION = "injected_instruction"
    PRIVACY_SCOPE_VIOLATION = "privacy_scope_violation"
    REDUNDANCY_BLOAT = "redundancy_bloat"
