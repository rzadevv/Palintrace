"""Defect class labels for memory-store findings and mutations."""

from enum import StrEnum

TAXONOMY_VERSION = "1.1"


class DefectClass(StrEnum):
    """Defect classes that checkers report and mutations inject."""

    UNSUPPORTED_CLAIM = "unsupported_claim"
    INTERNAL_CONTRADICTION = "internal_contradiction"
    STALE_ACTIVE = "stale_active"
    ORPHANED_PROVENANCE = "orphaned_provenance"
    RETRIEVAL_SHADOWING = "retrieval_shadowing"
    INJECTED_INSTRUCTION = "injected_instruction"
    PRIVACY_SCOPE_VIOLATION = "privacy_scope_violation"
    REDUNDANCY_BLOAT = "redundancy_bloat"
