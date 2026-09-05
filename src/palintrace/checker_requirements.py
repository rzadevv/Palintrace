"""External input requirements for the built-in public checkers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

PUBLIC_CHECKER_IDS = (
    "orphaned_provenance",
    "redundancy_bloat",
    "stale_active",
    "privacy_scope_violation",
    "unsupported_claim",
)


@dataclass(frozen=True)
class _CheckerRequirement:
    requires_transcripts: bool = False
    requires_scope_policy: bool = False
    requires_semantic_judge: bool = False


_CHECKER_REQUIREMENTS: Mapping[str, _CheckerRequirement] = MappingProxyType(
    {
        "orphaned_provenance": _CheckerRequirement(requires_transcripts=True),
        "redundancy_bloat": _CheckerRequirement(),
        "stale_active": _CheckerRequirement(),
        "privacy_scope_violation": _CheckerRequirement(requires_scope_policy=True),
        "unsupported_claim": _CheckerRequirement(
            requires_transcripts=True,
            requires_semantic_judge=True,
        ),
    }
)
