"""Small backend-independent checker protocol and deterministic ID helpers."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Sequence
from typing import Protocol

from palintrace.checkers.models import CheckerResult, EvidenceItem, _evidence_identity
from palintrace.models import NormalizedStore, TranscriptSet
from palintrace.taxonomy import DefectClass


def normalized_content(content: str) -> str:
    """Fold memory content for duplicate and replica matching."""

    collapsed = " ".join(unicodedata.normalize("NFKC", content).casefold().split())
    folded = collapsed
    # ignore trailing punctuation when matching
    while folded and unicodedata.category(folded[-1]).startswith("P"):
        folded = folded[:-1].rstrip()
    # content that is only punctuation stays distinguishable instead of folding to ""
    return folded or collapsed


class CheckerError(ValueError):
    """Base error raised by a checker that cannot complete its audit."""


class CheckerInputError(CheckerError):
    """Required normalized checker input was not supplied."""


class Checker(Protocol):
    """Stable interface implemented by normalized-data checkers."""

    checker_id: str
    checker_version: str
    defect_class: DefectClass

    def check(
        self,
        store: NormalizedStore,
        *,
        transcripts: TranscriptSet | None = None,
    ) -> CheckerResult:
        """Audit normalized inputs and return a deterministic result."""


def deterministic_finding_id(
    *,
    checker_id: str,
    checker_version: str,
    defect_class: DefectClass,
    memory_ids: Sequence[str],
    evidence: Sequence[EvidenceItem],
) -> str:
    """Build an opaque finding ID from stable semantic inputs."""

    evidence_identities = [_evidence_identity(item) for item in evidence]
    evidence_identities.sort(
        key=lambda item: json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    payload = {
        "checker_id": checker_id,
        "checker_version": checker_version,
        "defect_class": defect_class.value,
        "evidence": evidence_identities,
        "memory_ids": sorted(memory_ids),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:24]
    return f"finding-{digest}"
