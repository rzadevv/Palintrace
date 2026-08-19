"""Small backend-independent checker protocol and deterministic ID helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

from memlint.checkers.models import CheckerResult, EvidenceItem
from memlint.models import NormalizedStore, TranscriptSet
from memlint.taxonomy import DefectClass


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

    payload = {
        "checker_id": checker_id,
        "checker_version": checker_version,
        "defect_class": defect_class.value,
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "memory_ids": list(memory_ids),
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
