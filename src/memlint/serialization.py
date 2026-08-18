"""Stable JSON serialization entry points for normalized models."""

from __future__ import annotations

import json
from pathlib import Path

from memlint.models import NormalizedStore, TranscriptSet


def loads_store(text: str) -> NormalizedStore:
    """Validate a normalized store from JSON text."""

    return NormalizedStore.model_validate_json(text)


def load_store(path: str | Path) -> NormalizedStore:
    """Validate a normalized store from a UTF-8 JSON file."""

    return loads_store(Path(path).read_text(encoding="utf-8"))


def dumps_transcripts(transcripts: TranscriptSet, *, indent: int | None = 2) -> str:
    """Serialize transcripts deterministically."""

    text = json.dumps(
        transcripts.model_dump(mode="json"),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )
    return text + ("\n" if indent is not None else "")


def loads_transcripts(text: str) -> TranscriptSet:
    """Validate a transcript set from JSON text."""

    return TranscriptSet.model_validate_json(text)

